"""Run Factor-common Top-K, holding-period, and cost sensitivity on long holdouts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.factor_baseline import build_factor_only_signals
from model_and_train.pipeline import _load_panel_frames, build_labeled_panel
from model_and_train.portfolio_backtest import PortfolioBacktest


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def compare(
    report_path: Path,
    output_dir: Path,
    top_ks: tuple[int, ...] = (1, 2, 3, 5),
    hold_days: tuple[int, ...] = (5, 10, 20),
    cost_rates: tuple[float, ...] = (0.003, 0.005),
) -> pd.DataFrame:
    report = pd.read_csv(report_path)
    required = {"window", "test_start", "test_end"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"滚动报告缺少字段: {sorted(missing)}")

    if any(value <= 0 for value in top_ks + hold_days) or any(value < 0 for value in cost_rates):
        raise ValueError("Top-K、持有期和成本率必须合法")

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=0.0,
        hold_days=max(hold_days),
        date_mode="union",
        extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]

    signal_sets = {
        top_k: build_factor_only_signals(
            panel, top_k=top_k, mode="common", benchmark_frame=benchmark
        )
        for top_k in top_ks
    }
    rows = []
    for _, window_row in report.iterrows():
        test_start = int(window_row["test_start"])
        test_end = int(window_row["test_end"])
        window = int(window_row["window"])
        for top_k, signals in signal_sets.items():
            scoped = signals[
                (signals["trade_date"] >= test_start)
                & (signals["trade_date"] <= test_end)
            ]
            for hold in hold_days:
                for cost_rate in cost_rates:
                    bt = PortfolioBacktest(
                        cost_rate=cost_rate,
                        hold_days=hold,
                        label_threshold=0.0,
                    ).run(prices, scoped, end_date=test_end)
                    rows.append({
                        "window": window,
                        "test_start": test_start,
                        "test_end": test_end,
                        "market_regime": window_row.get("market_regime", "unknown"),
                        "top_k": top_k,
                        "hold_days": hold,
                        "cost_rate": cost_rate,
                        "total_return_net": bt.total_return_net,
                        "max_drawdown": bt.max_drawdown,
                        "n_trades": bt.n_trades,
                    })

    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["top_k", "hold_days", "cost_rate"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        total_net_return=("total_return_net", lambda values: float((1.0 + values).prod() - 1.0)),
    )
    by_regime = detail.groupby(
        ["market_regime", "top_k", "hold_days", "cost_rate"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "per_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    by_regime.to_csv(output_dir / "by_regime.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--hold-days", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--cost-rates", nargs="+", type=float, default=[0.003, 0.005])
    args = parser.parse_args()
    compare(
        args.report,
        args.output_dir,
        top_ks=tuple(args.top_ks),
        hold_days=tuple(args.hold_days),
        cost_rates=tuple(args.cost_rates),
    )


if __name__ == "__main__":
    main()
