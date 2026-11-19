"""Compare traditional factor modes on long holdout windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.factor_baseline import FACTOR_MODES, build_factor_only_signals
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
    modes: tuple[str, ...] = ("common", "roc", "trend", "ma_energy"),
    top_k: int = 2,
    hold_days: int = 20,
    cost_rates: tuple[float, ...] = (0.003, 0.005),
) -> pd.DataFrame:
    report = pd.read_csv(report_path)
    required = {"window", "test_start", "test_end"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"滚动报告缺少字段: {sorted(missing)}")
    invalid = set(modes) - FACTOR_MODES
    if invalid:
        raise ValueError(f"未知因子模式: {sorted(invalid)}")

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=0.0,
        hold_days=hold_days,
        date_mode="union",
        extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    signal_sets = {
        mode: build_factor_only_signals(
            panel, top_k=top_k, mode=mode, benchmark_frame=benchmark
        )
        for mode in modes
    }

    rows = []
    for _, window_row in report.iterrows():
        test_start = int(window_row["test_start"])
        test_end = int(window_row["test_end"])
        window = int(window_row["window"])
        regime = window_row.get("market_regime", "unknown")
        for mode, signals in signal_sets.items():
            scoped = signals[
                (signals["trade_date"] >= test_start)
                & (signals["trade_date"] <= test_end)
            ]
            for cost_rate in cost_rates:
                bt = PortfolioBacktest(
                    cost_rate=cost_rate,
                    hold_days=hold_days,
                    label_threshold=0.0,
                ).run(prices, scoped, end_date=test_end)
                rows.append({
                    "window": window,
                    "test_start": test_start,
                    "test_end": test_end,
                    "market_regime": regime,
                    "factor_mode": mode,
                    "top_k": top_k,
                    "hold_days": hold_days,
                    "cost_rate": cost_rate,
                    "total_return_net": bt.total_return_net,
                    "max_drawdown": bt.max_drawdown,
                    "n_trades": bt.n_trades,
                })

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["factor_mode", "cost_rate"], as_index=False).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        total_net_return=("total_return_net", lambda values: float((1.0 + values).prod() - 1.0)),
    )
    by_regime = detail.groupby(
        ["market_regime", "factor_mode", "cost_rate"], as_index=False
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
    parser.add_argument("--modes", nargs="+", default=["common", "roc", "trend", "ma_energy"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--cost-rates", nargs="+", type=float, default=[0.003, 0.005])
    args = parser.parse_args()
    compare(
        args.report,
        args.output_dir,
        modes=tuple(args.modes),
        top_k=args.top_k,
        hold_days=args.hold_days,
        cost_rates=tuple(args.cost_rates),
    )


if __name__ == "__main__":
    main()
