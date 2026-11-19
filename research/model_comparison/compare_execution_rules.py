"""Compare fixed-horizon and signal-replacement execution rules."""
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


def summarize_execution_rules(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "train_days", "cost_rate", "exit_mode", "total_return_net",
        "max_drawdown", "n_trades",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"执行规则明细缺少字段: {sorted(missing)}")
    return detail.groupby(
        ["train_days", "cost_rate", "exit_mode"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        mean_trades=("n_trades", "mean"),
        mean_turnover=("turnover", "mean") if "turnover" in detail.columns else ("n_trades", "mean"),
    )


def run(
    train_days: tuple[int, ...] = (500, 750, 1000),
    top_k: int = 2,
    cost_rates: tuple[float, ...] = (0.003, 0.005),
    exit_modes: tuple[str, ...] = ("fixed", "signal"),
    hold_days: int = 10,
    report_root: str = "output/research/model_comparison",
    output_dir: str = "output/research/reports",
) -> pd.DataFrame:
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0,
        hold_days=hold_days, date_mode="union", extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    signals = build_factor_only_signals(
        panel, top_k=top_k, mode="common", benchmark_frame=benchmark
    )
    rows = []
    for days in train_days:
        report_path = ROOT / report_root / f"factor_comparison_{days}" / "per_window.csv"
        if not report_path.exists():
            raise FileNotFoundError(f"缺少共同窗口报告: {report_path}")
        windows = pd.read_csv(report_path)[["window", "test_start", "test_end"]].drop_duplicates()
        for cost_rate in cost_rates:
            for exit_mode in exit_modes:
                for _, window in windows.iterrows():
                    scoped = signals[
                        (signals["trade_date"] >= int(window["test_start"]))
                        & (signals["trade_date"] <= int(window["test_end"]))
                    ]
                    report = PortfolioBacktest(
                        cost_rate=cost_rate, hold_days=hold_days,
                        label_threshold=0.0, exit_mode=exit_mode,
                    ).run(prices, scoped, end_date=int(window["test_end"]))
                    rows.append({
                        "train_days": days, "cost_rate": cost_rate,
                        "exit_mode": exit_mode, "window": int(window["window"]),
                        "total_return_net": report.total_return_net,
                        "max_drawdown": report.max_drawdown,
                        "n_trades": report.n_trades,
                        "turnover": report.turnover,
                    })
    detail = pd.DataFrame(rows)
    summary = summarize_execution_rules(detail)
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "factor_common_execution_rules_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "factor_common_execution_rules_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", default="500,750,1000")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rates", default="0.003,0.005")
    parser.add_argument("--exit-modes", default="fixed,signal")
    parser.add_argument("--hold-days", type=int, default=10)
    args = parser.parse_args()
    run(
        tuple(int(value) for value in args.train_days.split(",")),
        top_k=args.top_k,
        cost_rates=tuple(float(value) for value in args.cost_rates.split(",")),
        exit_modes=tuple(args.exit_modes.split(",")),
        hold_days=args.hold_days,
    )


if __name__ == "__main__":
    main()
