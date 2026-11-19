"""Replay Factor-Only rankings on common rolling OOS windows."""
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


def _parse_number_list(value: str, converter):
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError("列表参数不能包含空项")
    return [converter(item) for item in items]


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def run(
    train_days: tuple[int, ...] = (500, 750, 1000),
    top_ks: tuple[int, ...] = (1, 2, 3, 5),
    cost_rates: tuple[float, ...] = (0.003, 0.005),
    report_root: str = "output/research/model_comparison",
    output_dir: str = "output/research/model_comparison/factor_topk_rolling",
) -> pd.DataFrame:
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=0.0,
        hold_days=10,
        date_mode="union",
        extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    signals = {
        (mode, top_k): build_factor_only_signals(
            panel, top_k=top_k, mode=mode, benchmark_frame=benchmark
        )
        for mode in sorted(FACTOR_MODES)
        for top_k in top_ks
    }

    rows = []
    for window_days in train_days:
        report_path = ROOT / report_root / f"factor_comparison_{window_days}" / "per_window.csv"
        if not report_path.exists():
            raise FileNotFoundError(f"缺少共同窗口报告: {report_path}")
        windows = pd.read_csv(report_path)[["window", "test_start", "test_end"]].drop_duplicates()
        for top_k in top_ks:
            for mode in sorted(FACTOR_MODES):
                source = signals[(mode, top_k)]
                for cost_rate in cost_rates:
                    for _, window in windows.iterrows():
                        scoped = source[
                            (source["trade_date"] >= int(window["test_start"]))
                            & (source["trade_date"] <= int(window["test_end"]))
                        ]
                        report = PortfolioBacktest(
                            cost_rate=cost_rate,
                            hold_days=10,
                            label_threshold=0.0,
                            exit_mode="fixed",
                        ).run(prices, scoped, end_date=int(window["test_end"]))
                        rows.append({
                            "train_days": window_days,
                            "top_k": top_k,
                            "cost_rate": cost_rate,
                            "factor_mode": mode,
                            "window": int(window["window"]),
                            "test_start": int(window["test_start"]),
                            "test_end": int(window["test_end"]),
                            "total_return_net": report.total_return_net,
                            "max_drawdown": report.max_drawdown,
                            "n_trades": report.n_trades,
                        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["train_days", "top_k", "cost_rate", "factor_mode"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        total_net_return=("total_return_net", lambda values: float((1.0 + values).prod() - 1.0)),
    )
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "per_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"saved: {output / 'summary.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", default="500,750,1000")
    parser.add_argument("--top-ks", default="1,2,3,5")
    parser.add_argument("--cost-rates", default="0.003,0.005")
    parser.add_argument("--report-root", default="output/research/model_comparison")
    parser.add_argument("--output-dir", default="output/research/model_comparison/factor_topk_rolling")
    args = parser.parse_args()
    run(
        tuple(_parse_number_list(args.train_days, int)),
        tuple(_parse_number_list(args.top_ks, int)),
        tuple(_parse_number_list(args.cost_rates, float)),
        args.report_root,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
