"""Run interpretable factor-group baselines and ablations on quotes2."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from data.panel import load_universe_frames
from data.panel_pipeline import build_labeled_panel
from model_and_train.factor_library import add_common_factors
from model_and_train.portfolio_backtest import PortfolioBacktest
from model_and_train.ranking import rank_and_select
from research.model_comparison.run_quotes2_50etf_time_split import make_calendar_window


BENCHMARK = "510300.SH"
FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "common": ("roc_5", "roc_20", "trend_score_24", "ma_energy_41", "volume_ratio_5_20"),
    "trend": ("trend_score_24", "trend_score_25", "ma_gap_20", "legacy_trend_score"),
    "momentum": ("ret_5", "ret_10", "ret_20", "ret_60", "legacy_roc_10", "legacy_roc_18"),
    "relative": (
        "relative_return_to_benchmark_5", "relative_return_to_benchmark_10",
        "relative_return_to_benchmark_20", "relative_return_to_benchmark_60",
    ),
    "volume_risk": (
        "volume_ratio", "volume_ratio_5_20", "volatility_5", "volatility_20",
        "drawdown_20", "ma_energy_38", "ma_energy_39", "ma_energy_41",
    ),
}
FACTOR_GROUPS["all_selected"] = tuple(dict.fromkeys(sum(FACTOR_GROUPS.values(), ())))


def _rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def build_group_signals(panel: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    if mode not in FACTOR_GROUPS:
        raise ValueError(f"未知因子组合: {mode}")
    missing = [column for column in FACTOR_GROUPS[mode] if column not in panel.columns]
    if missing:
        raise ValueError(f"因子列缺失: {missing}")
    out = panel.copy()
    grouped = out.groupby("trade_date", sort=False)
    # Rank each factor within date before averaging, so numerical scales do not
    # determine the composite score.
    ranks = [grouped[column].transform(_rank) for column in FACTOR_GROUPS[mode]]
    out["score"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    out["factor_mode"] = mode
    return rank_and_select(out, top_k=top_k, score_column="score")


def _split_rows(window) -> dict[str, tuple[int, int]]:
    return {
        "train": (window.train_dates[0], window.train_dates[-1]),
        "val": (window.val_dates[0], window.val_dates[-1]),
        "test": (window.test_dates[0], window.test_dates[-1]),
    }


def _report_row(report, mode: str, split: str, start: int, end: int, top_k: int, hold_days: int, cost_rate: float) -> dict:
    return {
        "factor_mode": mode,
        "split": split,
        "start_date": start,
        "end_date": end,
        "top_k": top_k,
        "hold_days": hold_days,
        "cost_rate": cost_rate,
        "total_return_gross": report.total_return_gross,
        "total_return_net": report.total_return_net,
        "cagr": report.annualized_return,
        "max_drawdown": report.max_drawdown,
        "sharpe": report.sharpe,
        "calmar": report.calmar,
        "n_trades": report.n_trades,
        "hit_rate": report.hit_rate,
        "signal_count": report.signal_count,
    }


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(args.data_dir).glob("*.csv"))
    symbols = sorted(path.stem for path in files if path.stem != BENCHMARK)
    if len(symbols) != 50:
        raise RuntimeError(f"候选 ETF 数量应为 50，实际为 {len(symbols)}")
    frames = load_universe_frames(
        symbols, data_dir=args.data_dir, start=args.start, end=args.end,
        adjust="none", enforce_min_years=False,
    )
    benchmark = load_universe_frames(
        [BENCHMARK], data_dir=args.data_dir, start=args.start, end=args.end,
        adjust="none", enforce_min_years=False,
    )[BENCHMARK]
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0,
        hold_days=args.hold_days, date_mode="union", extended_features=True,
    )
    panel = add_common_factors(panel, benchmark_frame=benchmark)
    window = make_calendar_window(panel)
    split_ranges = {
        split: (int(start), int(end))
        for split, (start, end) in _split_rows(window).items()
    }
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    modes = args.modes or list(FACTOR_GROUPS)
    rows: list[dict] = []
    curve_rows: list[dict] = []
    for mode in modes:
        signals = build_group_signals(panel, mode, args.top_k)
        for split, (start, end) in split_ranges.items():
            scoped = signals[signals["trade_date"].between(start, end)]
            report = PortfolioBacktest(
                cost_rate=args.cost_rate, hold_days=args.hold_days, label_threshold=0.0,
            ).run(prices, scoped, end_date=end)
            rows.append(_report_row(report, mode, split, start, end, args.top_k, args.hold_days, args.cost_rate))
            if report.trades:
                trades = pd.DataFrame(report.trades)
                daily = trades.groupby("trade_date")["return_net"].mean()
                nav = (1.0 + daily).cumprod()
                for date, value in nav.items():
                    curve_rows.append({"factor_mode": mode, "split": split, "trade_date": int(date), "nav": float(value)})
    detail = pd.DataFrame(rows)
    indexed = detail.set_index(["factor_mode", "split"])
    summary = pd.DataFrame(index=sorted(detail["factor_mode"].unique()))
    for metric in ("total_return_net", "cagr"):
        for split in ("train", "val", "test"):
            name = "net_return" if metric == "total_return_net" else metric
            summary[f"{split}_{name}"] = indexed[metric].unstack("split")[split]
    test = indexed.xs("test", level="split")
    summary["test_max_drawdown"] = test["max_drawdown"]
    summary["test_sharpe"] = test["sharpe"]
    summary["test_trades"] = test["n_trades"]
    summary = summary.reset_index(names="factor_mode")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "factor_ablation_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "factor_ablation_summary.csv", index=False, encoding="utf-8-sig")
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(output_dir / "factor_ablation_nav.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_metadata.json").write_text(json.dumps({
        "data_dir": args.data_dir, "symbols": symbols, "benchmark": BENCHMARK,
        "hold_days": args.hold_days, "top_k": args.top_k, "cost_rate": args.cost_rate,
        "splits": split_ranges, "factor_groups": FACTOR_GROUPS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plot = summary.set_index("factor_mode")[["train_net_return", "val_net_return", "test_net_return"]]
    ax = plot.plot(kind="bar", figsize=(11, 6), color=["#f59e0b", "#16a34a", "#2563eb"])
    ax.axhline(0.0, color="#6b7280", lw=0.8)
    ax.set_title("quotes2 因子组合消融：分阶段净收益")
    ax.set_xlabel("因子组合")
    ax.set_ylabel("净收益")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "factor_ablation_returns.png", dpi=160, bbox_inches="tight")
    plt.close()

    if not curves.empty:
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
        for axis, split in zip(axes, ("train", "val", "test")):
            for mode, group in curves[curves["split"] == split].groupby("factor_mode"):
                axis.plot(group["trade_date"], group["nav"], label=mode, linewidth=1.2)
            axis.set_title(f"{split} 净值曲线")
            axis.set_ylabel("净值")
            axis.grid(alpha=0.25)
            axis.legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("交易日")
        plt.tight_layout()
        plt.savefig(output_dir / "factor_ablation_nav.png", dpi=160, bbox_inches="tight")
        plt.close()
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="quotes2")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2026-10-21")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rate", type=float, default=0.003)
    parser.add_argument("--modes", nargs="*", choices=sorted(FACTOR_GROUPS), default=None)
    parser.add_argument("--output-dir", default="output/research/reports/quotes2_factor_ablation")
    args = parser.parse_args()
    _, summary = run(args)
    print(summary.to_string(index=False))
    print(f"output_dir: {ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
