"""Analyze causal factor effects for the quotes2 50-ETF universe."""
from __future__ import annotations

import argparse
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


BENCHMARK = "510300.SH"
FACTOR_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "volatility_5", "volatility_20", "volume_ratio", "ma_gap_20",
    "drawdown_20", "benchmark_return_1", "relative_benchmark_return",
    "benchmark_return_20", "relative_benchmark_return_20",
    "roc_5", "roc_10", "roc_20", "roc_60",
    "trend_score_24", "trend_score_25", "ma_energy_38", "ma_energy_39",
    "ma_energy_41", "volume_ratio_5_20",
    "relative_return_to_benchmark_5", "relative_return_to_benchmark_10",
    "relative_return_to_benchmark_20", "relative_return_to_benchmark_60",
    "legacy_trend_score", "legacy_roc_5", "legacy_roc_10",
    "legacy_roc_18", "legacy_volume_ratio", "legacy_factor_score",
]


def _daily_factor_rows(frame: pd.DataFrame, factor: str, quantiles: int) -> list[dict]:
    rows: list[dict] = []
    for date, group in frame.groupby("trade_date", sort=True):
        clean = group[[factor, "future_return", "symbol"]].copy()
        clean[factor] = pd.to_numeric(clean[factor], errors="coerce")
        clean["future_return"] = pd.to_numeric(clean["future_return"], errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < quantiles or clean[factor].nunique() < 2:
            continue
        ic = clean[factor].corr(clean["future_return"], method="spearman")
        if pd.isna(ic):
            continue
        pct = clean[factor].rank(method="first", pct=True)
        low = clean.loc[pct <= 1.0 / quantiles, "future_return"]
        high = clean.loc[pct > (quantiles - 1.0) / quantiles, "future_return"]
        rows.append({
            "trade_date": int(date),
            "year": int(str(int(date))[:4]),
            "n_symbols": int(len(clean)),
            "rank_ic": float(ic),
            "low_return": float(low.mean()),
            "high_return": float(high.mean()),
            "high_low_spread": float(high.mean() - low.mean()),
            "high_win_rate": float((high > 0).mean()),
        })
    return rows


def build_report(panel: pd.DataFrame, quantiles: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"trade_date", "future_return", "valid_sample", "symbol"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    usable = panel[panel["valid_sample"].astype(bool)].copy()
    usable["trade_date"] = pd.to_numeric(usable["trade_date"], errors="raise").astype(int)
    usable = usable.dropna(subset=["future_return"])
    available = [factor for factor in FACTOR_COLUMNS if factor in usable.columns]
    if not available:
        raise ValueError("panel 中没有可分析的因子列")

    daily_rows: list[dict] = []
    for split, split_frame in usable.groupby("split", sort=True):
        for factor in available:
            for row in _daily_factor_rows(split_frame, factor, quantiles):
                row.update({"split": str(split), "factor": factor})
                daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("没有生成任何单因子日度统计")
    summary = (
        daily.groupby(["split", "factor"], as_index=False)
        .agg(
            n_dates=("trade_date", "nunique"),
            mean_rank_ic=("rank_ic", "mean"),
            median_rank_ic=("rank_ic", "median"),
            positive_ic_ratio=("rank_ic", lambda values: float((values > 0).mean())),
            mean_low_return=("low_return", "mean"),
            mean_high_return=("high_return", "mean"),
            mean_high_low_spread=("high_low_spread", "mean"),
            high_win_rate=("high_win_rate", "mean"),
        )
    )
    yearly = (
        daily.groupby(["split", "year", "factor"], as_index=False)
        .agg(
            n_dates=("trade_date", "nunique"),
            mean_rank_ic=("rank_ic", "mean"),
            positive_ic_ratio=("rank_ic", lambda values: float((values > 0).mean())),
            mean_high_low_spread=("high_low_spread", "mean"),
            mean_high_return=("high_return", "mean"),
            mean_low_return=("low_return", "mean"),
        )
    )
    return summary.sort_values(["split", "mean_rank_ic"], ascending=[True, False]), yearly, daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="quotes2")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2026-10-21")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="output/research/reports/quotes2_factor_effect",
    )
    args = parser.parse_args()
    if args.quantiles < 2:
        raise ValueError("quantiles 必须至少为 2")

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
    from research.model_comparison.run_quotes2_50etf_time_split import make_calendar_window

    panel = add_common_factors(panel, benchmark_frame=benchmark)
    window = make_calendar_window(panel)
    split_by_date = {
        **{date: "train" for date in window.train_dates},
        **{date: "val" for date in window.val_dates},
        **{date: "test" for date in window.test_dates},
    }
    panel["split"] = panel["trade_date"].map(split_by_date)
    summary, yearly, daily = build_report(panel, quantiles=args.quantiles)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "factor_effect_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "factor_effect_by_year.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "factor_effect_daily.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pivot = summary.pivot(index="factor", columns="split", values="mean_rank_ic")
    ax = pivot.plot(kind="bar", figsize=(15, 7), color=["#2563eb", "#f59e0b", "#16a34a"])
    ax.axhline(0.0, color="#6b7280", lw=0.8)
    ax.set_title("quotes2 50 ETF 单因子横截面 Rank IC")
    ax.set_xlabel("因子")
    ax.set_ylabel("平均 Rank IC")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=65, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "factor_rank_ic.png", dpi=160, bbox_inches="tight")
    plt.close()

    top = summary[summary["split"] == "test"].sort_values("mean_high_low_spread", ascending=False).head(12)
    if not top.empty:
        ax = top.set_index("factor")["mean_high_low_spread"].sort_values().plot(
            kind="barh", figsize=(10, 6), color="#2563eb"
        )
        ax.axvline(0.0, color="#6b7280", lw=0.8)
        ax.set_title("quotes2 测试期因子高低分组收益差")
        ax.set_xlabel("最高分组收益 - 最低分组收益")
        ax.set_ylabel("因子")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "factor_test_spread.png", dpi=160, bbox_inches="tight")
        plt.close()

    print(summary.to_string(index=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
