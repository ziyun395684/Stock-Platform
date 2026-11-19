"""Build a comparable matrix from legacy strategy replay reports."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = [
    ROOT / "output/legacy_replay/ma_energy86_13_summary.csv",
    ROOT / "output/legacy_replay/long_term_summary.csv",
    ROOT / "output/research/legacy_replay/global_asset77_13_summary.csv",
    ROOT / "output/research/legacy_replay/composite_strategies_summary.csv",
]


def build_strategy_matrix(report_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in report_paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_report"] = str(path)
        frame["cagr"] = pd.to_numeric(frame.get("annualized_return"), errors="coerce")
        frame["strategy_family"] = frame["strategy"].astype(str).map(_strategy_family)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("没有找到可用的旧策略复测报告")

    detail = pd.concat(frames, ignore_index=True)
    detail["start"] = detail["start"].astype(str)
    detail["end"] = detail["end"].astype(str)
    detail["cost_rate"] = pd.to_numeric(detail["cost_rate"], errors="coerce")
    detail["cagr"] = pd.to_numeric(detail["cagr"], errors="coerce")
    detail["max_drawdown"] = pd.to_numeric(detail["max_drawdown"], errors="coerce")
    detail["cost_drag"] = pd.to_numeric(
        detail.get("total_return_gross", detail["total_return_net"]), errors="coerce"
    ) - pd.to_numeric(detail["total_return_net"], errors="coerce")
    detail = detail.sort_values(
        ["execution_rule", "cost_rate", "cagr"], ascending=[True, True, False]
    ).reset_index(drop=True)

    summary = detail.groupby(["execution_rule", "cost_rate"], as_index=False).agg(
        n_strategies=("strategy", "nunique"),
        mean_cagr=("cagr", "mean"),
        median_cagr=("cagr", "median"),
        positive_cagr_count=("cagr", lambda values: int((values > 0).sum())),
        mean_max_drawdown=("max_drawdown", "mean"),
        best_cagr=("cagr", "max"),
        worst_cagr=("cagr", "min"),
        mean_sharpe=("sharpe", "mean"),
        mean_calmar=("calmar", "mean"),
        mean_turnover=("turnover", "mean"),
    )
    return detail, summary


def _strategy_family(strategy: str) -> str:
    if "复合排序" in strategy:
        return "composite"
    if "均线能量" in strategy:
        return "ma_energy"
    if "全球资产" in strategy:
        return "global_asset"
    if "长期" in strategy or "十年" in strategy:
        return "long_term"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/research/reports")
    parser.add_argument("--reports", nargs="*", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    detail, summary = build_strategy_matrix(args.reports)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "legacy_strategy_matrix_detail.csv"
    summary_path = output_dir / "legacy_strategy_matrix_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"saved: {detail_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
