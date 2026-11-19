"""Normalize and aggregate rolling reports across training-window lengths."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


_METRIC_COLUMNS = (
    "total_return_net",
    "max_drawdown",
    "auc",
    "rank_ic",
    "n_trades",
)


def build_window_comparison(
    reports: Mapping[int | str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return normalized per-window rows and aggregate metrics.

    Missing optional columns are represented as ``unknown`` for regime and
    ``NaN`` for numeric metrics. The function does not alter source frames.
    """
    frames: list[pd.DataFrame] = []
    for window_days, report in reports.items():
        if not isinstance(report, pd.DataFrame):
            raise TypeError("每个窗口报告必须是 pandas.DataFrame")
        required = {"window", "test_start", "test_end", "total_return_net", "max_drawdown"}
        missing = required - set(report.columns)
        if missing:
            raise ValueError(f"窗口 {window_days} 报告缺少字段: {sorted(missing)}")
        frame = report.copy()
        frame.insert(0, "train_window_days", int(window_days))
        if "market_regime" not in frame.columns:
            frame["market_regime"] = "unknown"
        for column in _METRIC_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frames.append(frame)
    if not frames:
        raise ValueError("至少需要一个窗口报告")

    per_window = pd.concat(frames, ignore_index=True, sort=False)
    per_window = per_window.sort_values(["train_window_days", "window"]).reset_index(drop=True)
    summary = (
        per_window.groupby("train_window_days", as_index=False)
        .agg(
            n_windows=("window", "size"),
            positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
            mean_net_return=("total_return_net", "mean"),
            compound_net_return=("total_return_net", lambda values: float((1.0 + values).prod() - 1.0)),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_auc=("auc", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            total_trades=("n_trades", "sum"),
        )
    )
    regime = (
        per_window.groupby(["train_window_days", "market_regime"], as_index=False)
        .agg(
            n_windows=("window", "size"),
            positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
            mean_net_return=("total_return_net", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
        )
    )
    return {"per_window": per_window, "summary": summary, "by_regime": regime}

