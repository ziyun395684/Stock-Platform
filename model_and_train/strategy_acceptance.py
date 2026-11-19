"""Transparent strategy-level acceptance gates.

The functions in this module evaluate evidence produced on rolling
out-of-sample windows.  They do not tune a strategy and never inspect a
frozen final-test result when deciding rolling gates.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少字段: {sorted(missing)}")


def _required_positive_windows(n_windows: int, fraction: float) -> int:
    if n_windows <= 0:
        return 1
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction 必须位于 (0, 1]")
    return int(math.ceil(n_windows * fraction))


def assess_rolling_generalization(
    report: pd.DataFrame,
    *,
    min_positive_fraction: float = 0.5,
    min_positive_regimes: int = 2,
) -> dict[str, Any]:
    """Check whether positive net returns are spread across regimes.

    A strategy passes only when at least half of the windows are positive and
    at least two distinct market regimes have a positive mean return.  Both
    thresholds are configurable for research experiments.
    """
    _require_columns(report, {"market_regime", "total_return_net"}, "rolling report")
    frame = report[["market_regime", "total_return_net"]].copy()
    frame["total_return_net"] = pd.to_numeric(frame["total_return_net"], errors="coerce")
    frame = frame.dropna(subset=["total_return_net"])
    n_windows = int(len(frame))
    positive_windows = int((frame["total_return_net"] > 0).sum())
    required_windows = _required_positive_windows(n_windows, min_positive_fraction)
    regime_means = frame.groupby("market_regime")["total_return_net"].mean()
    positive_regimes = int((regime_means > 0).sum())
    passed = (
        n_windows > 0
        and positive_windows >= required_windows
        and positive_regimes >= int(min_positive_regimes)
    )
    if positive_windows < required_windows:
        reason = "positive window fraction below threshold"
    elif positive_regimes < int(min_positive_regimes):
        reason = "positive returns are concentrated in too few regimes"
    else:
        reason = "positive windows span multiple regimes"
    return {
        "passed": bool(passed),
        "reason": reason,
        "n_windows": n_windows,
        "positive_windows": positive_windows,
        "positive_window_fraction": positive_windows / n_windows if n_windows else 0.0,
        "required_positive_windows": required_windows,
        "positive_regimes": positive_regimes,
        "regime_mean_net_return": regime_means.to_dict(),
    }


def assess_top_k_stability(
    report: pd.DataFrame,
    *,
    selected_top_k: int = 2,
    min_best_windows: int = 2,
) -> dict[str, Any]:
    """Check that the selected Top-K is not supported by one window only."""
    _require_columns(report, {"window", "top_k", "total_return_net"}, "Top-K report")
    frame = report.copy()
    frame["total_return_net"] = pd.to_numeric(frame["total_return_net"], errors="coerce")
    frame = frame.dropna(subset=["total_return_net"])
    frame["top_k"] = pd.to_numeric(frame["top_k"], errors="coerce")
    frame = frame.dropna(subset=["top_k"])
    best_by_window = frame.loc[frame.groupby("window")["total_return_net"].idxmax()]
    selected = best_by_window[best_by_window["top_k"] == int(selected_top_k)]
    selected_rows = frame[frame["top_k"] == int(selected_top_k)]
    selected_mean = (
        float(selected_rows["total_return_net"].mean()) if len(selected_rows) else None
    )
    selected_best_windows = int(len(selected))
    n_windows = int(best_by_window["window"].nunique())
    passed = n_windows >= int(min_best_windows) and selected_best_windows >= int(min_best_windows)
    reason = (
        "selected Top-K is best across multiple windows"
        if passed
        else "selected Top-K is best in too few rolling windows"
    )
    return {
        "passed": bool(passed),
        "reason": reason,
        "selected_top_k": int(selected_top_k),
        "n_windows": n_windows,
        "selected_best_windows": selected_best_windows,
        "selected_best_fraction": selected_best_windows / n_windows if n_windows else 0.0,
        "selected_mean_net_return": selected_mean,
        "best_top_k_by_window": best_by_window[["window", "top_k", "total_return_net"]]
        .to_dict("records"),
    }


def assess_incremental_value(
    comparison: pd.DataFrame,
    *,
    min_positive_fraction: float = 0.5,
) -> dict[str, Any]:
    """Check whether Model-3 beats Baseline-1 repeatedly out of sample."""
    _require_columns(comparison, {"window", "baseline_net", "model3_net"}, "incremental report")
    frame = comparison[["window", "baseline_net", "model3_net"]].copy()
    for column in ("baseline_net", "model3_net"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["baseline_net", "model3_net"])
    frame["delta"] = frame["model3_net"] - frame["baseline_net"]
    n_windows = int(len(frame))
    positive_delta_windows = int((frame["delta"] > 0).sum())
    required_windows = _required_positive_windows(n_windows, min_positive_fraction)
    mean_delta = float(frame["delta"].mean()) if n_windows else 0.0
    passed = n_windows > 0 and positive_delta_windows >= required_windows and mean_delta > 0
    reason = (
        "Model-3 has positive mean incremental value"
        if passed
        else "Model-3 does not show repeatable incremental value"
    )
    return {
        "passed": bool(passed),
        "reason": reason,
        "n_windows": n_windows,
        "positive_delta_windows": positive_delta_windows,
        "positive_delta_fraction": positive_delta_windows / n_windows if n_windows else 0.0,
        "required_positive_windows": required_windows,
        "mean_delta": mean_delta,
        "delta_by_window": frame[["window", "delta"]].to_dict("records"),
    }
