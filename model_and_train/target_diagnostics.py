"""Diagnostics for regression targets used by the shared panel model."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .panel_training import build_return_targets


def summarize_return_targets(
    sample_meta,
    indices: Optional[np.ndarray] = None,
    target_mode: str = "raw",
) -> dict[str, float | int | str]:
    """Summarize the effective regression-target distribution.

    The summary is calculated after applying the same target-mode logic used
    during training, so raw and cross-sectional experiments are comparable.
    """
    if "future_return" not in sample_meta.columns:
        raise ValueError("sample_meta 缺少字段: future_return")
    if indices is None:
        indices = np.arange(len(sample_meta), dtype=int)
    values = np.asarray(
        build_return_targets(sample_meta, np.asarray(indices, dtype=int), target_mode),
        dtype="float64",
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("没有可用于目标分布诊断的有限收益标签")
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "target_mode": target_mode,
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_targets_by_split(sample_meta, target_mode: str = "raw") -> list[dict]:
    """Return target summaries for each date-isolated dataset split."""
    if "split" not in sample_meta.columns:
        raise ValueError("sample_meta 缺少字段: split")
    rows = []
    for split, group in sample_meta.groupby("split", sort=True):
        summary = summarize_return_targets(
            group.reset_index(drop=True), target_mode=target_mode
        )
        rows.append({"split": str(split), **summary})
    return rows
