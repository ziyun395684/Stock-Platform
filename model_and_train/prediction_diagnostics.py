"""Diagnostics for detecting collapsed panel model predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_prediction_dispersion(
    prediction_frame: pd.DataFrame,
    *,
    collapse_std_threshold: float = 1e-8,
) -> dict[str, float | bool]:
    """Summarize prediction spread and flag effectively constant outputs."""
    required = {"pred_return", "event_probability"}
    missing = required - set(prediction_frame.columns)
    if missing:
        raise ValueError(f"prediction_frame 缺少字段: {sorted(missing)}")
    pred_return = pd.to_numeric(prediction_frame["pred_return"], errors="coerce").dropna()
    event_probability = pd.to_numeric(
        prediction_frame["event_probability"], errors="coerce"
    ).dropna()
    if pred_return.empty or event_probability.empty:
        raise ValueError("预测结果没有有效数值")
    return_std = float(np.std(pred_return.to_numpy(dtype=float)))
    event_std = float(np.std(event_probability.to_numpy(dtype=float)))
    if return_std <= collapse_std_threshold:
        return_std = 0.0
    if event_std <= collapse_std_threshold:
        event_std = 0.0
    unique_ratio = float(pred_return.nunique() / len(pred_return))
    return {
        "pred_return_std": return_std,
        "event_probability_std": event_std,
        "unique_pred_return_ratio": unique_ratio,
        "prediction_collapsed": bool(
            return_std <= collapse_std_threshold
            and event_std <= collapse_std_threshold
        ),
    }
