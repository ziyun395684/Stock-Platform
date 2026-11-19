"""Diagnostics for cross-sectional ranking direction and spread."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame) -> None:
    required = {"trade_date", "symbol", "future_return", "pred_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame 缺少字段: {sorted(missing)}")


def summarize_symbol_direction(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-symbol sign agreement between prediction and outcome."""
    _require_columns(frame)
    rows = []
    for symbol, group in frame.groupby("symbol", sort=True):
        actual = pd.to_numeric(group["future_return"], errors="coerce")
        predicted = pd.to_numeric(group["pred_return"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid].to_numpy(dtype="float64")
        predicted = predicted[valid].to_numpy(dtype="float64")
        if len(actual) == 0:
            continue
        rows.append({
            "symbol": symbol,
            "n_samples": int(len(actual)),
            "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
            "mean_future_return": float(np.mean(actual)),
            "mean_pred_return": float(np.mean(predicted)),
            "pred_return_std": float(np.std(predicted)),
        })
    return pd.DataFrame(rows)


def summarize_cross_sectional_slices(
    frame: pd.DataFrame, quantiles: int = 5
) -> dict[str, float | int]:
    """Compare actual returns of daily prediction top and bottom slices."""
    _require_columns(frame)
    if quantiles < 2:
        raise ValueError("quantiles 必须至少为 2")
    top_returns = []
    bottom_returns = []
    for _, group in frame.groupby("trade_date", sort=True):
        group = group.dropna(subset=["future_return", "pred_return"])
        if len(group) < 2:
            continue
        count = max(1, int(np.ceil(len(group) / quantiles)))
        ordered = group.sort_values(["pred_return", "symbol"], ascending=[False, True])
        top_returns.extend(ordered.head(count)["future_return"].astype(float).tolist())
        bottom_returns.extend(ordered.tail(count)["future_return"].astype(float).tolist())
    if not top_returns or not bottom_returns:
        raise ValueError("没有足够的横截面样本")
    top_mean = float(np.mean(top_returns))
    bottom_mean = float(np.mean(bottom_returns))
    return {
        "n_dates": int(frame["trade_date"].nunique()),
        "top_mean_future_return": top_mean,
        "bottom_mean_future_return": bottom_mean,
        "top_bottom_spread": top_mean - bottom_mean,
    }
