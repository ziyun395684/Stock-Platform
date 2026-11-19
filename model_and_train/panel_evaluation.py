"""Evaluation reports for cross-ETF multitask predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


def _rank_ic(group: pd.DataFrame) -> float | None:
    if len(group) < 2 or group["future_return"].nunique() < 2 or group["pred_return"].nunique() < 2:
        return None
    return float(group["pred_return"].corr(group["future_return"], method="spearman"))


def mean_daily_rank_ic(frame: pd.DataFrame) -> float | None:
    """Calculate the mean cross-sectional Spearman IC by trade date."""
    required = {"trade_date", "future_return", "pred_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame 缺少字段: {sorted(missing)}")
    daily_ic = frame.groupby("trade_date", sort=True).apply(_rank_ic)
    valid_ic = daily_ic.dropna().to_numpy(dtype="float64")
    return float(np.mean(valid_ic)) if len(valid_ic) else None


def _metrics(frame: pd.DataFrame) -> dict:
    y_return = frame["future_return"].to_numpy(dtype="float64")
    pred_return = frame["pred_return"].to_numpy(dtype="float64")
    y_event = frame["event_label"].to_numpy(dtype="int64")
    event_probability = frame["event_probability"].to_numpy(dtype="float64")
    predicted_event = (event_probability >= 0.5).astype(int)
    auc = None
    if len(np.unique(y_event)) == 2:
        auc = float(roc_auc_score(y_event, event_probability))
    rank_ic = mean_daily_rank_ic(frame)
    return {
        "n_samples": int(len(frame)),
        "mae": float(mean_absolute_error(y_return, pred_return)),
        "rmse": float(np.sqrt(mean_squared_error(y_return, pred_return))),
        "f1": float(f1_score(y_event, predicted_event, zero_division=0)),
        "auc": auc,
        "rank_ic": rank_ic,
    }


def evaluate_panel_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return metrics by split, ETF symbol, and pooled ALL group."""
    required = {
        "trade_date", "symbol", "split", "future_return", "event_label",
        "pred_return", "event_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions 缺少字段: {sorted(missing)}")
    rows = []
    for split, split_frame in predictions.groupby("split", sort=True):
        groups = [(symbol, group) for symbol, group in split_frame.groupby("symbol", sort=True)]
        groups.append(("ALL", split_frame))
        for symbol, group in groups:
            rows.append({"split": split, "symbol": symbol, **_metrics(group)})
    return pd.DataFrame(rows)
