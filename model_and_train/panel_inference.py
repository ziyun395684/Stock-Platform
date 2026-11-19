"""Inference aggregation for shared multi-ETF models."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .ranking import rank_and_select


def aggregate_panel_predictions(
    predictions_by_window: Mapping[int, tuple[pd.DataFrame, Mapping[str, np.ndarray]]],
    top_k: int = 2,
    min_pred_return: float | None = None,
    min_event_probability: float | None = None,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Average available window predictions, then rank each trade date."""
    frames = []
    for look_back, (meta, prediction) in predictions_by_window.items():
        required = {"trade_date", "symbol"}
        if not required.issubset(meta.columns):
            raise ValueError(f"窗口 {look_back}: meta 缺少字段 {sorted(required - set(meta.columns))}")
        n = len(meta)
        pred_return = np.asarray(prediction["pred_return"]).reshape(-1)
        event_probability = np.asarray(prediction["event_probability"]).reshape(-1)
        if len(pred_return) != n or len(event_probability) != n:
            raise ValueError(f"窗口 {look_back}: 预测长度与 meta 不一致")
        item = meta[["trade_date", "symbol"]].copy()
        item["pred_return"] = pred_return
        item["event_probability"] = event_probability
        item["look_back"] = int(look_back)
        extra_columns = [
            column for column in meta.columns
            if column not in {"trade_date", "symbol"}
            and pd.api.types.is_numeric_dtype(meta[column])
        ]
        for column in extra_columns:
            item[column] = pd.to_numeric(meta[column], errors="coerce").to_numpy()
        frames.append(item)
    if not frames:
        raise ValueError("没有可用的面板预测")
    merged = pd.concat(frames, ignore_index=True)
    aggregations = {
        "pred_return": ("pred_return", "mean"),
        "event_probability": ("event_probability", "mean"),
    }
    for column in merged.columns:
        if column in {"trade_date", "symbol", "pred_return", "event_probability", "look_back"}:
            continue
        if pd.api.types.is_numeric_dtype(merged[column]):
            aggregations[column] = (column, "mean")
    averaged = merged.groupby(["trade_date", "symbol"], as_index=False).agg(**aggregations)
    return rank_and_select(
        averaged,
        top_k=top_k,
        min_pred_return=min_pred_return,
        min_event_probability=min_event_probability,
        score_weights=score_weights,
    )
