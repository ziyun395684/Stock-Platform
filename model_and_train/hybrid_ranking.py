"""Causal factor and model score blending for ranking experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def blend_model_and_factor_scores(
    predictions: pd.DataFrame,
    factor_scores: pd.DataFrame,
    model_weight: float,
) -> pd.DataFrame:
    """Blend cross-sectional percentile ranks of model and factor scores."""
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight 必须在 [0, 1] 范围内")
    prediction_required = {"trade_date", "symbol", "pred_return"}
    factor_required = {"trade_date", "symbol", "factor_score"}
    missing_prediction = prediction_required - set(predictions.columns)
    missing_factor = factor_required - set(factor_scores.columns)
    if missing_prediction:
        raise ValueError(f"predictions 缺少字段: {sorted(missing_prediction)}")
    if missing_factor:
        raise ValueError(f"factor_scores 缺少字段: {sorted(missing_factor)}")
    out = predictions[list(prediction_required)].merge(
        factor_scores[list(factor_required)],
        on=["trade_date", "symbol"],
        how="inner",
    )
    if out.empty:
        raise ValueError("模型预测与因子分数没有共同样本")
    grouped = out.groupby("trade_date", sort=False)
    out["model_score_rank"] = grouped["pred_return"].rank(method="average", pct=True)
    out["factor_score_rank"] = grouped["factor_score"].rank(method="average", pct=True)
    out["model_score_rank"] = out["model_score_rank"].fillna(0.0)
    out["factor_score_rank"] = out["factor_score_rank"].fillna(0.0)
    out["score"] = (
        model_weight * out["model_score_rank"]
        + (1.0 - model_weight) * out["factor_score_rank"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
