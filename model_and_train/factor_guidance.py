"""Shared utilities for factor-guided training and hybrid ranking."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def blend_factor_model_scores(
    predictions: pd.DataFrame,
    *,
    factor_weight: float = 0.5,
    model_weight: float = 0.5,
    factor_column: str = "legacy_factor_score",
    model_column: str = "pred_return",
) -> pd.DataFrame:
    """Blend causal factor and model scores using date-wise percentiles."""
    required = {"trade_date", "symbol", factor_column, model_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions 缺少字段: {sorted(missing)}")
    weights = (float(factor_weight), float(model_weight))
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("权重必须非负且总和大于 0")

    out = predictions.copy()
    factor = pd.to_numeric(out[factor_column], errors="coerce")
    model = pd.to_numeric(out[model_column], errors="coerce")
    if factor.isna().any() or model.isna().any():
        raise ValueError("因子分数和模型分数不能包含 NaN")
    out["factor_score_normalized"] = out.groupby("trade_date")[factor_column].transform(
        lambda series: pd.to_numeric(series, errors="coerce").rank(
            method="average", pct=True
        )
    )
    out["model_score_normalized"] = out.groupby("trade_date")[model_column].transform(
        lambda series: pd.to_numeric(series, errors="coerce").rank(
            method="average", pct=True
        )
    )
    total = sum(weights)
    out["hybrid_score"] = (
        out["factor_score_normalized"] * weights[0]
        + out["model_score_normalized"] * weights[1]
    ) / total
    return out


def build_factor_guidance_schedule(
    *,
    initial_weight: float = 0.5,
    final_weight: float = 0.0,
    stages: int = 4,
) -> list[float]:
    """Return a monotone linear factor-teacher weight schedule."""
    initial = float(initial_weight)
    final = float(final_weight)
    if not 0.0 <= final <= initial <= 1.0:
        raise ValueError("因子权重必须递减且位于 [0, 1]")
    if int(stages) < 2:
        raise ValueError("stages 至少为 2")
    return np.linspace(initial, final, int(stages)).round(12).tolist()


def validate_factor_guidance_schedule(schedule: Sequence[float]) -> list[float]:
    """Validate and normalize a user-provided factor guidance schedule."""
    values = [float(value) for value in schedule]
    if len(values) < 2:
        raise ValueError("因子权重 schedule 至少需要两个阶段")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("因子权重必须位于 [0, 1]")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError("因子权重 schedule 必须递减")
    return values
