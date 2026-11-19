"""Configuration helpers for comparable indicator-learning experiments.

The module deliberately keeps experiment definitions small and explicit.  The
rolling runner remains the single execution path so every mode shares the same
windows, costs, Top-K rule and evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from model_and_train.config import ModelConfig, merge_config
from model_and_train.feature_ablation import get_feature_group
from data.panel_sequences import AUXILIARY_INDICATOR_COLUMNS


EXPERIMENT_MODES = (
    "raw_only",
    "explicit_indicators",
    "indicator_auxiliary",
    "factor_only",
)


@dataclass(frozen=True)
class IndicatorExperimentSpec:
    name: str
    feature_columns: Optional[tuple[str, ...]]
    indicator_auxiliary_weight: float = 0.0
    indicator_target_columns: tuple[str, ...] = ()
    factor_only: bool = False


def get_experiment_spec(mode: str, auxiliary_weight: float = 0.2) -> IndicatorExperimentSpec:
    """Return a deterministic definition for one comparison arm."""
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"未知指标实验模式: {mode!r}")
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary_weight 不能为负")
    raw = tuple(get_feature_group("default"))
    if mode == "raw_only":
        return IndicatorExperimentSpec(mode, tuple(get_feature_group("raw")) if "raw" else raw)
    if mode == "explicit_indicators":
        columns = tuple(dict.fromkeys(raw + AUXILIARY_INDICATOR_COLUMNS))
        return IndicatorExperimentSpec(mode, columns)
    if mode == "indicator_auxiliary":
        return IndicatorExperimentSpec(
            mode,
            raw,
            indicator_auxiliary_weight=float(auxiliary_weight),
            indicator_target_columns=tuple(AUXILIARY_INDICATOR_COLUMNS),
        )
    return IndicatorExperimentSpec(mode, None, factor_only=True)


def build_model_config(
    base: Optional[ModelConfig | dict] = None,
    mode: str = "raw_only",
    auxiliary_weight: float = 0.2,
    return_target_scaling: bool = False,
    ranking_loss_weight: float = 0.0,
    dispersion_loss_weight: float = 0.0,
    dispersion_target_std: float = 0.0,
    batch_mode: str = "auto",
    ranking_target: str = "raw",
) -> ModelConfig:
    """Apply one experiment arm without changing the global defaults."""
    if ranking_loss_weight < 0.0:
        raise ValueError("ranking_loss_weight 不能为负")
    if dispersion_loss_weight < 0.0 or dispersion_target_std < 0.0:
        raise ValueError("离散度参数不能为负")
    if batch_mode not in {"auto", "random", "date_grouped"}:
        raise ValueError("batch_mode 必须是 auto、random 或 date_grouped")
    if ranking_target not in {"raw", "cross_sectional", "factor_residual"}:
        raise ValueError("ranking_target 必须是 raw、cross_sectional 或 factor_residual")
    spec = get_experiment_spec(mode, auxiliary_weight=auxiliary_weight)
    overrides = {
        "indicator_auxiliary_weight": spec.indicator_auxiliary_weight,
        "indicator_target_columns": list(spec.indicator_target_columns) or None,
        "return_target_scaling": bool(return_target_scaling),
        "ranking_loss_weight": float(ranking_loss_weight),
        "dispersion_loss_weight": float(dispersion_loss_weight),
        "dispersion_target_std": float(dispersion_target_std),
        "batch_mode": batch_mode,
        "ranking_target": ranking_target,
    }
    return merge_config(base, overrides)
