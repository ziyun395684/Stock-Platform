"""Named, reproducible feature groups for panel ablation experiments."""
from __future__ import annotations

from typing import Tuple


_FEATURE_GROUPS: dict[str, Tuple[str, ...]] = {
    "legacy": (
        "legacy_trend_score",
        "legacy_roc_5",
        "legacy_roc_10",
        "legacy_roc_18",
        "legacy_volume_ratio",
        "legacy_factor_score",
    ),
    "returns": (
        "ret_1",
        "ret_5",
        "ret_20",
        "ret_1_rank",
        "ret_5_rank",
        "ret_20_rank",
        "benchmark_return_1",
        "relative_benchmark_return",
    ),
    "risk": (
        "volatility_20",
        "volatility_20_rank",
        "volume_ratio",
        "volume_ratio_rank",
    ),
}

_FEATURE_GROUPS["engineered"] = tuple(
    dict.fromkeys(_FEATURE_GROUPS["legacy"] + _FEATURE_GROUPS["returns"] + _FEATURE_GROUPS["risk"])
)

_RAW_FEATURES = (
    "open", "high", "low", "close", "volume", "amount", "is_suspend",
)
_FEATURE_GROUPS["default"] = tuple(_RAW_FEATURES) + _FEATURE_GROUPS["engineered"]
_FEATURE_GROUPS["raw"] = _RAW_FEATURES
_FEATURE_GROUPS["no_legacy"] = tuple(
    column for column in _FEATURE_GROUPS["default"] if not column.startswith("legacy_")
)
_FEATURE_GROUPS["no_rank"] = tuple(
    column for column in _FEATURE_GROUPS["default"] if not column.endswith("_rank")
)
_FEATURE_GROUPS["no_benchmark"] = tuple(
    column for column in _FEATURE_GROUPS["default"]
    if column not in {"benchmark_return_1", "relative_benchmark_return"}
)
_FEATURE_GROUPS["no_raw"] = _FEATURE_GROUPS["engineered"]


def get_feature_group(name: str) -> list[str]:
    """Return a copy of a named feature group for configuration use."""
    try:
        return list(_FEATURE_GROUPS[name])
    except KeyError as exc:
        raise ValueError(f"未知特征组: {name!r}") from exc


def list_feature_groups() -> list[str]:
    """Return available ablation group names in stable order."""
    return list(_FEATURE_GROUPS)
