"""Validation rules for factor research metadata."""
from __future__ import annotations

from .schemas import FactorDefinition


FUTURE_DATA_MARKERS = ("future", "next_close", "next_open", "forward_return", "未来", "未来收益")


def validate_factor_definition(factor: FactorDefinition) -> list[str]:
    errors: list[str] = []
    searchable = " ".join([factor.version.formula, *factor.version.input_columns]).lower()
    for marker in FUTURE_DATA_MARKERS:
        if marker.lower() in searchable:
            errors.append(f"future-data marker is not allowed: {marker}")
    boundary = factor.version.time_boundary.lower()
    if "t+" in boundary or "未来" in boundary:
        errors.append("time_boundary must describe information available at or before decision time")
    return errors
