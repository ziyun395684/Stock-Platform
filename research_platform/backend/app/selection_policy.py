"""Deterministic guards preventing test-set driven configuration selection."""
from __future__ import annotations

from typing import Any


FORBIDDEN_FLAGS = ("use_test_for_config", "select_on_test", "test_tuning", "test_set_selection")


def validate_selection_policy(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether model/config selection is isolated from the test set."""
    violations = [name for name in FORBIDDEN_FLAGS if config.get(name) is True]
    selected_split = config.get("selection_split") or config.get("hyperparameter_selection_split")
    if selected_split is not None and str(selected_split).lower() in {"test", "holdout_test", "oos_test"}:
        violations.append(f"selection_split={selected_split}")
    selection_metric_split = config.get("selection_metric_split")
    if selection_metric_split is not None and str(selection_metric_split).lower() in {"test", "holdout_test", "oos_test"}:
        violations.append(f"selection_metric_split={selection_metric_split}")
    return not violations, violations
