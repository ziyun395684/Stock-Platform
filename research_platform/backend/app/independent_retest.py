"""Validation and configuration for frozen-strategy independent retests."""
from __future__ import annotations

from datetime import date
from typing import Any

from .training_metadata import config_digest


def _parse_day(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是 YYYY-MM-DD") from exc


def build_retest_config(
    strategy: dict[str, Any],
    version: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Create an immutable, explicitly independent OOS configuration."""
    if strategy.get("status") != "frozen":
        raise ValueError("only frozen strategies can be independently retested")
    config = dict(version.get("config") or {})
    reference_end = request.get("reference_end") or config.get("test_end") or config.get("end") or config.get("train_end")
    test_start = request.get("start")
    test_end = request.get("end")
    if not reference_end:
        raise ValueError("frozen strategy version must declare a reference end date")
    if not test_start or not test_end:
        raise ValueError("independent retest requires start and end dates")
    reference_day = _parse_day(reference_end, "reference_end")
    start_day = _parse_day(test_start, "start")
    end_day = _parse_day(test_end, "end")
    if start_day <= reference_day:
        raise ValueError("independent retest start must be after the frozen strategy reference end")
    if end_day <= start_day:
        raise ValueError("independent retest end must be after start")
    result = {
        **config,
        **request,
        "independent_oos": True,
        "strategy_id": strategy["id"],
        "strategy_key": strategy.get("strategy_id"),
        "strategy_version_id": version["id"],
        "reference_end": reference_day.isoformat(),
        "start": start_day.isoformat(),
        "end": end_day.isoformat(),
        "source": "frozen_strategy_independent_oos",
    }
    result.pop("reference_end", None)
    result["reference_end"] = reference_day.isoformat()
    result["retest_config_digest"] = config_digest(result)
    return result
