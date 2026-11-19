"""Operational notifications and data freshness checks."""
from __future__ import annotations

from datetime import date
from typing import Any


def build_run_notification(run_id: str, status: str, error: Exception | None = None) -> dict[str, Any] | None:
    if status == "failed":
        return {"event_type": "task_failed_alert", "severity": "error", "run_id": run_id, "message": str(error)[:4000] if error else "task failed"}
    if status == "succeeded":
        return {"event_type": "training_completed_notification", "severity": "info", "run_id": run_id, "message": "research task completed"}
    return None


def check_data_freshness(latest_date: str | None, expected_date: str, *, max_lag_days: int = 1) -> dict[str, Any]:
    try:
        expected = date.fromisoformat(expected_date)
        latest = date.fromisoformat(latest_date) if latest_date else None
    except ValueError as exc:
        raise ValueError("数据日期必须是 YYYY-MM-DD") from exc
    lag = None if latest is None else (expected - latest).days
    healthy = latest is not None and lag is not None and lag <= max_lag_days and lag >= 0
    return {"healthy": healthy, "latest_date": latest_date, "expected_date": expected_date, "lag_days": lag, "max_lag_days": max_lag_days, "alert": not healthy}
