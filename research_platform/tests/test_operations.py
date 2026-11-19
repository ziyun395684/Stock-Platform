import pytest

from research_platform.backend.app.operations import build_run_notification, check_data_freshness


def test_run_notifications_cover_failure_and_completion() -> None:
    assert build_run_notification("run-1", "failed", RuntimeError("boom"))["event_type"] == "task_failed_alert"
    assert build_run_notification("run-1", "succeeded")["event_type"] == "training_completed_notification"
    assert build_run_notification("run-1", "running") is None


def test_data_freshness_detects_missing_and_stale_data() -> None:
    assert check_data_freshness("2026-11-17", "2026-11-18")["healthy"] is True
    assert check_data_freshness("2026-11-15", "2026-11-18")["alert"] is True
    assert check_data_freshness(None, "2026-11-18")["alert"] is True
    with pytest.raises(ValueError):
        check_data_freshness("bad", "2026-11-18")
