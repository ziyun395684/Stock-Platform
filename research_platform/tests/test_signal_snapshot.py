import pytest

from research_platform.backend.app.signal_snapshot import build_signal_snapshot


def test_signal_snapshot_ranks_and_creates_manual_review_checklist() -> None:
    signals, checklist = build_signal_snapshot([{"symbol": "B", "score": 0.2}, {"symbol": "A", "score": 0.9}], "2026-11-18", 1)
    assert signals[0]["symbol"] == "A"
    assert signals[0]["action"] == "REVIEW_BUY"
    assert signals[1]["action"] == "WATCH"
    assert checklist[0]["checks"]["human_approval"] == "pending"


def test_signal_snapshot_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="snapshot_date"):
        build_signal_snapshot([{"symbol": "A", "score": 1}], "20261118")
    with pytest.raises(ValueError, match="symbol"):
        build_signal_snapshot([{"score": 1}], "2026-11-18")
