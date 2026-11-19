from research_platform.backend.app.paper_trading import apply_paper_snapshot


def test_paper_snapshot_updates_holdings_nav_and_costs() -> None:
    state = apply_paper_snapshot({"capital": 1.0, "holdings": ["OLD"]}, [{"symbol": "A", "score": 2, "approved": True, "return_1d": 0.1}, {"symbol": "B", "score": 1, "approved": False, "return_1d": 0.2}], "2026-11-18", 1, 0.01)
    assert state["holdings"] == ["A"]
    assert state["automatic_ordering"] is False
    assert state["capital"] == 1.09


def test_paper_snapshot_requires_manual_approval() -> None:
    state = apply_paper_snapshot(None, [{"symbol": "A", "score": 2, "approved": False, "return_1d": 0.5}], "2026-11-18", 1)
    assert state["holdings"] == []
    assert state["capital"] == 1.0
