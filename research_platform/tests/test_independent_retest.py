import pytest

from research_platform.backend.app.independent_retest import build_retest_config


def _strategy(status: str = "frozen") -> tuple[dict, dict]:
    return (
        {"id": "strategy-db-id", "strategy_id": "factor-v2", "status": status},
        {"id": "version-db-id", "config": {"end": "2025-12-31", "top_k": 2}},
    )


def test_independent_retest_requires_later_date_and_preserves_lineage() -> None:
    strategy, version = _strategy()
    config = build_retest_config(strategy, version, {"start": "2026-01-01", "end": "2026-03-31"})
    assert config["independent_oos"] is True
    assert config["strategy_id"] == "strategy-db-id"
    assert config["strategy_version_id"] == "version-db-id"
    assert config["reference_end"] == "2025-12-31"
    assert config["start"] == "2026-01-01"


def test_independent_retest_rejects_overlap_and_non_frozen_strategy() -> None:
    strategy, version = _strategy()
    with pytest.raises(ValueError, match="after"):
        build_retest_config(strategy, version, {"start": "2025-12-31", "end": "2026-03-31"})
    strategy["status"] = "candidate"
    with pytest.raises(ValueError, match="only frozen"):
        build_retest_config(strategy, version, {"start": "2026-01-01", "end": "2026-03-31"})
