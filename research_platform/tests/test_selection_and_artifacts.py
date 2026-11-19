import pytest

from research_platform.backend.app.artifact_repository import validate_artifact_record
from research_platform.backend.app.candidate_report import summarize_observations
from research_platform.backend.app.selection_policy import validate_selection_policy


def test_selection_policy_rejects_test_split_and_forbidden_flag() -> None:
    valid, violations = validate_selection_policy({"selection_split": "test", "use_test_for_config": True})
    assert valid is False
    assert "selection_split=test" in violations
    assert "use_test_for_config" in violations


def test_candidate_report_cannot_recommend_with_test_selection() -> None:
    observations = [
        {"window": 500, "seed": 7, "cost_rate": 0.003, "regime": "bull", "total_return_net": 0.1},
        {"window": 750, "seed": 42, "cost_rate": 0.005, "regime": "bear", "total_return_net": 0.2},
    ]
    report = summarize_observations(observations, total_runs=2, config={"selection_split": "test"})
    assert report["checks"]["test_set_not_used_for_selection"] is False
    assert report["recommendation"]["eligible"] is False


def test_artifact_index_validates_path_digest_and_size() -> None:
    validate_artifact_record("output/platform/report.json", "a" * 64, 0)
    with pytest.raises(ValueError):
        validate_artifact_record("../secret.txt", "a" * 64, 1)
    with pytest.raises(ValueError):
        validate_artifact_record("report.json", "bad", 1)
    with pytest.raises(ValueError):
        validate_artifact_record("report.json", "a" * 64, -1)
