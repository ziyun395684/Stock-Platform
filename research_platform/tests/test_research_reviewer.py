from research_platform.backend.app.research_reviewer import review_research
from research_platform.backend.app.schemas import ResearchReviewRequest


def test_review_research_passes_with_required_evidence() -> None:
    result = review_research(ResearchReviewRequest(
        config={"oos": True, "embargo_days": 5, "training_windows": [500, 1000], "seeds": [7, 42], "cost_rates": [0.003, 0.005]},
        metrics={"market_regimes": ["up", "down"], "prediction_std": 0.2, "max_drawdown": 0.1, "turnover": 2.0},
    ))
    assert result.passed is True
    assert all(item.status != "fail" for item in result.findings)


def test_review_research_blocks_leakage_and_test_selection() -> None:
    result = review_research(ResearchReviewRequest(
        config={"formula": "next_close / close", "use_test_for_config": True},
    ))
    assert result.passed is False
    assert {item.check for item in result.findings if item.status == "fail"} >= {"data_leakage", "test_set_usage"}
