from research_platform.backend.app.research_planner import draft_research_plan
from research_platform.backend.app.schemas import ResearchGoal


def test_draft_plan_recognizes_baseline_and_splits_tasks() -> None:
    plan = draft_research_plan(ResearchGoal(
        objective="比较因子质量、Transformer训练和样本外回测",
        baseline_strategy_id="long_term_v1_quotes2",
        budget=10,
    ))
    assert plan.goal.baseline_strategy_id == "long_term_v1_quotes2"
    assert plan.tasks == ["factor_quality", "factor_correlation", "factor_common_train", "factor_common_oos"]
    assert plan.estimated_runs == 4


def test_draft_plan_has_safe_default_tasks() -> None:
    plan = draft_research_plan(ResearchGoal(objective="研究策略", budget=3))
    assert plan.tasks == ["factor_quality", "factor_common_train", "factor_common_oos"]
