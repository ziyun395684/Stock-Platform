from research_platform.backend.app.experiment_designer import design_experiments
from research_platform.backend.app.schemas import ExperimentDesignRequest


def test_experiment_design_is_bounded_and_records_rationale() -> None:
    result = design_experiments(ExperimentDesignRequest(
        tasks=["factor_quality"],
        training_windows=[500, 750],
        seeds=[7, 42],
        cost_rates=[0.003],
        max_experiments=3,
        budget=3,
    ))
    assert result.valid is True
    assert result.estimated_runs == 3
    assert len(result.experiments) == 3
    assert result.rationale


def test_experiment_design_rejects_unsupported_grid() -> None:
    result = design_experiments(ExperimentDesignRequest(tasks=["factor_quality"], training_windows=[365]))
    assert result.valid is False
    assert "supported research grid" in (result.reason or "")


def test_experiment_design_requires_budget() -> None:
    result = design_experiments(ExperimentDesignRequest(tasks=["factor_quality"], max_experiments=2, budget=1))
    assert result.valid is False
    assert result.estimated_runs == 2
