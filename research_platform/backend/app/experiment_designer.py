"""Bounded experiment-matrix generation for research planning."""
from __future__ import annotations

from itertools import product

from .schemas import ExperimentDesignRequest, ExperimentDesignResponse


ALLOWED_WINDOWS = {500, 750, 1000}
ALLOWED_SEEDS = {7, 42, 123}
ALLOWED_COST_RATES = {0.003, 0.005}


def design_experiments(request: ExperimentDesignRequest) -> ExperimentDesignResponse:
    windows = [value for value in request.training_windows if value in ALLOWED_WINDOWS]
    seeds = [value for value in request.seeds if value in ALLOWED_SEEDS]
    costs = [value for value in request.cost_rates if value in ALLOWED_COST_RATES]
    if not windows or not seeds or not costs:
        return ExperimentDesignResponse(
            valid=False,
            experiments=[],
            estimated_runs=0,
            budget=request.budget,
            reason="windows, seeds, and cost_rates must use the supported research grid",
        )

    combinations = list(product(request.tasks, windows, seeds, costs))
    limited = combinations[: request.max_experiments]
    experiments = [
        {"task": task, "training_window": window, "seed": seed, "cost_rate": cost}
        for task, window, seed, cost in limited
    ]
    estimated = len(experiments)
    valid = estimated <= request.budget
    return ExperimentDesignResponse(
        valid=valid,
        experiments=experiments,
        estimated_runs=estimated,
        budget=request.budget,
        reason=None if valid else "experiment matrix exceeds research budget",
        rationale=[
            "Only supported windows, seeds, and cost rates are retained.",
            f"The matrix is capped at max_experiments={request.max_experiments}.",
            "The test set is not used by the design generator.",
        ],
    )
