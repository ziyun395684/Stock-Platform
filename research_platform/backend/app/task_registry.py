"""Single registry for API-to-Celery research task dispatch."""
from __future__ import annotations

from typing import Any

from research_platform.workers.celery_app import celery_app
from research_platform.workers.correlation_tasks import factor_correlation
from research_platform.workers.decay_tasks import factor_decay
from research_platform.workers.factor_tasks import factor_baseline
from research_platform.workers.factor_compute_tasks import factor_compute
from research_platform.workers.gate_tasks import strategy_gate
from research_platform.workers.oos_tasks import factor_common_oos
from research_platform.workers.quality_tasks import factor_quality
from research_platform.workers.regime_tasks import factor_regime
from research_platform.workers.research_tasks import engine_probe
from research_platform.workers.sensitivity_tasks import factor_sensitivity
from research_platform.workers.signal_tasks import signal_snapshot
from research_platform.workers.paper_tasks import paper_portfolio
from research_platform.workers.stratified_tasks import factor_stratified_returns
from research_platform.workers.training_tasks import factor_common_train
from research_platform.workers.transformer_tasks import transformer_train


TASKS: dict[str, Any] = {
    "engine_probe": engine_probe,
    "factor_baseline": factor_baseline,
    "factor_compute": factor_compute,
    "factor_quality": factor_quality,
    "factor_stratified_returns": factor_stratified_returns,
    "factor_correlation": factor_correlation,
    "factor_decay": factor_decay,
    "factor_regime": factor_regime,
    "factor_common_train": factor_common_train,
    "transformer_train": transformer_train,
    "factor_common_oos": factor_common_oos,
    "factor_sensitivity": factor_sensitivity,
    "strategy_gate": strategy_gate,
    "signal_snapshot": signal_snapshot,
    "paper_portfolio": paper_portfolio,
}


def task_for(task_type: str) -> Any:
    try:
        return TASKS[task_type]
    except KeyError as exc:
        raise ValueError(f"unsupported research task type: {task_type}") from exc


def registered_task_names() -> set[str]:
    return set(TASKS)


def celery_task_names() -> set[str]:
    return {str(name).removeprefix("research_platform.") for name in celery_app.tasks}
