"""Celery adapters for deterministic StockRNN research-engine tasks."""
from __future__ import annotations

from typing import Any

from .celery_app import celery_app
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.engine_probe")
def engine_probe(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify the existing factor/backtest engine contract without changing it.

    This intentionally does not mutate strategies, train a model, or write an
    arbitrary artifact. It imports the source-of-truth engine entry points and
    records their supported factor modes as the first auditable async run.
    """
    _set_run_status(run_id, "running")
    try:
        from model_and_train.factor_baseline import FACTOR_MODES, build_factor_only_signals
        from model_and_train.portfolio_backtest import PortfolioBacktest
        from research.factor_baseline.run_factor_baselines import run

        result = {
            "status": "succeeded",
            "task": "engine_probe",
            "factor_modes": sorted(FACTOR_MODES),
            "entry_points": {
                "factor_signal_builder": f"{build_factor_only_signals.__module__}.{build_factor_only_signals.__name__}",
                "portfolio_backtest": f"{PortfolioBacktest.__module__}.{PortfolioBacktest.__name__}",
                "factor_baseline_runner": f"{run.__module__}.{run.__name__}",
            },
            "config": config or {},
        }
        _set_run_status(run_id, "succeeded")
        return result
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
