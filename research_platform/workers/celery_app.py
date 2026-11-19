"""Celery application placeholder for platform jobs.

The worker keeps orchestration tasks in this package; quantitative work remains
implemented by the existing StockRNN engine modules and is called through
explicit adapters.
"""
from __future__ import annotations

import os

from celery import Celery


celery_app = Celery(
    "stockrnn_research_platform",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["research_platform.workers.tasks", "research_platform.workers.research_tasks", "research_platform.workers.factor_tasks", "research_platform.workers.factor_compute_tasks", "research_platform.workers.quality_tasks", "research_platform.workers.stratified_tasks", "research_platform.workers.transformer_tasks", "research_platform.workers.correlation_tasks", "research_platform.workers.decay_tasks", "research_platform.workers.regime_tasks", "research_platform.workers.training_tasks", "research_platform.workers.oos_tasks", "research_platform.workers.sensitivity_tasks", "research_platform.workers.gate_tasks", "research_platform.workers.signal_tasks", "research_platform.workers.paper_tasks"],
    task_time_limit=int(os.getenv("PLATFORM_TASK_TIME_LIMIT", "3600")),
    task_soft_time_limit=int(os.getenv("PLATFORM_TASK_SOFT_TIME_LIMIT", "3300")),
)
