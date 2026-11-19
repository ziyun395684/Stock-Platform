"""Minimal Celery tasks used to verify the platform worker path."""
from __future__ import annotations

from datetime import datetime, timezone

from .celery_app import celery_app


def _set_run_status(run_id: str, status: str, *, error: Exception | None = None) -> None:
    from research_platform.backend.app.run_repository import update_run

    update_run(run_id, status, error=error)


@celery_app.task(name="research_platform.smoke_task")
def smoke_task(message: str = "ok") -> dict[str, str]:
    """Return a deterministic result without touching the quant engine."""
    return {
        "status": "ok",
        "message": message,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
