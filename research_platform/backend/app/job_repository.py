"""Read-only task/job queries for the platform API."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from .models import Artifact, ExperimentRun, TrainingJob
from .repositories import _session_factory


def _job_dict(job: TrainingJob, run: ExperimentRun, artifacts: list[Artifact]) -> dict[str, Any]:
    return {
        "id": job.id,
        "experiment_run_id": job.experiment_run_id,
        "task_id": job.task_id,
        "status": job.status,
        "progress": float(job.progress or 0),
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "experiment_id": run.experiment_id,
        "run_key": run.run_key,
        "run_status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error_type": run.error_type,
        "error_message": run.error_message,
        "artifacts": [{"id": item.id, "artifact_type": item.artifact_type, "relative_path": item.relative_path, "sha256": item.sha256, "size_bytes": item.size_bytes, "created_at": item.created_at.isoformat() if item.created_at else None} for item in artifacts],
    }


def list_jobs(limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    with _session_factory()() as session:
        query = select(TrainingJob, ExperimentRun).join(
            ExperimentRun, TrainingJob.experiment_run_id == ExperimentRun.id
        ).order_by(TrainingJob.created_at.desc()).limit(limit)
        if status:
            query = query.where(TrainingJob.status == status)
        rows = session.execute(query).all()
        return [_job_dict(job, run, list(session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all())) for job, run in rows]


def get_job(job_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.execute(
            select(TrainingJob, ExperimentRun)
            .join(ExperimentRun, TrainingJob.experiment_run_id == ExperimentRun.id)
            .where(TrainingJob.id == job_id)
        ).first()
        return _job_dict(row[0], row[1], list(session.scalars(select(Artifact).where(Artifact.experiment_run_id == row[1].id)).all())) if row else None
