"""Database operations for asynchronous experiment runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from .database import get_engine
from .models import Artifact, Experiment, ExperimentRun, TrainingJob
from .repositories import _session_factory


def _artifact_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "artifact_type": artifact.artifact_type,
        "relative_path": artifact.relative_path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


def _run_dict(run: ExperimentRun, job: TrainingJob | None = None, artifacts: list[Artifact] | None = None) -> dict[str, Any]:
    result = {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "run_key": run.run_key,
        "status": run.status,
        "seed": run.seed,
        "window_name": run.window_name,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error_type": run.error_type,
        "error_message": run.error_message,
    }
    if job:
        result["task_id"] = job.task_id
        result["progress"] = float(job.progress or 0)
    result["artifacts"] = [_artifact_dict(item) for item in (artifacts or [])]
    if "_retry_of_" in run.run_key:
        result["retry_of"] = run.run_key.split("_retry_of_", 1)[1].rsplit("_", 1)[0]
    else:
        result["retry_of"] = None
    return result


def create_run(experiment_id: str, run_key: str, config: dict[str, Any]) -> dict[str, Any]:
    with _session_factory()() as session:
        experiment = session.scalar(select(Experiment).where(Experiment.id == experiment_id))
        if experiment is None:
            raise LookupError("experiment not found")
        run = ExperimentRun(
            id=str(uuid4()),
            experiment_id=experiment_id,
            run_key=run_key,
            status="queued",
            seed=config.get("seed"),
            window_name=config.get("window_name"),
        )
        job = TrainingJob(id=str(uuid4()), experiment_run_id=run.id, status="queued", progress=0)
        session.add_all([run, job])
        session.commit()
        return _run_dict(run, job)


def get_run_by_key(experiment_id: str, run_key: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        run = session.scalar(select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id, ExperimentRun.run_key == run_key))
        if run is None:
            return None
        job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run.id))
        artifacts = session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all()
        return _run_dict(run, job, artifacts)


def get_run(run_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        run = session.scalar(select(ExperimentRun).where(ExperimentRun.id == run_id))
        if run is None:
            return None
        job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run.id))
        artifacts = session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all()
        return _run_dict(run, job, artifacts)


def list_runs(experiment_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with _session_factory()() as session:
        runs = session.scalars(select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id).order_by(ExperimentRun.created_at.desc()).limit(limit)).all()
        output = []
        for run in runs:
            job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run.id))
            artifacts = session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all()
            output.append(_run_dict(run, job, artifacts))
        return output


def attach_task(run_id: str, task_id: str) -> None:
    with _session_factory()() as session:
        job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run_id))
        if job:
            job.task_id = task_id
            session.commit()


def cancel_run(run_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        run = session.scalar(select(ExperimentRun).where(ExperimentRun.id == run_id))
        job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run_id))
        if run is None or job is None:
            return None
        if run.status in {"succeeded", "failed", "cancelled"}:
            return _run_dict(run, job, session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all())
        run.status = "cancelled"
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.status = "cancelled"
        session.commit()
        return _run_dict(run, job, session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all())


def update_run(run_id: str, status: str, *, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with _session_factory()() as session:
        run = session.scalar(select(ExperimentRun).where(ExperimentRun.id == run_id))
        job = session.scalar(select(TrainingJob).where(TrainingJob.experiment_run_id == run_id))
        if run is None:
            return
        run.status = status
        if status == "running":
            run.started_at = now
        if status in {"succeeded", "failed", "cancelled"}:
            run.finished_at = now
        if error:
            run.error_type = type(error).__name__
            run.error_message = str(error)[:4000]
        if job:
            job.status = status
            job.progress = 100 if status == "succeeded" else job.progress
        session.commit()
    if status in {"succeeded", "failed"}:
        try:
            from .operations import build_run_notification
            from .approval_repository import create_audit_event
            notification = build_run_notification(run_id, status, error)
            if notification:
                create_audit_event(notification["event_type"], "experiment_run", run_id, "platform", notification)
        except Exception:
            # Notifications must never hide the primary task status update.
            pass
