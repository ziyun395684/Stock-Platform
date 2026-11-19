"""Read-only report and factor result summaries."""
from __future__ import annotations

from sqlalchemy import select

from .artifact_repository import _session_factory
from .models import Artifact, ExperimentRun


def list_result_artifacts(experiment_id: str | None = None, artifact_type: str | None = None, limit: int = 100) -> list[dict]:
    with _session_factory()() as session:
        query = select(Artifact, ExperimentRun).join(ExperimentRun, Artifact.experiment_run_id == ExperimentRun.id)
        if experiment_id:
            query = query.where(ExperimentRun.experiment_id == experiment_id)
        if artifact_type:
            query = query.where(Artifact.artifact_type == artifact_type)
        rows = session.execute(query.order_by(Artifact.created_at.desc()).limit(limit)).all()
        return [
            {
                "id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "relative_path": artifact.relative_path,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "experiment_run_id": run.id,
                "experiment_id": run.experiment_id,
                "run_key": run.run_key,
                "run_status": run.status,
                "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            }
            for artifact, run in rows
        ]
