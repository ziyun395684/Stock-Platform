"""Read-only trained model artifact API."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..report_repository import list_result_artifacts
from ..artifact_repository import get_artifact
from ..model_integrity import verify_artifact_file

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(experiment_id: str | None = None, limit: int = 100) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    rows = list_result_artifacts(experiment_id=experiment_id, limit=min(max(limit, 1), 500))
    return [row for row in rows if row["artifact_type"] in {"transformer_checkpoint", "transformer_training_metadata", "factor_common_model_metadata"}]


@router.get("/{artifact_id}/integrity")
def model_integrity(artifact_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="model artifact not found")
    artifact = get_artifact(artifact_id)
    if artifact is None or artifact.artifact_type not in {"transformer_checkpoint", "transformer_training_metadata", "factor_common_model_metadata"}:
        raise HTTPException(status_code=404, detail="model artifact not found")
    root = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[3]
    return {"id": artifact.id, "artifact_type": artifact.artifact_type, "relative_path": artifact.relative_path, **verify_artifact_file(root, artifact.relative_path, artifact.sha256, artifact.size_bytes)}
