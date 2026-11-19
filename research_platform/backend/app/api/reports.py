"""Read-only report result API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..report_repository import list_result_artifacts
from ..candidate_report_repository import build_candidate_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports(experiment_id: str | None = None, artifact_type: str | None = None, limit: int = 100) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_result_artifacts(experiment_id, artifact_type, min(max(limit, 1), 500))


@router.get("/candidate/{experiment_id}")
def candidate_report(experiment_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="candidate report not found")
    result = build_candidate_report(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return result


@router.get("/{artifact_id}")
def get_report(artifact_id: str) -> dict:
    rows = list_result_artifacts(limit=500)
    result = next((row for row in rows if row["id"] == artifact_id), None)
    if result is None:
        raise HTTPException(status_code=404, detail="report not found")
    return result
