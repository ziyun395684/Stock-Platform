"""Read-only factor result API."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..factor_repository import list_factor_metadata
from ..factor_validation import validate_factor_definition
from ..report_repository import list_result_artifacts
from ..schemas import FactorDefinition

router = APIRouter(prefix="/factors", tags=["factors"])


@router.post("/validate")
def validate_factor(payload: FactorDefinition) -> dict:
    errors = validate_factor_definition(payload)
    return {"valid": not errors, "errors": errors, "factor_id": payload.factor_id, "version": payload.version.version}


@router.get("")
def list_factors(experiment_id: str | None = None, artifact_type: str | None = None, limit: int = 100) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    metadata = list_factor_metadata(min(max(limit, 1), 500))
    rows = list_result_artifacts(experiment_id, artifact_type, min(max(limit, 1), 500))
    artifacts = [row for row in rows if row["artifact_type"].startswith("factor_")]
    if experiment_id or artifact_type:
        return artifacts
    return metadata + artifacts
