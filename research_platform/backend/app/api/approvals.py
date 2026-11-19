"""Human approval API; it never auto-freezes a strategy."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..approval_repository import create_approval, list_approvals
from ..config import settings
from ..schemas import ApprovalCreate, ApprovalResponse

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("", response_model=list[ApprovalResponse])
def read_approvals(target_type: str | None = None, target_id: str | None = None, reviewer: str | None = None) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_approvals(target_type, target_id, reviewer)


@router.post("", response_model=ApprovalResponse, status_code=status.HTTP_201_CREATED)
def submit_approval(payload: ApprovalCreate) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="approvals require MySQL storage")
    try:
        return create_approval(payload.target_type, payload.target_id, payload.decision, payload.reviewer, payload.comment, payload.report_artifact_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
