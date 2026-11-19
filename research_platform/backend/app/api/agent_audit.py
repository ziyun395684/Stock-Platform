"""Read-only Agent audit event API."""
from fastapi import APIRouter
from ..config import settings
from ..repositories import _session_factory
from ..models import AuditEvent
from sqlalchemy import select

router = APIRouter(prefix="/agent-audit", tags=["agent-audit"])

@router.get("")
def list_agent_audit(event_type: str | None = None, target_type: str | None = None, limit: int = 100) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    with _session_factory()() as session:
        query = select(AuditEvent)
        if event_type:
            query = query.where(AuditEvent.event_type == event_type)
        if target_type:
            query = query.where(AuditEvent.target_type == target_type)
        rows = session.scalars(query.order_by(AuditEvent.created_at.desc()).limit(min(max(limit, 1), 500))).all()
        return [{"id": row.id, "event_type": row.event_type, "target_type": row.target_type, "target_id": row.target_id, "actor": row.actor, "payload": row.payload_json, "created_at": row.created_at.isoformat() if row.created_at else None} for row in rows]
