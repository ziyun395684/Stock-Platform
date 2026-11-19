"""Approval and audit-event persistence."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select

from .models import Approval, AuditEvent, Artifact
from .repositories import _session_factory


def create_approval(target_type: str, target_id: str, decision: str, reviewer: str, comment: str | None, report_artifact_ids: list[str]) -> dict[str, Any]:
    with _session_factory()() as session:
        if report_artifact_ids:
            found = set(session.scalars(select(Artifact.id).where(Artifact.id.in_(report_artifact_ids))).all())
            missing = sorted(set(report_artifact_ids) - found)
            if missing:
                raise ValueError(f"report artifacts not found: {missing}")
        row = Approval(id=str(uuid4()), target_type=target_type, target_id=target_id, decision=decision, reviewer=reviewer, comment=comment, report_artifact_ids=report_artifact_ids)
        session.add(row)
        session.commit()
        return {"id": row.id, "target_type": row.target_type, "target_id": row.target_id, "decision": row.decision, "reviewer": row.reviewer, "comment": row.comment, "report_artifact_ids": row.report_artifact_ids or []}


def list_approvals(target_type: str | None = None, target_id: str | None = None, reviewer: str | None = None) -> list[dict[str, Any]]:
    with _session_factory()() as session:
        query = select(Approval)
        if target_type:
            query = query.where(Approval.target_type == target_type)
        if target_id:
            query = query.where(Approval.target_id == target_id)
        if reviewer:
            query = query.where(Approval.reviewer == reviewer)
        rows = session.scalars(query.order_by(Approval.created_at.desc())).all()
        return [{"id": row.id, "target_type": row.target_type, "target_id": row.target_id, "decision": row.decision, "reviewer": row.reviewer, "comment": row.comment, "report_artifact_ids": row.report_artifact_ids or []} for row in rows]


def list_approvals_for_target(target_id: str) -> list[dict[str, Any]]:
    return list_approvals(target_id=target_id)


def has_decision(target_id: str, decision: str) -> bool:
    return any(row["decision"] == decision for row in list_approvals_for_target(target_id))


def create_audit_event(event_type: str, target_type: str, target_id: str, actor: str | None, payload: dict[str, Any]) -> None:
    with _session_factory()() as session:
        session.add(AuditEvent(id=str(uuid4()), event_type=event_type, target_type=target_type, target_id=target_id, actor=actor, payload_json=payload))
        session.commit()
