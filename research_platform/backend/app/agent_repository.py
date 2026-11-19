"""Persistence for governed Agent sessions and tasks."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select

from .models import AgentSession, AgentTask
from .repositories import _session_factory
from .task_registry import task_for


ALLOWED_TOOLS = frozenset({"list_factors", "list_reports", "list_models", "get_artifact", "create_experiment", "submit_run"})
FORBIDDEN_TOOL_PREFIXES = ("broker", "order", "trade", "place_order", "execute_order")


def is_tool_allowed(tool_name: str | None) -> bool:
    """Keep Agent execution inside the research API allowlist."""
    if not tool_name:
        return True
    normalized = tool_name.strip().lower()
    if any(normalized == prefix or normalized.startswith(f"{prefix}.") or normalized.startswith(f"{prefix}_") for prefix in FORBIDDEN_TOOL_PREFIXES):
        return False
    return normalized in ALLOWED_TOOLS


def update_session_plan(session_id: str, plan: dict[str, Any], status: str = "confirmed") -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.get(AgentSession, session_id)
        if row is None:
            return None
        row.plan_json = plan
        row.status = status
        session.commit()
        return {"id": row.id, "objective": row.objective, "status": row.status, "created_by": row.created_by, "plan": row.plan_json}


def create_session(objective: str, created_by: str | None) -> dict[str, Any]:
    with _session_factory()() as session:
        row = AgentSession(id=str(uuid4()), objective=objective, created_by=created_by, status="draft")
        session.add(row)
        session.commit()
        return {"id": row.id, "objective": row.objective, "status": row.status, "created_by": row.created_by}


def list_sessions() -> list[dict[str, Any]]:
    with _session_factory()() as session:
        rows = session.scalars(select(AgentSession).order_by(AgentSession.created_at.desc())).all()
        return [{"id": row.id, "objective": row.objective, "status": row.status, "created_by": row.created_by, "plan": row.plan_json} for row in rows]


def update_session_status(session_id: str, status: str) -> dict[str, Any] | None:
    allowed = {"draft", "confirmed", "running", "completed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"invalid agent session status: {status}")
    with _session_factory()() as session:
        row = session.get(AgentSession, session_id)
        if row is None:
            return None
        row.status = status
        session.commit()
        from .approval_repository import create_audit_event
        create_audit_event("agent_session_status_changed", "agent_session", session_id, None, {"status": status})
        return {"id": row.id, "objective": row.objective, "status": row.status, "created_by": row.created_by, "plan": row.plan_json}


def update_task_status(task_id: str, status: str, error: str | None = None) -> dict[str, Any] | None:
    allowed = {"planned", "queued", "running", "succeeded", "failed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"invalid agent task status: {status}")
    with _session_factory()() as session:
        row = session.get(AgentTask, task_id)
        if row is None:
            return None
        row.status = status
        payload = dict(row.payload_json or {})
        if error:
            payload["error"] = error
        row.payload_json = payload
        session.commit()
        from .approval_repository import create_audit_event
        create_audit_event("agent_task_status_changed", "agent_task", task_id, None, {"status": status, "error": error})
        return {"id": row.id, "session_id": row.session_id, "task_type": row.task_type, "tool_name": row.tool_name, "status": row.status, "payload": row.payload_json}


def update_task_payload(task_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.get(AgentTask, task_id)
        if row is None:
            return None
        row.payload_json = payload
        session.commit()
        return {"id": row.id, "session_id": row.session_id, "task_type": row.task_type, "tool_name": row.tool_name, "status": row.status, "payload": row.payload_json}


def list_tasks(session_id: str) -> list[dict[str, Any]]:
    with _session_factory()() as session:
        rows = session.scalars(select(AgentTask).where(AgentTask.session_id == session_id).order_by(AgentTask.created_at)).all()
        return [{"id": row.id, "session_id": row.session_id, "task_type": row.task_type, "tool_name": row.tool_name, "status": row.status, "payload": row.payload_json} for row in rows]


def get_task(task_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.get(AgentTask, task_id)
        if row is None:
            return None
        return {"id": row.id, "session_id": row.session_id, "task_type": row.task_type, "tool_name": row.tool_name, "status": row.status, "payload": row.payload_json}


def create_task(session_id: str, task_type: str, tool_name: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name and not is_tool_allowed(tool_name):
        raise ValueError(f"tool is not allowed: {tool_name}")
    if tool_name == "submit_run":
        task_for(task_type)
    with _session_factory()() as session:
        if session.scalar(select(AgentSession).where(AgentSession.id == session_id)) is None:
            raise LookupError("agent session not found")
        row = AgentTask(id=str(uuid4()), session_id=session_id, task_type=task_type, tool_name=tool_name, payload_json=payload, status="planned")
        session.add(row)
        session.commit()
        from .approval_repository import create_audit_event
        create_audit_event("agent_task_created", "agent_task", row.id, None, {"session_id": session_id, "task_type": task_type, "tool_name": tool_name})
        return {"id": row.id, "session_id": row.session_id, "task_type": row.task_type, "tool_name": row.tool_name, "status": row.status, "payload": row.payload_json}
