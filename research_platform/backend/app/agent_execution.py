"""Bridge governed Agent tasks to the single Celery task registry."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from .agent_repository import create_session, create_task, update_task_payload, update_task_status, update_session_plan, update_session_status
from .approval_repository import create_audit_event
from .repositories import create_experiment
from .run_repository import attach_task, create_run
from .task_registry import task_for


def submit_agent_task(task: dict[str, Any], *, actor: str = "agent") -> dict[str, Any]:
    task_type = str(task.get("task_type", ""))
    task_id = str(task.get("id", ""))
    if not task_id or not task_type:
        raise ValueError("agent task must contain id and task_type")
    if task.get("tool_name") != "submit_run":
        raise ValueError("only submit_run Agent tasks can be submitted")
    if task.get("status") not in {"planned", "queued"}:
        raise ValueError(f"agent task is not submit-ready: {task.get('status')}")
    if task.get("status") == "queued":
        existing_task_id = (task.get("payload") or {}).get("celery_task_id")
        return {"agent_task_id": task_id, "task_type": task_type, "run_id": (task.get("payload") or {}).get("run_id"), "celery_task_id": existing_task_id, "status": "queued", "idempotent": True}
    task_fn = task_for(task_type)
    payload = dict(task.get("payload") or {})
    run_id = payload.get("run_id")
    if not run_id:
        raise ValueError("agent submit_run payload requires run_id")
    config = dict(payload.get("config") or {})
    queued = task_fn.apply_async(args=[run_id, config])
    payload_with_task = {**payload, "celery_task_id": queued.id}
    update_task_status(task_id, "queued")
    update_task_payload(task_id, payload_with_task)
    create_audit_event("agent_task_submitted", "agent_task", task_id, actor, {"task_type": task_type, "celery_task_id": queued.id, "run_id": run_id})
    return {"agent_task_id": task_id, "task_type": task_type, "run_id": run_id, "celery_task_id": queued.id, "status": "queued", "idempotent": False}


def execute_research_plan(plan: dict[str, Any], *, actor: str = "agent") -> dict[str, Any]:
    """Persist a confirmed plan and enqueue one governed run per task."""
    tasks = list(dict.fromkeys(str(item) for item in plan.get("tasks", [])))
    budget = int(plan.get("goal", {}).get("budget", 0))
    if not tasks:
        raise ValueError("research plan contains no tasks")
    if len(tasks) > budget:
        raise ValueError("research plan exceeds budget")
    for task_type in tasks:
        task_for(task_type)
    objective = str(plan.get("goal", {}).get("objective", "Agent research plan"))
    session = create_session(objective, actor)
    update_session_plan(session["id"], plan, "confirmed")
    update_session_status(session["id"], "running")
    queued: list[dict[str, Any]] = []
    for index, task_type in enumerate(tasks, start=1):
        suffix = uuid4().hex[:10]
        experiment = create_experiment({"experiment_id": f"agent-{session['id'][:8]}-{index}-{suffix}", "objective": objective, "config": {"source": "agent-plan", "agent_session_id": session["id"], "task_type": task_type}, "created_by": actor})
        config = {"source": "agent-plan", "agent_session_id": session["id"], "objective": objective, "data_dir": "quotes2"}
        run = create_run(experiment["id"], f"agent:{session['id']}:{task_type}:{suffix}", config)
        agent_task = create_task(session["id"], task_type, "submit_run", {"run_id": run["id"], "config": config})
        submission = submit_agent_task(agent_task, actor=actor)
        attach_task(run["id"], submission["celery_task_id"])
        queued.append({"task_id": agent_task["id"], "task_type": task_type, "experiment": experiment, "run": {**run, "task_id": submission["celery_task_id"]}, "submission": submission})
    create_audit_event("agent_plan_executed", "agent_session", session["id"], actor, {"task_count": len(queued), "task_types": tasks})
    return {"session": {**session, "status": "running", "plan": plan}, "queued": queued, "status": "running"}

