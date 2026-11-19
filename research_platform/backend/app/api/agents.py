"""Governed Agent planning API; no arbitrary code execution."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..agent_repository import create_session, create_task, get_task, list_sessions, list_tasks, update_task_status, update_session_plan, update_session_status
from ..agent_execution import execute_research_plan, submit_agent_task
from ..agent_feedback import build_agent_feedback
from ..config import settings
from ..research_planner import draft_research_plan
from ..experiment_designer import design_experiments
from ..research_reviewer import review_research
from ..schemas import AgentSessionCreate, AgentSessionResponse, AgentTaskCreate, AgentTaskResponse, ExperimentDesignRequest, ExperimentDesignResponse, ResearchGoal, ResearchPlan, ResearchReviewRequest, ResearchReviewResponse
from ..task_registry import registered_task_names

router = APIRouter(prefix="/agents", tags=["agents"])


@router.patch("/sessions/{session_id}/status")
def change_agent_session_status(session_id: str, status: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent sessions require MySQL storage")
    try:
        result = update_session_status(session_id, status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="agent session not found")
    return result


@router.patch("/tasks/{task_id}/status")
def change_agent_task_status(task_id: str, status: str, error: str | None = None) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent tasks require MySQL storage")
    try:
        result = update_task_status(task_id, status, error)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="agent task not found")
    return result


@router.get("/sessions")
def read_agent_sessions() -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_sessions()


@router.get("/sessions/{session_id}/tasks")
def read_agent_tasks(session_id: str) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_tasks(session_id)


@router.get("/sessions/{session_id}/next-steps")
def agent_next_steps(session_id: str) -> dict:
    if settings.storage_backend != "mysql":
        return {"session_id": session_id, "suggestions": []}
    tasks = list_tasks(session_id)
    failed = [task for task in tasks if task["status"] == "failed"]
    suggestions = [{"task_id": task["id"], "action": "retry", "reason": "任务失败，可检查错误后重跑；重跑会创建新 run 并保留旧 artifact"} for task in failed]
    if not failed and tasks and all(task["status"] == "succeeded" for task in tasks):
        suggestions.append({"action": "review", "reason": "任务全部完成，建议人工查看报告后提交审批"})
    return {"session_id": session_id, "suggestions": suggestions}


@router.get("/sessions/{session_id}/summary")
def agent_session_summary(session_id: str) -> dict:
    if settings.storage_backend != "mysql":
        return {"session_id": session_id, "task_count": 0, "statuses": {}}
    tasks = list_tasks(session_id)
    statuses: dict[str, int] = {}
    for task in tasks:
        statuses[task["status"]] = statuses.get(task["status"], 0) + 1
    return {"session_id": session_id, "task_count": len(tasks), "statuses": statuses, "completed": statuses.get("succeeded", 0), "failed": statuses.get("failed", 0)}


@router.get("/sessions/{session_id}/state")
def agent_session_state(session_id: str) -> dict:
    """Return one compact polling payload for the Agent Console."""
    if settings.storage_backend != "mysql":
        return {"session_id": session_id, "session": None, "tasks": [], "summary": {"task_count": 0, "completed": 0, "failed": 0, "statuses": {}}, "suggestions": []}
    session = next((item for item in list_sessions() if item["id"] == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail="agent session not found")
    tasks = list_tasks(session_id)
    statuses: dict[str, int] = {}
    for task in tasks:
        statuses[task["status"]] = statuses.get(task["status"], 0) + 1
    failed = [task for task in tasks if task["status"] == "failed"]
    suggestions = [{"action": "retry", "task_id": task["id"], "reason": "任务失败，可检查错误后重跑；旧运行和产物会保留"} for task in failed]
    if tasks and not failed and all(task["status"] == "succeeded" for task in tasks):
        suggestions.append({"action": "review", "reason": "任务全部完成，建议查看报告并进行人工审批"})
    return {
        "session_id": session_id,
        "session": session,
        "tasks": tasks,
        "summary": {"task_count": len(tasks), "completed": statuses.get("succeeded", 0), "failed": statuses.get("failed", 0), "statuses": statuses},
        "suggestions": suggestions,
    }


@router.get("/sessions/{session_id}/feedback")
def agent_session_feedback(session_id: str) -> dict:
    if settings.storage_backend != "mysql":
        return {"session_id": session_id, "outcome": "running", "completed": 0, "failed": 0, "runs": [], "metrics": {}, "suggestions": []}
    if not any(item["id"] == session_id for item in list_sessions()):
        raise HTTPException(status_code=404, detail="agent session not found")
    return build_agent_feedback(session_id)


@router.post("/sessions/{session_id}/next-plan")
def agent_next_research_plan(session_id: str) -> dict:
    """Build a governed, user-confirmed plan from the latest feedback."""
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="next research plans require MySQL storage")
    session = next((item for item in list_sessions() if item["id"] == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail="agent session not found")
    feedback = build_agent_feedback(session_id)
    mapping = {
        "adjust_hypothesis": ["factor_quality", "factor_common_oos"],
        "reduce_risk": ["factor_sensitivity", "factor_common_oos"],
        "test_costs": ["factor_sensitivity"],
        "candidate_review": ["factor_common_oos", "strategy_gate"],
        "retry_failed": [],
    }
    tasks: list[str] = []
    for suggestion in feedback["suggestions"]:
        if suggestion["action"] == "retry_failed":
            for failed_run in feedback["runs"]:
                task_type = failed_run.get("task_type")
                if failed_run.get("status") == "failed" and task_type in registered_task_names() and task_type not in tasks:
                    tasks.append(task_type)
        for task_type in mapping.get(suggestion["action"], []):
            if task_type in registered_task_names() and task_type not in tasks:
                tasks.append(task_type)
    if not tasks:
        tasks = ["factor_common_oos", "factor_sensitivity"]
    budget = int((session.get("plan") or {}).get("goal", {}).get("budget") or max(4, len(tasks)))
    plan = ResearchPlan(goal=ResearchGoal(objective=f"基于上一轮研究结果继续：{session['objective']}", constraints=[f"parent_session_id={session_id}"], budget=budget), tasks=tasks, estimated_runs=len(tasks), requires_approval=True)
    return {"session_id": session_id, "parent_session_id": session_id, "feedback": feedback, "plan": plan.model_dump(), "requires_confirmation": True}


@router.post("/plan/validate")
def validate_plan(plan: ResearchPlan) -> dict:
    within_budget = plan.estimated_runs <= plan.goal.budget
    return {"valid": within_budget, "estimated_runs": plan.estimated_runs, "budget": plan.goal.budget, "requires_approval": plan.requires_approval, "reason": None if within_budget else "estimated_runs exceeds research budget"}


@router.post("/plan/draft", response_model=ResearchPlan)
def draft_plan(goal: ResearchGoal) -> ResearchPlan:
    return draft_research_plan(goal)


@router.post("/plan/design", response_model=ExperimentDesignResponse)
def design_plan(request: ExperimentDesignRequest) -> ExperimentDesignResponse:
    return design_experiments(request)


@router.post("/review", response_model=ResearchReviewResponse)
def review_plan(request: ResearchReviewRequest) -> ResearchReviewResponse:
    return review_research(request)


@router.post("/plan/cancel")
def cancel_plan(plan: ResearchPlan) -> dict:
    return {"status": "cancelled", "objective": plan.goal.objective, "tasks": plan.tasks, "estimated_runs": plan.estimated_runs}


@router.post("/plan/confirm")
def confirm_plan(plan: ResearchPlan, session_id: str | None = None) -> dict:
    within_budget = plan.estimated_runs <= plan.goal.budget
    if not within_budget:
        raise HTTPException(status_code=409, detail="research plan exceeds budget")
    result = {"status": "confirmed", "objective": plan.goal.objective, "tasks": plan.tasks, "estimated_runs": plan.estimated_runs, "requires_approval": plan.requires_approval}
    if session_id and settings.storage_backend == "mysql":
        saved = update_session_plan(session_id, plan.model_dump(), "confirmed")
        if saved is None:
            raise HTTPException(status_code=404, detail="agent session not found")
        result["session_id"] = session_id
    return result


@router.post("/plan/execute")
def execute_plan(plan: ResearchPlan) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent plan execution requires MySQL storage")
    if plan.estimated_runs != len(plan.tasks):
        raise HTTPException(status_code=422, detail="estimated_runs must equal the number of plan tasks")
    try:
        return execute_research_plan(plan.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="unable to execute agent research plan") from exc


@router.post("/sessions", response_model=AgentSessionResponse, status_code=status.HTTP_201_CREATED)
def create_agent_session(payload: AgentSessionCreate) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent sessions require MySQL storage")
    return create_session(payload.objective, payload.created_by)


@router.post("/sessions/{session_id}/tasks", response_model=AgentTaskResponse, status_code=status.HTTP_201_CREATED)
def create_agent_task(session_id: str, payload: AgentTaskCreate) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent tasks require MySQL storage")
    try:
        return create_task(session_id, payload.task_type, payload.tool_name, payload.payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/submit")
def submit_agent_task_run(task_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="agent tasks require MySQL storage")
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="agent task not found")
    try:
        return submit_agent_task(task)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
