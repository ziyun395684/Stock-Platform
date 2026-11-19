from types import SimpleNamespace
from pathlib import Path

import pytest

from research_platform.backend.app import agent_execution
from research_platform.backend.app import agent_feedback
from research_platform.backend.app.api import agents


def test_agent_submission_maps_to_registered_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    fake = SimpleNamespace(apply_async=lambda args: (calls.append(tuple(args)) or SimpleNamespace(id="celery-1")))
    monkeypatch.setattr(agent_execution, "task_for", lambda name: fake)
    monkeypatch.setattr(agent_execution, "update_task_status", lambda *args: calls.append(args))
    monkeypatch.setattr(agent_execution, "update_task_payload", lambda *args: calls.append(args))
    monkeypatch.setattr(agent_execution, "create_audit_event", lambda *args: calls.append(args))
    result = agent_execution.submit_agent_task({"id": "agent-1", "task_type": "factor_compute", "tool_name": "submit_run", "status": "planned", "payload": {"run_id": "run-1", "config": {"start": "2024-01-01"}}})
    assert result["celery_task_id"] == "celery-1"
    assert calls[0] == ("run-1", {"start": "2024-01-01"})


def test_agent_submission_requires_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        agent_execution.submit_agent_task({"id": "agent-1", "task_type": "factor_compute", "tool_name": "submit_run", "status": "planned", "payload": {}})


def test_agent_submission_rejects_non_submit_tool() -> None:
    with pytest.raises(ValueError, match="submit_run"):
        agent_execution.submit_agent_task({"id": "agent-1", "task_type": "factor_compute", "tool_name": "list_factors", "status": "planned", "payload": {"run_id": "run-1"}})


def test_execute_research_plan_creates_and_queues_one_run_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []
    monkeypatch.setattr(agent_execution, "create_session", lambda objective, actor: {"id": "session-1", "objective": objective, "status": "draft", "created_by": actor})
    monkeypatch.setattr(agent_execution, "update_session_plan", lambda *args: None)
    monkeypatch.setattr(agent_execution, "update_session_status", lambda *args: None)
    monkeypatch.setattr(agent_execution, "create_experiment", lambda payload: {"id": f"exp-{len(created)}", **payload})
    monkeypatch.setattr(agent_execution, "create_run", lambda experiment_id, run_key, config: {"id": f"run-{len(created)}", "experiment_id": experiment_id, "run_key": run_key, "status": "queued", "artifacts": []})
    monkeypatch.setattr(agent_execution, "create_task", lambda session_id, task_type, tool_name, payload: {"id": f"task-{len(created)}", "session_id": session_id, "task_type": task_type, "tool_name": tool_name, "status": "planned", "payload": payload})
    monkeypatch.setattr(agent_execution, "submit_agent_task", lambda task, actor="agent": {"celery_task_id": f"celery-{task['id']}", "status": "queued"})
    monkeypatch.setattr(agent_execution, "attach_task", lambda *args: None)
    monkeypatch.setattr(agent_execution, "create_audit_event", lambda *args: None)
    result = agent_execution.execute_research_plan({"goal": {"objective": "factor quality", "budget": 2}, "tasks": ["factor_quality", "factor_correlation"], "estimated_runs": 2})
    assert result["status"] == "running"
    assert [item["task_type"] for item in result["queued"]] == ["factor_quality", "factor_correlation"]


def test_agent_session_state_aggregates_progress_and_suggestions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agents, "settings", SimpleNamespace(storage_backend="mysql"))
    monkeypatch.setattr(agents, "list_sessions", lambda: [{"id": "session-1", "objective": "factor quality", "status": "running"}])
    monkeypatch.setattr(agents, "list_tasks", lambda session_id: [
        {"id": "task-1", "session_id": session_id, "task_type": "factor_quality", "tool_name": "submit_run", "status": "succeeded", "payload": {}},
        {"id": "task-2", "session_id": session_id, "task_type": "factor_correlation", "tool_name": "submit_run", "status": "failed", "payload": {"error": "worker unavailable"}},
    ])
    result = agents.agent_session_state("session-1")
    assert result["summary"] == {"task_count": 2, "completed": 1, "failed": 1, "statuses": {"succeeded": 1, "failed": 1}}
    assert result["suggestions"][0]["action"] == "retry"


def test_agent_feedback_suggests_risk_reduction_for_high_drawdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_feedback, "list_tasks", lambda session_id: [{"id": "task-1", "task_type": "factor_quality", "status": "succeeded", "payload": {"run_id": "run-1", "config": {"objective": "质量因子"}}}])
    monkeypatch.setattr(agent_feedback, "get_run", lambda run_id: {"id": run_id, "status": "succeeded", "error_message": None, "artifacts": []})
    monkeypatch.setattr(agent_feedback, "_read_run_metrics", lambda run: {"total_return_net": [-0.02], "max_drawdown": [0.35]})
    result = agent_feedback.build_agent_feedback("session-1")
    assert result["outcome"] == "completed"
    assert result["metrics"]["max_drawdown"] == 0.35
    assert {item["action"] for item in result["suggestions"]} >= {"reduce_risk", "adjust_hypothesis"}


def test_agent_feedback_reads_csv_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    report.write_text("total_return_net,max_drawdown,turnover\n0.12,0.08,1.4\n", encoding="utf-8")
    monkeypatch.setattr(agent_feedback, "_artifact_root", lambda: tmp_path)
    metrics = agent_feedback._read_run_metrics({"artifacts": [{"relative_path": "report.csv"}]})
    assert metrics == {"total_return_net": [0.12], "max_drawdown": [0.08], "turnover": [1.4]}

