from types import SimpleNamespace

from research_platform.backend.app import agent_feedback
from research_platform.backend.app.api import agents, reports_phase10


def test_next_plan_maps_feedback_to_whitelisted_tasks(monkeypatch):
    monkeypatch.setattr(agents, "settings", SimpleNamespace(storage_backend="mysql"))
    monkeypatch.setattr(agents, "list_sessions", lambda: [{"id": "session-1", "objective": "成本敏感性", "status": "completed", "plan": {"goal": {"budget": 8}}}])
    monkeypatch.setattr(agents, "build_agent_feedback", lambda _: {"suggestions": [{"action": "test_costs", "reason": "cost"}], "outcome": "completed", "completed": 1, "failed": 0, "runs": [], "metrics": {}})
    result = agents.agent_next_research_plan("session-1")
    assert result["requires_confirmation"] is True
    assert result["plan"]["tasks"] == ["factor_sensitivity"]
    assert result["plan"]["goal"]["budget"] == 8


def test_compare_experiments_aggregates_metrics(monkeypatch):
    monkeypatch.setattr(reports_phase10, "settings", SimpleNamespace(storage_backend="mysql"))
    monkeypatch.setattr(reports_phase10, "db_list_experiments", lambda: [{"id": "exp-1", "experiment_id": "e1", "objective": "质量"}])
    monkeypatch.setattr(reports_phase10, "list_runs", lambda _: [{"status": "succeeded", "artifacts": []}, {"status": "failed", "artifacts": []}])
    monkeypatch.setattr(reports_phase10, "read_run_metrics", lambda _: {"total_return_net": [0.1, 0.2], "max_drawdown": [0.1], "sharpe": [1.2], "turnover": [1.0]})
    result = reports_phase10.compare_experiments("exp-1")
    assert result["experiments"][0]["completed_runs"] == 1
    assert result["experiments"][0]["failed_runs"] == 1
    assert result["experiments"][0]["metrics"]["total_return_net"] == 0.15
    assert result["comparison"]["best_sharpe"] == "exp-1"


def test_markdown_report_contains_metrics_and_disclaimer(monkeypatch):
    monkeypatch.setattr(reports_phase10, "settings", SimpleNamespace(storage_backend="mysql"))
    monkeypatch.setattr(reports_phase10, "list_sessions", lambda: [{"id": "session-1", "objective": "验证动量", "status": "completed"}])
    monkeypatch.setattr(reports_phase10, "build_agent_feedback", lambda _: {"outcome": "completed", "completed": 1, "failed": 0, "metrics": {"sharpe": 1.2}, "runs": [{"task_type": "factor_quality", "status": "succeeded", "error": None}], "suggestions": [{"action": "candidate_review", "reason": "查看候选"}]})
    response = reports_phase10.agent_markdown_report("session-1")
    assert response.status_code == 200
    assert "# 研究结果报告" in response.body.decode("utf-8")
    assert "sharpe" in response.body.decode("utf-8")
    assert "不连接券商" in response.body.decode("utf-8")
