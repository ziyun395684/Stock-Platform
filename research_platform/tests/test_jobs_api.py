from fastapi.testclient import TestClient

from research_platform.backend.app.main import app


def test_jobs_are_empty_in_memory_test_mode() -> None:
    response = TestClient(app).get("/api/jobs")
    assert response.status_code == 200
    assert response.json() == []


def test_unknown_job_is_not_found() -> None:
    response = TestClient(app).get("/api/jobs/missing")
    assert response.status_code == 404


def test_research_plan_cancel_endpoint() -> None:
    client = TestClient(app)
    response = client.post("/api/agents/plan/cancel", json={"goal": {"objective": "test", "budget": 2}, "estimated_runs": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_research_plan_confirm_endpoint() -> None:
    client = TestClient(app)
    ok = client.post("/api/agents/plan/confirm", json={"goal": {"objective": "test", "budget": 2}, "estimated_runs": 1})
    assert ok.status_code == 200
    assert ok.json()["status"] == "confirmed"
    bad = client.post("/api/agents/plan/confirm", json={"goal": {"objective": "test", "budget": 1}, "estimated_runs": 2})
    assert bad.status_code == 409


def test_research_plan_budget_endpoint() -> None:
    client = TestClient(app)
    response = client.post("/api/agents/plan/validate", json={"goal": {"objective": "test", "budget": 2}, "estimated_runs": 3})
    assert response.status_code == 200
    assert response.json()["valid"] is False


def test_research_goal_and_plan_schema() -> None:
    from research_platform.backend.app.schemas import ResearchGoal, ResearchPlan
    plan = ResearchPlan(goal=ResearchGoal(objective="factor quality"), tasks=["factor_quality"], estimated_runs=2)
    assert plan.requires_approval is True
    assert plan.estimated_runs == 2


def test_factor_and_report_queries_are_empty_without_mysql() -> None:
    client = TestClient(app)
    assert client.get("/api/factors").status_code == 200
    assert client.get("/api/factors").json() == []
    assert client.get("/api/reports").status_code == 200
    assert client.get("/api/reports").json() == []
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/models").json() == []
    assert client.get("/api/approvals").status_code == 200
    assert client.get("/api/approvals").json() == []
    assert client.post("/api/approvals", json={"target_type":"strategy","target_id":"s","decision":"approved","reviewer":"qa","report_artifact_ids":["missing"]}).status_code == 503
    assert client.post("/api/agents/sessions", json={"objective": "test"}).status_code == 503
    assert client.get("/api/agents/sessions").json() == []
    assert client.get("/api/agents/sessions/missing/tasks").json() == []
    assert client.get("/api/agents/sessions/missing/summary").json()["task_count"] == 0
    assert client.get("/api/agents/sessions/missing/next-steps").json()["suggestions"] == []
    assert client.patch("/api/agents/sessions/missing/status?status=cancelled").status_code == 503
    assert client.get("/api/agent-audit?event_type=agent_task_created&target_type=agent_task").json() == []
    assert client.patch("/api/agents/tasks/missing/status?status=running").status_code == 503


def test_unknown_report_and_retry_are_not_found() -> None:
    client = TestClient(app)
    assert client.get("/api/jobs/missing/report").status_code == 404
    assert client.post("/api/jobs/missing/retry").status_code == 404


def test_data_freshness_endpoint() -> None:
    client = TestClient(app)
    response = client.post("/api/operations/data-freshness", json={"latest_date": "2026-11-17", "expected_date": "2026-11-18"})
    assert response.status_code == 200
    assert response.json()["healthy"] is True
