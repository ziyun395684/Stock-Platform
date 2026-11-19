from __future__ import annotations

from fastapi.testclient import TestClient

from research_platform.backend.app.main import app
from research_platform.backend.app.memory_store import store


client = TestClient(app)


def setup_function() -> None:
    store.strategies.clear()
    store.experiments.clear()


def test_frontend_strategy_experiment_contract_round_trip() -> None:
    strategy = client.post("/api/strategies", json={"strategy_id": "ui-contract", "name": "UI Contract"})
    assert strategy.status_code == 201
    strategy_payload = strategy.json()
    assert {"id", "strategy_id", "name", "status", "versions"}.issubset(strategy_payload)

    experiment = client.post("/api/experiments", json={"experiment_id": "ui-exp", "objective": "contract"})
    assert experiment.status_code == 201
    assert {"id", "experiment_id", "objective", "status", "config"}.issubset(experiment.json())
    assert client.get("/api/strategies").status_code == 200
    assert client.get("/api/experiments").status_code == 200


def test_frontend_validation_error_uses_stable_error_shape() -> None:
    response = client.post("/api/strategies", json={"name": "missing id"})
    assert response.status_code == 422
    payload = response.json()["error"]
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["path"] == "/api/strategies"
    assert payload["details"]


def test_frontend_agent_plan_contract() -> None:
    plan = client.post("/api/agents/plan/draft", json={"objective": "factor quality", "budget": 4})
    assert plan.status_code == 200
    payload = plan.json()
    assert {"goal", "tasks", "estimated_runs", "requires_approval"}.issubset(payload)
    validation = client.post("/api/agents/plan/validate", json=payload)
    assert validation.status_code == 200
    assert {"valid", "estimated_runs", "budget", "requires_approval"}.issubset(validation.json())

