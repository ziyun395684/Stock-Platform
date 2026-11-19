from __future__ import annotations

from fastapi.testclient import TestClient

from research_platform.backend.app.main import app
from research_platform.backend.app.memory_store import store


client = TestClient(app)


def setup_function() -> None:
    store.strategies.clear()
    store.experiments.clear()


def test_create_strategy_and_version() -> None:
    strategy_response = client.post(
        "/api/strategies",
        json={"strategy_id": "factor_common_v2", "name": "Factor Common v2"},
    )
    assert strategy_response.status_code == 201
    strategy = strategy_response.json()

    version_response = client.post(
        f"/api/strategies/{strategy['id']}/versions",
        json={"version": "v1", "config": {"top_k": 2}},
    )
    assert version_response.status_code == 201
    assert version_response.json()["strategy_id"] == strategy["id"]


def test_frozen_strategy_is_read_only() -> None:
    strategy_response = client.post(
        "/api/strategies",
        json={
            "strategy_id": "frozen_baseline_copy",
            "name": "Frozen baseline copy",
            "status": "draft",
        },
    )
    strategy = strategy_response.json()
    store.strategies[strategy["id"]]["status"] = "frozen"

    response = client.post(
        f"/api/strategies/{strategy['id']}/versions",
        json={"version": "v2", "config": {}},
    )
    assert response.status_code == 409


def test_strategy_status_transition_and_illegal_transition() -> None:
    strategy = client.post("/api/strategies", json={"strategy_id": "lifecycle-v1", "name": "Lifecycle"}).json()
    response = client.post(f"/api/strategies/{strategy['id']}/transition", json={"status": "research"})
    assert response.status_code == 200
    assert response.json()["status"] == "research"
    illegal = client.post(f"/api/strategies/{strategy['id']}/transition", json={"status": "frozen"})
    assert illegal.status_code == 409


def test_strategy_approvals_empty_in_memory() -> None:
    strategy = client.post("/api/strategies", json={"strategy_id": "approval-query", "name": "Approval query"}).json()
    response = client.get(f"/api/strategies/{strategy['id']}/approvals")
    assert response.status_code == 200
    assert response.json() == []


def test_frozen_strategy_clone_creates_research_child() -> None:
    source = client.post("/api/strategies", json={"strategy_id": "frozen-source", "name": "Source"}).json()
    store.strategies[source["id"]]["status"] = "frozen"
    response = client.post(f"/api/strategies/{source['id']}/clone", json={"strategy_id": "child", "name": "Child"})
    assert response.status_code == 201
    assert response.json()["status"] == "research"
    assert response.json()["parent_strategy_id"] == source["id"]


def test_independent_retest_requires_mysql_backend_in_contract_mode() -> None:
    source = client.post("/api/strategies", json={"strategy_id": "frozen-retest", "name": "Frozen retest"}).json()
    store.strategies[source["id"]]["status"] = "frozen"
    response = client.post(
        f"/api/strategies/{source['id']}/independent-oos",
        json={"start": "2026-01-01", "end": "2026-03-31"},
    )
    assert response.status_code == 503


def test_experiment_rejects_unknown_strategy_version() -> None:
    response = client.post(
        "/api/experiments",
        json={
            "experiment_id": "exp-001",
            "objective": "test factor research",
            "strategy_version_id": "missing-version",
        },
    )
    assert response.status_code == 404
