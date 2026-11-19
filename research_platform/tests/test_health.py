from __future__ import annotations

from fastapi.testclient import TestClient

from research_platform.backend.app.main import app


def test_health_does_not_require_external_services() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_runtime_config_does_not_expose_database_url() -> None:
    client = TestClient(app)
    response = client.get("/api/runtime-config")
    assert response.status_code == 200
    payload = response.json()
    assert "database_url" not in payload
    assert "database_configured" in payload
