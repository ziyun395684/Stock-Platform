"""Opt-in real MySQL API contract test.

Run with STOCKRNN_MYSQL_INTEGRATION=1 and PLATFORM_MYSQL_TEST_URL set. The
test creates uniquely named metadata rows and never drops tables.
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from research_platform.backend.app import database
from research_platform.backend.app.config import settings
from research_platform.backend.app.main import app


pytestmark = pytest.mark.skipif(
    os.getenv("STOCKRNN_MYSQL_INTEGRATION") != "1" or not os.getenv("PLATFORM_MYSQL_TEST_URL"),
    reason="set STOCKRNN_MYSQL_INTEGRATION=1 and PLATFORM_MYSQL_TEST_URL to run",
)


def test_api_round_trip_against_mysql() -> None:
    previous_backend, previous_url, previous_engine = settings.storage_backend, settings.database_url, database._engine
    object.__setattr__(settings, "storage_backend", "mysql")
    object.__setattr__(settings, "database_url", os.environ["PLATFORM_MYSQL_TEST_URL"])
    database._engine = None
    suffix = uuid4().hex[:10]
    try:
        with TestClient(app) as client:
            strategy = client.post("/api/strategies", json={"strategy_id": f"mysql-{suffix}", "name": "MySQL integration"})
            assert strategy.status_code == 201, strategy.text
            strategy_id = strategy.json()["id"]
            assert client.get("/api/strategies").status_code == 200
            experiment = client.post("/api/experiments", json={"experiment_id": f"mysql-exp-{suffix}", "objective": "API MySQL integration"})
            assert experiment.status_code == 201, experiment.text
            assert client.get("/api/experiments").status_code == 200
            assert client.get(f"/api/strategies/{strategy_id}").json()["strategy_id"] == f"mysql-{suffix}"
    finally:
        database.dispose_engine()
        object.__setattr__(settings, "storage_backend", previous_backend)
        object.__setattr__(settings, "database_url", previous_url)
        database._engine = previous_engine

