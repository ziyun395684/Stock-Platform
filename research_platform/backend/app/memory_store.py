"""Small in-memory store for API smoke tests before MySQL is connected.

This is intentionally non-persistent and must not be used for production.
It lets the frontend/API contract be exercised while the external database is
not configured.
"""
from __future__ import annotations

from threading import RLock
from typing import Any


class MemoryStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self.strategies: dict[str, dict[str, Any]] = {}
        self.experiments: dict[str, dict[str, Any]] = {}

    def list_strategies(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.strategies.values())

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.strategies.get(strategy_id)

    def create_strategy(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.strategies[payload["id"]] = payload
            return payload

    def update_strategy_status(self, strategy_id: str, status: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.strategies.get(strategy_id)
            if record is not None:
                record["status"] = status
            return record

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.experiments.values())

    def create_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.experiments[payload["id"]] = payload
            return payload


store = MemoryStore()
