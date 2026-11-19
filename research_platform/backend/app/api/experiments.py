"""Experiment API routes."""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..memory_store import store
from ..repositories import (
    create_experiment as db_create_experiment,
    list_experiments as db_list_experiments,
)
from ..schemas import ExperimentCreate, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments() -> list[dict]:
    return db_list_experiments() if settings.storage_backend == "mysql" else store.list_experiments()


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate) -> dict:
    if any(item["experiment_id"] == payload.experiment_id for item in list_experiments()):
        raise HTTPException(status_code=409, detail="experiment_id already exists")
    if payload.strategy_version_id:
        from ..api.strategies import list_strategies

        known_versions = {
            version["id"]
            for strategy in list_strategies()
            for version in strategy["versions"]
        }
        if payload.strategy_version_id not in known_versions:
            raise HTTPException(status_code=404, detail="strategy version not found")
    record = {"id": str(uuid4()), **payload.model_dump(), "status": "draft"}
    if settings.storage_backend == "mysql":
        record.pop("id")
        return db_create_experiment(record)
    return store.create_experiment(record)
