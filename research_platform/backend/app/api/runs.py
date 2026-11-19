"""Asynchronous experiment-run API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..run_repository import attach_task, create_run, get_run, get_run_by_key, list_runs, update_run
from ..schemas import ExperimentRunCreate, ExperimentRunResponse
from ..task_registry import task_for

router = APIRouter(prefix="/experiments/{experiment_id}/runs", tags=["runs"])


@router.post("", response_model=ExperimentRunResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_run(experiment_id: str, payload: ExperimentRunCreate) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="async runs require MySQL storage")
    idempotency_key = payload.config.get("idempotency_key")
    effective_run_key = f"{experiment_id}:{idempotency_key}" if idempotency_key else payload.run_key
    if idempotency_key:
        existing = get_run_by_key(experiment_id, effective_run_key)
        if existing is not None:
            return existing
    try:
        run = create_run(experiment_id, effective_run_key, payload.config)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if "Duplicate" in type(exc).__name__ or "duplicate" in str(exc).lower():
            raise HTTPException(status_code=409, detail="run_key already exists") from exc
        raise
    try:
        task_fn = task_for(payload.task_type)
        task = task_fn.apply_async(args=[run["id"], payload.config])
        attach_task(run["id"], task.id)
    except Exception as exc:
        update_run(run["id"], "failed", error=exc)
        raise HTTPException(status_code=503, detail="unable to enqueue research task") from exc
    run["task_id"] = task.id
    return run


@router.get("", response_model=list[ExperimentRunResponse])
def read_runs(experiment_id: str, limit: int = 50) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_runs(experiment_id, limit=min(max(limit, 1), 100))


@router.get("/{run_id}", response_model=ExperimentRunResponse)
def read_run(experiment_id: str, run_id: str) -> dict:
    run = get_run(run_id)
    if run is None or run["experiment_id"] != experiment_id:
        raise HTTPException(status_code=404, detail="experiment run not found")
    return run
