"""Strategy and strategy-version API routes.

The first implementation uses a non-persistent store until MySQL is connected.
The same request/response contracts will be backed by SQLAlchemy repositories
in the next step.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..memory_store import store
from ..repositories import (
    create_strategy as db_create_strategy,
    create_strategy_version as db_create_strategy_version,
    clone_frozen_strategy as db_clone_frozen_strategy,
    update_strategy_status as db_update_strategy_status,
    get_strategy as db_get_strategy,
    list_strategies as db_list_strategies,
)
from ..schemas import (
    StrategyCreate,
    StrategyResponse,
    StrategyVersionCreate,
    StrategyVersionResponse,
    StrategyTransition,
    IndependentRetestRequest,
)
from ..independent_retest import build_retest_config
from ..repositories import create_experiment as db_create_experiment, list_experiments as db_list_experiments
from ..run_repository import attach_task, create_run, get_run_by_key, update_run
from research_platform.workers.oos_tasks import factor_common_oos

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("", response_model=list[StrategyResponse])
def list_strategies() -> list[dict]:
    return db_list_strategies() if settings.storage_backend == "mysql" else store.list_strategies()


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
def create_strategy(payload: StrategyCreate) -> dict:
    if any(item["strategy_id"] == payload.strategy_id for item in list_strategies()):
        raise HTTPException(status_code=409, detail="strategy_id already exists")
    if payload.status == "frozen":
        raise HTTPException(status_code=400, detail="new strategies cannot start as frozen")
    record = {"id": str(uuid4()), **payload.model_dump(), "versions": []}
    if settings.storage_backend == "mysql":
        record.pop("id")
        return db_create_strategy(record)
    return store.create_strategy(record)


@router.get("/{strategy_id}", response_model=StrategyResponse)
def get_strategy(strategy_id: str) -> dict:
    record = db_get_strategy(strategy_id) if settings.storage_backend == "mysql" else store.get_strategy(strategy_id)
    if record is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    return record


@router.post("/{strategy_id}/clone", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
def clone_strategy(strategy_id: str, payload: StrategyCreate) -> dict:
    source = db_get_strategy(strategy_id) if settings.storage_backend == "mysql" else store.get_strategy(strategy_id)
    if source is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    if source["status"] != "frozen":
        raise HTTPException(status_code=409, detail="only frozen strategies can be cloned")
    if settings.storage_backend == "mysql":
        try:
            return db_clone_frozen_strategy(strategy_id, payload.strategy_id, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    record = {"id": str(uuid4()), "strategy_id": payload.strategy_id, "name": payload.name, "status": "research", "description": payload.description or f"Clone of {source['strategy_id']}", "parent_strategy_id": source["id"], "versions": []}
    return store.create_strategy(record)


@router.post("/{strategy_id}/independent-oos", status_code=status.HTTP_202_ACCEPTED)
def independent_oos(strategy_id: str, payload: IndependentRetestRequest) -> dict:
    """Run a frozen strategy on a strictly later, independent date range."""
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=503, detail="independent retests require MySQL storage")
    strategy = db_get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    if strategy["status"] != "frozen":
        raise HTTPException(status_code=409, detail="only frozen strategies can be independently retested")
    version = strategy["versions"][-1] if strategy["versions"] else None
    if version is None:
        raise HTTPException(status_code=409, detail="frozen strategy has no version")
    try:
        config = build_retest_config(strategy, version, payload.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    experiment_id = payload.experiment_id or f"{strategy['strategy_id']}_independent_oos_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    run_key = payload.run_key or f"{experiment_id}:factor_common_oos"
    try:
        experiment = next((item for item in db_list_experiments() if item["experiment_id"] == experiment_id), None)
        if experiment is None:
            experiment = db_create_experiment({
                "experiment_id": experiment_id,
                "objective": f"Independent OOS retest for frozen strategy {strategy['strategy_id']}",
                "config": config,
                "strategy_version_id": version["id"],
                "created_by": "platform-independent-retest",
            })
        existing = get_run_by_key(experiment["id"], run_key)
        if existing is not None:
            return {"experiment": experiment, "run": existing, "idempotent": True}
        run = create_run(experiment["id"], run_key, config)
        task = factor_common_oos.apply_async(args=[run["id"], config])
        attach_task(run["id"], task.id)
        run["task_id"] = task.id
        return {"experiment": experiment, "run": run, "idempotent": False}
    except Exception as exc:
        if "Duplicate" in type(exc).__name__ or "duplicate" in str(exc).lower():
            raise HTTPException(status_code=409, detail="experiment_id or run_key already exists") from exc
        if "run" in locals():
            update_run(run["id"], "failed", error=exc)
        raise HTTPException(status_code=503, detail="unable to enqueue independent retest") from exc


@router.get("/{strategy_id}/approvals")
def strategy_approvals(strategy_id: str) -> list[dict]:
    from ..approval_repository import list_approvals_for_target
    return list_approvals_for_target(strategy_id) if settings.storage_backend == "mysql" else []


@router.post("/{strategy_id}/transition", response_model=StrategyResponse)
def transition_strategy(strategy_id: str, payload: StrategyTransition) -> dict:
    current = db_get_strategy(strategy_id) if settings.storage_backend == "mysql" else store.get_strategy(strategy_id)
    if current is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    allowed = {
        "draft": {"research", "retired"},
        "research": {"training", "backtested", "retired"},
        "training": {"backtested", "research", "retired"},
        "backtested": {"candidate", "research", "retired"},
        "candidate": {"approved", "research", "retired"},
        "approved": {"frozen", "retired"},
        "frozen": {"retired"},
        "retired": set(),
    }
    old_status = current["status"]
    if payload.status not in allowed.get(old_status, set()):
        raise HTTPException(status_code=409, detail=f"illegal strategy transition: {old_status} -> {payload.status}")
    if settings.storage_backend == "mysql" and payload.status in {"approved", "frozen"}:
        from ..approval_repository import has_decision
        required = "freeze_requested" if payload.status == "frozen" else "approved"
        if not has_decision(current["id"], required):
            raise HTTPException(status_code=409, detail=f"missing approval decision: {required}")
    if settings.storage_backend == "mysql":
        result = db_update_strategy_status(current["id"], payload.status)
        from ..approval_repository import create_audit_event
        create_audit_event("strategy_status_transition", "strategy", current["id"], None, {"from": old_status, "to": payload.status, "reason": payload.reason})
        return result
    return store.update_strategy_status(current["id"], payload.status)


@router.post("/{strategy_id}/versions", response_model=StrategyVersionResponse, status_code=status.HTTP_201_CREATED)
def create_strategy_version(strategy_id: str, payload: StrategyVersionCreate) -> dict:
    strategy = db_get_strategy(strategy_id) if settings.storage_backend == "mysql" else store.get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    if strategy["status"] == "frozen":
        raise HTTPException(status_code=409, detail="frozen strategy is read-only")
    if payload.status == "frozen":
        raise HTTPException(status_code=400, detail="freeze requires a separate approval step")
    if any(item["version"] == payload.version for item in strategy["versions"]):
        raise HTTPException(status_code=409, detail="strategy version already exists")
    version = {"id": str(uuid4()), "strategy_id": strategy["id"], **payload.model_dump()}
    if settings.storage_backend == "mysql":
        version.pop("id")
        return db_create_strategy_version(strategy["id"], version)
    strategy["versions"].append(version)
    return version
