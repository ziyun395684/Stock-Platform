"""Persistence adapters for platform strategies and experiments."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .database import get_engine
from .models import Experiment, ExperimentRun, Strategy, StrategyVersion, TrainingJob


def _session_factory():
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def _version_dict(version: StrategyVersion) -> dict[str, Any]:
    return {
        "id": version.id,
        "strategy_id": version.strategy_id,
        "version": version.version,
        "parent_version_id": version.parent_version_id,
        "config": version.config_json or {},
        "status": version.status,
        "data_version": version.data_version,
        "factor_version_ids": version.factor_version_ids,
        "model_version": version.model_version,
    }


def _strategy_dict(strategy: Strategy) -> dict[str, Any]:
    return {
        "id": strategy.id,
        "strategy_id": strategy.strategy_id,
        "name": strategy.name,
        "status": strategy.status,
        "description": strategy.description,
        "parent_strategy_id": strategy.parent_strategy_id,
        "versions": [_version_dict(v) for v in strategy.versions],
    }


def list_strategies() -> list[dict[str, Any]]:
    with _session_factory()() as session:
        rows = session.scalars(
            select(Strategy).options(selectinload(Strategy.versions)).order_by(Strategy.created_at)
        ).all()
        return [_strategy_dict(row) for row in rows]


def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.scalar(
            select(Strategy)
            .where(Strategy.id == strategy_id)
            .options(selectinload(Strategy.versions))
        )
        return _strategy_dict(row) if row else None


def create_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    with _session_factory()() as session:
        row = Strategy(
            id=str(uuid4()),
            strategy_id=payload["strategy_id"],
            name=payload["name"],
            status=payload["status"],
            description=payload.get("description"),
            parent_strategy_id=payload.get("parent_strategy_id"),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _strategy_dict(row)


def clone_frozen_strategy(strategy_id: str, new_strategy_id: str, name: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        source = session.scalar(select(Strategy).where(Strategy.id == strategy_id).options(selectinload(Strategy.versions)))
        if source is None:
            return None
        if source.status != "frozen":
            raise ValueError("only frozen strategies can be cloned")
        clone = Strategy(id=str(uuid4()), strategy_id=new_strategy_id, name=name, status="research", parent_strategy_id=source.id, description=f"Clone of {source.strategy_id}")
        session.add(clone)
        session.commit()
        session.refresh(clone)
        return _strategy_dict(clone)


def update_strategy_status(strategy_id: str, status: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        row = session.scalar(select(Strategy).where(Strategy.id == strategy_id).options(selectinload(Strategy.versions)))
        if row is None:
            return None
        row.status = status
        session.commit()
        session.refresh(row)
        return _strategy_dict(row)


def create_strategy_version(strategy_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    with _session_factory()() as session:
        row = StrategyVersion(
            id=str(uuid4()),
            strategy_id=strategy_id,
            version=payload["version"],
            parent_version_id=payload.get("parent_version_id"),
            status=payload["status"],
            config_json=payload.get("config", {}),
            data_version=payload.get("data_version"),
            factor_version_ids=payload.get("factor_version_ids"),
            model_version=payload.get("model_version"),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _version_dict(row)


def list_experiments() -> list[dict[str, Any]]:
    with _session_factory()() as session:
        rows = session.scalars(select(Experiment).order_by(Experiment.created_at)).all()
        return [_experiment_dict(row) for row in rows]


def create_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    with _session_factory()() as session:
        row = Experiment(
            id=str(uuid4()),
            experiment_id=payload["experiment_id"],
            objective=payload["objective"],
            config_json=payload.get("config", {}),
            strategy_version_id=payload.get("strategy_version_id"),
            created_by=payload.get("created_by"),
            status="draft",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _experiment_dict(row)


def _experiment_dict(row: Experiment) -> dict[str, Any]:
    return {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "objective": row.objective,
        "config": row.config_json or {},
        "strategy_version_id": row.strategy_version_id,
        "created_by": row.created_by,
        "status": row.status,
    }
