"""SQLAlchemy metadata models for the first platform migration.

The models describe platform metadata only. They do not own the existing
StockRNN model files, reports, or frozen strategy source files.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_strategy_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("strategies.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    versions: Mapped[list["StrategyVersion"]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.id"), index=True)
    parent_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("strategy_versions.id"), nullable=True, index=True)
    version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    data_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    factor_version_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    strategy: Mapped[Strategy] = relationship(back_populates="versions")


class Factor(Base):
    __tablename__ = "factors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    factor_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    economic_hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    versions: Mapped[list["FactorVersion"]] = relationship(
        back_populates="factor", cascade="all, delete-orphan"
    )


class FactorVersion(Base):
    __tablename__ = "factor_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    factor_id: Mapped[str] = mapped_column(String(36), ForeignKey("factors.id"), index=True)
    version: Mapped[str] = mapped_column(String(64))
    formula: Mapped[str] = mapped_column(Text)
    input_columns: Mapped[list[str]] = mapped_column(JSON)
    parameters_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    time_boundary: Mapped[str | None] = mapped_column(String(255), nullable=True)
    code_artifact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    factor: Mapped[Factor] = relationship(back_populates="versions")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    strategy_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    objective: Mapped[str] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), index=True)
    run_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    artifact_type: Mapped[str] = mapped_column(String(64))
    relative_path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experiment_run_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("experiment_runs.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiment_runs.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiment_runs.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    split: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    objective: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("agent_sessions.id"), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="planned", index=True)
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(64), index=True)
    target_id: Mapped[str] = mapped_column(String(36), index=True)
    decision: Mapped[str] = mapped_column(String(32))
    reviewer: Mapped[str] = mapped_column(String(128))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_artifact_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(64), index=True)
    target_id: Mapped[str] = mapped_column(String(36), index=True)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiment_runs.id"), index=True)
    task_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Numeric(6, 3), default=0)
    current_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_artifact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
