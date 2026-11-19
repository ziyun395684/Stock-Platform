"""Pydantic request/response contracts for platform APIs."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StrategyStatus = Literal[
    "draft", "research", "training", "backtested", "candidate", "approved", "frozen", "retired"
]


class StrategyCreate(BaseModel):
    strategy_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    status: StrategyStatus = "draft"
    description: str | None = None
    parent_strategy_id: str | None = None


class ResearchGoal(BaseModel):
    objective: str = Field(min_length=1)
    baseline_strategy_id: str | None = None
    constraints: list[str] = Field(default_factory=list)
    budget: int = Field(default=10, ge=1, le=1000)


class ResearchPlan(BaseModel):
    goal: ResearchGoal
    tasks: list[str] = Field(default_factory=list)
    estimated_runs: int = Field(default=0, ge=0, le=10000)
    requires_approval: bool = True


class FactorVersionDefinition(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    formula: str = Field(min_length=1)
    input_columns: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    time_boundary: str = Field(min_length=1)


class FactorDefinition(BaseModel):
    factor_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    economic_hypothesis: str = Field(min_length=1)
    version: FactorVersionDefinition


class ExperimentDesignRequest(BaseModel):
    tasks: list[str] = Field(min_length=1)
    training_windows: list[int] = Field(default_factory=lambda: [500, 750, 1000])
    seeds: list[int] = Field(default_factory=lambda: [7, 42, 123])
    cost_rates: list[float] = Field(default_factory=lambda: [0.003, 0.005])
    max_experiments: int = Field(default=12, ge=1, le=100)
    budget: int = Field(default=12, ge=1, le=1000)


class ExperimentDesignResponse(BaseModel):
    valid: bool
    experiments: list[dict[str, Any]]
    estimated_runs: int
    budget: int
    reason: str | None = None
    rationale: list[str] = Field(default_factory=list)


class ResearchReviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ResearchReviewFinding(BaseModel):
    check: str
    status: Literal["pass", "warn", "fail"]
    evidence: str
    recommendation: str | None = None


class ResearchReviewResponse(BaseModel):
    passed: bool
    findings: list[ResearchReviewFinding]
    summary: str


class AgentSessionCreate(BaseModel):
    objective: str = Field(min_length=1)
    created_by: str | None = None


class AgentSessionResponse(AgentSessionCreate):
    id: str
    status: str


class AgentTaskCreate(BaseModel):
    task_type: str = Field(min_length=1, max_length=64)
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentTaskResponse(AgentTaskCreate):
    id: str
    session_id: str
    status: str


class ApprovalCreate(BaseModel):
    target_type: Literal["strategy", "strategy_version", "experiment"]
    target_id: str
    decision: Literal["approved", "rejected", "freeze_requested"]
    reviewer: str = Field(min_length=1, max_length=128)
    comment: str | None = None
    report_artifact_ids: list[str] = Field(default_factory=list)


class ApprovalResponse(ApprovalCreate):
    id: str


class StrategyTransition(BaseModel):
    status: Literal["draft", "research", "training", "backtested", "candidate", "approved", "frozen", "retired"]
    reason: str | None = None


class StrategyVersionCreate(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)
    parent_version_id: str | None = None
    status: StrategyStatus = "draft"
    data_version: str | None = None
    factor_version_ids: list[str] | None = None
    model_version: str | None = None


class StrategyVersionResponse(StrategyVersionCreate):
    id: str
    strategy_id: str


class IndependentRetestRequest(BaseModel):
    start: str = Field(min_length=10, max_length=10)
    end: str = Field(min_length=10, max_length=10)
    experiment_id: str | None = Field(default=None, max_length=128)
    run_key: str | None = Field(default=None, max_length=255)
    reference_end: str | None = Field(default=None, max_length=10)
    data_dir: str = "quotes2"
    config_path: str = "config.yaml"
    symbols: list[str] = Field(default_factory=list)
    benchmark_symbol: str | None = None
    window_size: int = Field(default=20, ge=5, le=5000)
    hold_days: int = Field(default=3, ge=1, le=252)
    top_k: int = Field(default=2, ge=1, le=10)
    cost_rate: float = Field(default=0.003, ge=0, le=0.1)


class StrategyResponse(StrategyCreate):
    id: str
    versions: list[StrategyVersionResponse] = Field(default_factory=list)


class ExperimentCreate(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    strategy_version_id: str | None = None
    created_by: str | None = None


class ExperimentResponse(ExperimentCreate):
    id: str
    status: str


class ExperimentRunCreate(BaseModel):
    run_key: str = Field(min_length=1, max_length=255)
    task_type: Literal["engine_probe", "factor_compute", "factor_baseline", "factor_quality", "factor_stratified_returns", "factor_correlation", "factor_decay", "factor_regime", "factor_common_train", "transformer_train", "factor_common_oos", "factor_sensitivity", "strategy_gate", "signal_snapshot", "paper_portfolio"] = "engine_probe"
    config: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    id: str
    experiment_run_id: str
    task_id: str | None = None
    status: str
    progress: float
    current_epoch: int | None = None
    total_epochs: int | None = None
    experiment_id: str
    run_key: str
    run_status: str
    started_at: str | None = None
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class ArtifactResponse(BaseModel):
    id: str
    artifact_type: str
    relative_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: str | None = None


class DataFreshnessRequest(BaseModel):
    latest_date: str | None = None
    expected_date: str
    max_lag_days: int = Field(default=1, ge=0, le=30)


class ExperimentRunResponse(BaseModel):
    id: str
    experiment_id: str
    run_key: str
    status: str
    task_id: str | None = None
    progress: float = 0
    seed: int | None = None
    window_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
