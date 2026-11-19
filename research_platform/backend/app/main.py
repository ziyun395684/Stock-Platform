"""Minimal FastAPI entrypoint for the StockRNN research platform."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.experiments import router as experiments_router
from .api.strategies import router as strategies_router
from .api.runs import router as runs_router
from .api.jobs import router as jobs_router
from .api.artifacts import router as artifacts_router
from .api.reports import router as reports_router
from .api.factors import router as factors_router
from .api.models import router as models_router
from .api.approvals import router as approvals_router
from .api.agents import router as agents_router
from .api.agent_audit import router as agent_audit_router
from .api.operations import router as operations_router
from .api.reports_phase10 import router as phase10_router
from .config import settings
from .dependencies import dependency_status


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_metadata_tables()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Research orchestration API for StockRNN. "
        "It does not place orders or mutate frozen strategies."
    ),
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail), "path": request.url.path}})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": "request validation failed", "details": exc.errors(), "path": request.url.path}})


app.include_router(strategies_router, prefix=settings.api_prefix)
app.include_router(experiments_router, prefix=settings.api_prefix)
app.include_router(runs_router, prefix=settings.api_prefix)
app.include_router(jobs_router, prefix=settings.api_prefix)
app.include_router(artifacts_router, prefix=settings.api_prefix)
app.include_router(reports_router, prefix=settings.api_prefix)
app.include_router(factors_router, prefix=settings.api_prefix)
app.include_router(models_router, prefix=settings.api_prefix)
app.include_router(approvals_router, prefix=settings.api_prefix)
app.include_router(agents_router, prefix=settings.api_prefix)
app.include_router(agent_audit_router, prefix=settings.api_prefix)
app.include_router(operations_router, prefix=settings.api_prefix)
app.include_router(phase10_router, prefix=settings.api_prefix)


def ensure_metadata_tables() -> None:
    if settings.storage_backend == "mysql":
        from .database import get_engine
        from .models import Base
        engine = get_engine()
        Base.metadata.create_all(engine, checkfirst=True)
        # Lightweight forward-compatible migration for metadata columns added after MVP.
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        columns = {column["name"] for column in inspector.get_columns("audit_events")}
        additions = {"target_type": "VARCHAR(64) NOT NULL DEFAULT 'unknown'", "target_id": "VARCHAR(36) NOT NULL DEFAULT ''", "actor": "VARCHAR(128) NULL", "payload_json": "JSON NULL"}
        session_columns = {column["name"] for column in inspector.get_columns("agent_sessions")}
        approval_columns = {column["name"] for column in inspector.get_columns("approvals")}
        strategy_version_columns = {column["name"] for column in inspector.get_columns("strategy_versions")}
        with engine.begin() as connection:
            if "plan_json" not in session_columns:
                connection.execute(text("ALTER TABLE agent_sessions ADD COLUMN plan_json JSON NULL"))
            if "report_artifact_ids" not in approval_columns:
                connection.execute(text("ALTER TABLE approvals ADD COLUMN report_artifact_ids JSON NULL"))
            if "parent_version_id" not in strategy_version_columns:
                connection.execute(text("ALTER TABLE strategy_versions ADD COLUMN parent_version_id CHAR(36) NULL"))
                connection.execute(text("ALTER TABLE strategy_versions ADD KEY idx_strategy_versions_parent (parent_version_id)"))
        
        with engine.begin() as connection:
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE audit_events ADD COLUMN {name} {definition}"))
            connection.execute(text("CREATE TABLE IF NOT EXISTS platform_schema_version (version VARCHAR(64) NOT NULL PRIMARY KEY, applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6))"))
            connection.execute(text("INSERT INTO platform_schema_version(version) VALUES ('003_strategy_lineage') ON DUPLICATE KEY UPDATE version=VALUES(version)"))


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    """Process health endpoint that does not require external services."""
    return {
        "status": "ok",
        "service": "research-platform-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(f"{settings.api_prefix}/health", tags=["system"])
def api_health() -> dict[str, object]:
    """Versioned API health endpoint."""
    return health()


@app.get(f"{settings.api_prefix}/dependencies", tags=["system"])
def dependencies() -> dict[str, object]:
    """Check MySQL and Redis reachability without exposing credentials."""
    return dependency_status()


@app.get(f"{settings.api_prefix}/runtime-config", tags=["system"])
def runtime_config() -> dict[str, object]:
    """Return non-secret runtime configuration for the frontend diagnostics page."""
    return settings.public_dict()
