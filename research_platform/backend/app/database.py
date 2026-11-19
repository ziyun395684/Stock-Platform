"""Database helpers for the platform metadata store.

The platform uses the existing MySQL service. Connections are created lazily;
importing the API must not require MySQL to be reachable.
"""
from __future__ import annotations

from typing import Any

from .config import settings


_engine: Any | None = None


def get_engine() -> Any:
    """Create and cache a SQLAlchemy engine on first use."""
    global _engine
    if _engine is not None:
        return _engine
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    from sqlalchemy import create_engine

    _engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )
    return _engine


def check_database() -> dict[str, str | bool]:
    """Run a short SELECT 1 dependency check without exposing credentials."""
    if not settings.database_url:
        return {"configured": False, "reachable": False, "detail": "DATABASE_URL missing"}
    try:
        from sqlalchemy import text

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"configured": True, "reachable": True, "detail": "ok"}
    except Exception as exc:  # dependency diagnostics must not crash the API
        return {"configured": True, "reachable": False, "detail": type(exc).__name__}


def dispose_engine() -> None:
    """Dispose the cached engine during shutdown/tests."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
