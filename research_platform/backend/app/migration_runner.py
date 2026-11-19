"""Minimal ordered SQL migration runner for platform metadata."""
from __future__ import annotations

from pathlib import Path
from sqlalchemy import text

from .database import get_engine

MIGRATION_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def apply_migrations() -> list[str]:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS platform_schema_version (version VARCHAR(64) PRIMARY KEY, applied_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6))"))
        applied = {row[0] for row in connection.execute(text("SELECT version FROM platform_schema_version"))}
        applied_now: list[str] = []
        for path in sorted(MIGRATION_DIR.glob("*.sql")):
            version = path.stem
            if version in applied:
                continue
            sql = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("--"))
            statements = [part.strip() for part in sql.split(";") if part.strip()]
            for statement in statements:
                connection.execute(text(statement))
            connection.execute(text("INSERT INTO platform_schema_version(version) VALUES (:version)"), {"version": version})
            applied_now.append(version)
        return applied_now
