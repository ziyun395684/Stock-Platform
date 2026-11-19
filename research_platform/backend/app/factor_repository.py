"""Read-only factor metadata queries."""
from __future__ import annotations

from sqlalchemy import select

from .models import Factor
from .repositories import _session_factory


def list_factor_metadata(limit: int = 100) -> list[dict]:
    with _session_factory()() as session:
        rows = session.scalars(
            select(Factor).order_by(Factor.updated_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": factor.id,
                "factor_id": factor.factor_id,
                "name": factor.name,
                "status": factor.status,
                "economic_hypothesis": factor.economic_hypothesis,
                "versions": [
                    {
                        "id": version.id,
                        "version": version.version,
                        "formula": version.formula,
                        "input_columns": version.input_columns or [],
                        "parameters": version.parameters_json or {},
                        "time_boundary": version.time_boundary,
                        "code_artifact_id": version.code_artifact_id,
                        "created_at": version.created_at.isoformat() if version.created_at else None,
                    }
                    for version in sorted(factor.versions, key=lambda item: item.created_at, reverse=True)
                ],
            }
            for factor in rows
        ]
