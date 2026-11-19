"""Persistence for immutable research artifacts."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from .models import Artifact
from .repositories import _session_factory


def validate_artifact_record(relative_path: str, sha256: str, size_bytes: int) -> None:
    """Reject unsafe or internally inconsistent artifact index records."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not relative_path.strip():
        raise ValueError("artifact path must be a non-empty repository-relative path")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256.lower()):
        raise ValueError("artifact sha256 must be a 64-character hexadecimal digest")
    if size_bytes < 0:
        raise ValueError("artifact size_bytes must be non-negative")


def record_artifact(
    run_id: str,
    artifact_type: str,
    relative_path: str,
    sha256: str,
    size_bytes: int,
) -> str:
    validate_artifact_record(relative_path, sha256, size_bytes)
    artifact_id = str(uuid4())
    with _session_factory()() as session:
        session.add(Artifact(
            id=artifact_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            sha256=sha256,
            size_bytes=size_bytes,
            experiment_run_id=run_id,
        ))
        session.commit()
    return artifact_id


def get_artifact(artifact_id: str) -> Artifact | None:
    with _session_factory()() as session:
        return session.scalar(select(Artifact).where(Artifact.id == artifact_id))
