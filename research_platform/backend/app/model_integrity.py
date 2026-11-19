"""Integrity checks for persisted model and checkpoint artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


def verify_artifact_file(root: Path, relative_path: str, expected_sha256: str | None, expected_size: int | None) -> dict[str, object]:
    path = (root / relative_path).resolve()
    root = root.resolve()
    if root not in path.parents or not path.is_file():
        return {"valid": False, "reason": "artifact file is missing or outside root"}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    checks = {"sha256": expected_sha256 is None or digest == expected_sha256, "size_bytes": expected_size is None or size == expected_size}
    return {"valid": all(checks.values()), "checks": checks, "actual_sha256": digest, "actual_size_bytes": size}
