"""Repository-local secret hygiene checks for release gates."""
from __future__ import annotations

from pathlib import Path


FORBIDDEN_NAMES = {".env", ".env.local", ".env.docker", ".env.production", "id_rsa", "id_ed25519"}
SECRET_MARKERS = ("BEGIN PRIVATE KEY", "AWS_SECRET_ACCESS_KEY=", "DATABASE_PASSWORD=", "REDIS_PASSWORD=")


def scan_secret_hygiene(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in FORBIDDEN_NAMES:
            findings.append(f"secret-like file present: {path.relative_to(root)}")
            continue
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in SECRET_MARKERS:
            if marker in text:
                findings.append(f"secret marker {marker} in {path.relative_to(root)}")
    return findings
