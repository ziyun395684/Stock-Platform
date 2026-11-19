"""Safe MySQL backup and restore-drill helpers.

The helpers only construct commands and validate local backup files. They do not
execute shell commands, so callers can apply their own approval and runtime
policy before running mysqldump or mysql.
"""
from __future__ import annotations

import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _database_parts(database_url: str) -> dict[str, Any]:
    parsed = urlparse(database_url.replace("mysql+pymysql://", "mysql://", 1))
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector", "mysql+pymysql"}:
        raise ValueError("backup only supports MySQL database URLs")
    if not parsed.hostname or not parsed.path.strip("/"):
        raise ValueError("database URL must include host and database")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": parsed.username or "",
        "database": parsed.path.strip("/"),
    }


def build_mysqldump_command(database_url: str, output_path: str | Path) -> list[str]:
    """Return a password-free argv list for mysqldump."""
    parts = _database_parts(database_url)
    destination = Path(output_path)
    if destination.suffix.lower() not in {".sql", ".gz"}:
        raise ValueError("backup output must be a .sql or .gz file")
    command = ["mysqldump", "--single-transaction", "--routines", "--triggers", "--host", parts["host"], "--port", str(parts["port"]), "--user", parts["user"], parts["database"]]
    return command + [">", str(destination)]


def build_restore_command(database_url: str, backup_path: str | Path) -> list[str]:
    """Return a password-free argv list for a restore drill."""
    parts = _database_parts(database_url)
    source = Path(backup_path)
    if source.suffix.lower() not in {".sql", ".gz"}:
        raise ValueError("restore input must be a .sql or .gz file")
    return ["mysql", "--host", parts["host"], "--port", str(parts["port"]), "--user", parts["user"], parts["database"], "<", str(source)]


def create_backup_manifest(database_url: str, backup_path: str | Path, *, created_at: datetime | None = None) -> dict[str, Any]:
    path = Path(backup_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parts = _database_parts(database_url)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"database": parts["database"], "host": parts["host"], "created_at": (created_at or datetime.now(timezone.utc)).isoformat(), "backup_path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def write_backup_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_backup_manifest(manifest: dict[str, Any], backup_path: str | Path) -> bool:
    path = Path(backup_path).resolve()
    if not path.is_file() or int(manifest.get("size_bytes", -1)) != path.stat().st_size:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == manifest.get("sha256")

