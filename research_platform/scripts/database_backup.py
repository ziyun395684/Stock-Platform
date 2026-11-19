"""Run an explicitly approved MySQL backup or restore drill.

Dry-run is the default. Credentials are read from MYSQL_PASSWORD and never
passed on the command line. The script intentionally operates outside Celery.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from research_platform.backend.app.config import settings
from research_platform.backend.app.database_backup import (
    _database_parts,
    create_backup_manifest,
    validate_backup_manifest,
    write_backup_manifest,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StockRNN MySQL backup/restore drill")
    parser.add_argument("mode", choices=("backup", "restore"))
    parser.add_argument("path", type=Path, help="SQL backup file")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="execute the external MySQL command")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")
    if args.path.suffix.lower() != ".sql":
        raise SystemExit("only .sql drill files are supported")
    parts = _database_parts(args.database_url)
    env = os.environ.copy()
    if "MYSQL_PASSWORD" in env:
        env["MYSQL_PWD"] = env["MYSQL_PASSWORD"]
    if args.mode == "backup":
        command = ["mysqldump", "--single-transaction", "--routines", "--triggers", "--host", parts["host"], "--port", str(parts["port"]), "--user", parts["user"], parts["database"]]
        if not args.execute:
            print("DRY RUN:", " ".join(command), ">", args.path)
            return 0
        args.path.parent.mkdir(parents=True, exist_ok=True)
        with args.path.open("wb") as output:
            subprocess.run(command, check=True, env=env, stdout=output)
        manifest = create_backup_manifest(args.database_url, args.path)
        manifest_path = write_backup_manifest(manifest, args.manifest or args.path.with_suffix(".manifest.json"))
        print(f"backup written: {args.path}")
        print(f"manifest written: {manifest_path}")
        return 0
    manifest_path = args.manifest or args.path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not validate_backup_manifest(manifest, args.path):
        raise SystemExit("backup manifest validation failed")
    command = ["mysql", "--host", parts["host"], "--port", str(parts["port"]), "--user", parts["user"], parts["database"]]
    if not args.execute:
        print("DRY RUN:", " ".join(command), "<", args.path)
        return 0
    with args.path.open("rb") as source:
        subprocess.run(command, check=True, env=env, stdin=source)
    print("restore completed and manifest was validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

