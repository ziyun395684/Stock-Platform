"""Safe artifact metadata and download endpoints."""
from __future__ import annotations

from pathlib import Path
import csv
import io
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..artifact_repository import get_artifact
from ..config import settings

router = APIRouter(prefix="/artifacts", tags=["artifacts"])
ROOT = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[3]


@router.get("/{artifact_id}")
def artifact_metadata(artifact_id: str) -> dict:
    artifact = get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"id": artifact.id, "artifact_type": artifact.artifact_type, "relative_path": artifact.relative_path, "sha256": artifact.sha256, "size_bytes": artifact.size_bytes, "created_at": artifact.created_at.isoformat() if artifact.created_at else None}


@router.get("/{artifact_id}/preview")
def preview_artifact(artifact_id: str, lines: int = 20) -> dict:
    artifact = get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = (ROOT / artifact.relative_path).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file not found")
    if path.stat().st_size > 2_000_000:
        raise HTTPException(status_code=413, detail="artifact preview is limited to 2 MB")
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="artifact is not a UTF-8 text file") from exc
    limited = content.splitlines()[:min(max(lines, 1), 50)]
    columns: list[str] = []
    rows: list[list[str]] = []
    if path.suffix.lower() == ".csv" and limited:
        parsed = list(csv.reader(io.StringIO("\n".join(limited))))
        columns, rows = (parsed[0], parsed[1:]) if parsed else ([], [])
    elif path.suffix.lower() == ".json":
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                columns = ["key", "value"]
                rows = [[str(key), str(value)] for key, value in list(payload.items())[:50]]
        except json.JSONDecodeError:
            pass
    return {"id": artifact.id, "artifact_type": artifact.artifact_type, "relative_path": artifact.relative_path, "content": "\n".join(limited), "columns": columns, "rows": rows}


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str) -> FileResponse:
    artifact = get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = (ROOT / artifact.relative_path).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file not found")
    return FileResponse(path, media_type="text/plain", filename=path.name, headers={"X-Artifact-SHA256": artifact.sha256 or ""})
