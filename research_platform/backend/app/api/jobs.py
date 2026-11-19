"""Task/job status API."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import math
from io import StringIO
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..job_repository import get_job, list_jobs
from ..run_repository import attach_task, cancel_run, create_run
from ..schemas import ExperimentRunResponse
from research_platform.workers.celery_app import celery_app
from ..schemas import JobResponse
from ..task_registry import TASKS

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _report_kind(artifact_type: str, relative_path: str) -> str:
    value = f"{artifact_type} {relative_path}".lower()
    if any(token in value for token in ("log", "training")):
        return "training_log"
    if any(token in value for token in ("cost", "sensitivity")):
        return "cost_sensitivity"
    if any(token in value for token in ("rolling", "window", "oos")):
        return "rolling_window"
    if any(token in value for token in ("drawdown", "回撤")):
        return "drawdown"
    if any(token in value for token in ("nav", "equity", "净值")):
        return "nav"
    return "report"


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _csv_payload(text: str) -> tuple[list[str], list[list[str]], list[dict]]:
    reader = csv.DictReader(StringIO(text))
    columns = reader.fieldnames or []
    rows = list(reader)
    preview_rows = [[row.get(column, "") for column in columns] for row in rows[:200]]
    points = []
    x_column = columns[0] if columns else "x"
    for column in columns[1:]:
        values = []
        for row in rows[:500]:
            y = _number(row.get(column, ""))
            if y is not None:
                values.append({"x": row.get(x_column, str(len(values))), "y": y})
        if len(values) >= 2:
            points.append({"name": column, "points": values})
    return columns, preview_rows, points


@router.get("", response_model=list[JobResponse])
def jobs(
    limit: int = Query(default=50, ge=1, le=200),
    status: str | None = Query(default=None),
) -> list[dict]:
    if settings.storage_backend != "mysql":
        return []
    return list_jobs(limit=limit, status=status)


@router.get("/{job_id}", response_model=JobResponse)
def job(job_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="job not found")
    result = get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="job not found")
    return result


@router.get("/{job_id}/report")
def report(job_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="job not found")
    current = get_job(job_id)
    if current is None:
        raise HTTPException(status_code=404, detail="job not found")
    root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    summaries = []
    for artifact in current.get("artifacts", []):
        path = (root / artifact["relative_path"]).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            kind = _report_kind(artifact["artifact_type"], artifact["relative_path"])
            if path.suffix.lower() == ".json":
                payload = json.loads(text)
                metrics = {}
                def collect(values: dict, prefix: str = "") -> None:
                    for key, value in values.items():
                        name = f"{prefix}.{key}" if prefix else str(key)
                        if isinstance(value, dict):
                            collect(value, name)
                        elif isinstance(value, (int, float)) and key not in {"seed"}:
                            metrics[name] = value
                if isinstance(payload, dict):
                    collect(payload)
                summaries.append({"artifact_type": artifact["artifact_type"], "relative_path": artifact["relative_path"], "kind": kind, "metrics": metrics, "preview": "", "columns": [], "rows": [], "series": []})
            elif path.suffix.lower() == ".csv":
                columns, rows, series = _csv_payload(text)
                summaries.append({"artifact_type": artifact["artifact_type"], "relative_path": artifact["relative_path"], "kind": kind, "metrics": {}, "preview": "", "columns": columns, "rows": rows, "series": series})
            else:
                summaries.append({"artifact_type": artifact["artifact_type"], "relative_path": artifact["relative_path"], "kind": kind, "metrics": {}, "preview": "\\n".join(text.splitlines()[:40]), "columns": [], "rows": [], "series": []})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {"job": current, "summaries": summaries}


@router.post("/{job_id}/retry", response_model=ExperimentRunResponse, status_code=202)
def retry(job_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="job not found")
    current = get_job(job_id)
    if current is None:
        raise HTTPException(status_code=404, detail="job not found")
    if current["status"] != "failed":
        raise HTTPException(status_code=409, detail="only failed jobs can be retried")
    key = (current.get("run_key") or "").lower()
    task_fn = next((fn for name, fn in TASKS.items() if name in key and name != "engine_probe"), None)
    if task_fn is None:
        raise HTTPException(status_code=422, detail="cannot infer task type from run_key")
    run_key = f"{current['run_key']}_retry_of_{current['experiment_run_id']}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    try:
        run = create_run(current["experiment_id"], run_key, {"source": "platform-retry", "retry_of": current["experiment_run_id"]})
        task = task_fn.apply_async(args=[run["id"], {"source": "platform-retry", "retry_of": current["experiment_run_id"]}])
        attach_task(run["id"], task.id)
        run["task_id"] = task.id
        return run
    except Exception as exc:
        raise HTTPException(status_code=503, detail="unable to enqueue retry") from exc


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel(job_id: str) -> dict:
    if settings.storage_backend != "mysql":
        raise HTTPException(status_code=404, detail="job not found")
    current = get_job(job_id)
    if current is None:
        raise HTTPException(status_code=404, detail="job not found")
    if current["status"] in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="job is already terminal")
    if current.get("task_id"):
        celery_app.control.revoke(current["task_id"], terminate=False)
    result = cancel_run(current["experiment_run_id"])
    if result is None:
        raise HTTPException(status_code=404, detail="experiment run not found")
    return {**current, "status": "cancelled", "run_status": "cancelled", "finished_at": result.get("finished_at"), "artifacts": result.get("artifacts", [])}
