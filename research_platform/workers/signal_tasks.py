"""Worker task for immutable daily signal snapshots; no order placement."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.signal_snapshot")
def signal_snapshot(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    _set_run_status(run_id, "running")
    try:
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.signal_snapshot import build_signal_snapshot

        raw = dict(config or {})
        snapshot_date = str(raw.get("snapshot_date", ""))
        signals, checklist = build_signal_snapshot(list(raw.get("signals", [])), snapshot_date, int(raw.get("top_k", 2)))
        metadata = {"task": "signal_snapshot", "snapshot_date": snapshot_date, "top_k": int(raw.get("top_k", 2)), "selected_count": len(checklist), "manual_review_required": True, "automatic_ordering": False}
        artifacts = []
        for name, kind, content in (("daily_signal_snapshot.csv", "daily_signal_snapshot", _csv(signals).encode("utf-8-sig")), ("manual_review_checklist.json", "manual_review_checklist", json.dumps({**metadata, "checklist": checklist}, ensure_ascii=False, indent=2).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise


def _csv(rows: list[dict[str, Any]]) -> str:
    import csv
    from io import StringIO
    output = StringIO()
    columns = ["snapshot_date", "symbol", "score", "rank", "selected", "action", "source"]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
