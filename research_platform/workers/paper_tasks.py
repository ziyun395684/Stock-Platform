"""Worker task for paper holdings and NAV artifacts; never places orders."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.paper_portfolio")
def paper_portfolio(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    _set_run_status(run_id, "running")
    try:
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.paper_trading import apply_paper_snapshot

        raw = dict(config or {})
        state = apply_paper_snapshot(raw.get("previous"), list(raw.get("signals", [])), str(raw["snapshot_date"]), int(raw.get("top_k", 2)), float(raw.get("cost_rate", 0.003)))
        metadata = {"task": "paper_portfolio", "manual_approval_only": True, "automatic_ordering": False, "state": state}
        artifacts = []
        for name, kind, content in (("paper_holdings.json", "paper_holdings", json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")), ("paper_nav.json", "paper_nav", json.dumps({"date": state["date"], "capital": state["capital"], "net_return": state["net_return"], "turnover": state["turnover"]}, ensure_ascii=False, indent=2).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
