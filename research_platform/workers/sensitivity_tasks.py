"""Celery adapter for Factor-common Top-K and cost sensitivity."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import _factor_config, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_sensitivity")
def factor_sensitivity(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing factor baseline grid for Top-K/cost sensitivity."""
    _set_run_status(run_id, "running")
    try:
        from research.factor_baseline.run_factor_baselines import run
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.engine_contract import validate_backtest_result

        raw = dict(config or {})
        raw.setdefault("top_ks", [1, 2, 3])
        raw.setdefault("cost_rates", [0.001, 0.003, 0.005, 0.01])
        normalized = _factor_config(raw)
        result = run(**normalized)
        validate_backtest_result(result)
        summary = {
            "task": "factor_sensitivity",
            "rows": int(len(result)),
            "top_ks": list(normalized["top_ks"]),
            "cost_rates": list(normalized["cost_rates"]),
            "hold_days": normalized["hold_days"],
            "best_total_return_net": float(result["total_return_net"].max()) if len(result) else None,
            "worst_total_return_net": float(result["total_return_net"].min()) if len(result) else None,
            "config": {**normalized, "top_ks": list(normalized["top_ks"]), "cost_rates": list(normalized["cost_rates"])},
        }
        artifacts = []
        for name, kind, content in (("factor_sensitivity_grid.csv", "factor_sensitivity_grid", result.to_csv(index=False).encode("utf-8-sig")), ("factor_sensitivity_summary.json", "factor_sensitivity_summary", json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **summary, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
