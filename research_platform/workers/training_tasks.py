"""Celery adapter for deterministic Factor-common strategy evaluation."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import _factor_config, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_common_train")
def factor_common_train(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing Factor-common evaluator and persist auditable metadata.

    This first platform adapter is intentionally deterministic: Factor-common is
    a rule-based research candidate, so no neural checkpoint is fabricated.
    """
    _set_run_status(run_id, "running")
    try:
        from research.factor_baseline.run_factor_baselines import run
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.engine_contract import validate_backtest_result
        from research_platform.backend.app.training_metadata import config_digest, validate_training_metadata

        raw = dict(config or {})
        normalized = _factor_config(raw)
        windows = [int(value) for value in raw.get("training_windows", [500, 750, 1000])]
        seeds = [int(value) for value in raw.get("seeds", [7, 42, 123])]
        if any(window not in {500, 750, 1000} for window in windows) or not windows:
            raise ValueError("training_windows 只能包含 500、750、1000")
        if any(seed not in {7, 42, 123} for seed in seeds) or not seeds:
            raise ValueError("seeds 只能包含 7、42、123")
        result = run(**normalized)
        validate_backtest_result(result)
        artifacts = []
        csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
        metadata = {
            "task": "factor_common_train",
            "training_mode": "deterministic_factor_common",
            "model_type": "factor-common-v1",
            "checkpoint": None,
            "model_metadata": {"model_type": "factor-common-v1", "windows": windows, "seeds": seeds},
            "rows": int(len(result)),
            "training_log": [{"stage": "data_loaded", "status": "succeeded"}, {"stage": "evaluation", "status": "succeeded", "rows": int(len(result))}],
            "config": {**normalized, "top_ks": list(normalized["top_ks"]), "cost_rates": list(normalized["cost_rates"]), "training_windows": windows, "seeds": seeds},
            "data_version": f"{normalized['data_dir']}:{normalized['start']}:{normalized['end']}",
            "config_digest": config_digest({**normalized, "top_ks": list(normalized["top_ks"]), "cost_rates": list(normalized["cost_rates"]), "training_windows": windows, "seeds": seeds}),
            "artifact_manifest": [],
            "best_total_return_net": float(result["total_return_net"].max()) if len(result) else None,
            "selection_policy": "research artifact only; no automatic strategy promotion",
        }
        for name, kind, content in (
            ("factor_common_evaluation.csv", "factor_common_evaluation", csv_bytes),
            ("factor_common_training_log.json", "factor_common_training_log", json.dumps(metadata["training_log"], ensure_ascii=False, indent=2).encode("utf-8")),
            ("factor_common_model_metadata.json", "factor_common_model_metadata", json.dumps(metadata["model_metadata"], ensure_ascii=False, indent=2).encode("utf-8")),
        ):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        metadata["artifact_manifest"] = artifacts.copy()
        validate_training_metadata(metadata)
        path, digest, size = _write_artifact(run_id, "factor_common_training_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
        artifact_id = record_artifact(run_id, "factor_common_training_metadata", path, digest, size)
        artifacts.append({"id": artifact_id, "artifact_type": "factor_common_training_metadata", "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
