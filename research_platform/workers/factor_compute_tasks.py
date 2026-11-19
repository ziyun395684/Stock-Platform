"""Unified adapter for computing the platform's common factor panel."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_compute")
def factor_compute(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    _set_run_status(run_id, "running")
    try:
        from data.panel import load_universe_frames
        from data.panel import align_etf_frames
        from model_and_train.factor_library import add_common_factors
        from research_platform.backend.app.artifact_repository import record_artifact

        raw = dict(config or {})
        data_dir = _relative_input(raw.get("data_dir", "quotes2"), "quotes2")
        config_path = _relative_input(raw.get("config_path", "config.yaml"), "config.yaml")
        import yaml
        project_config = yaml.safe_load((ROOT / config_path).open(encoding="utf-8"))
        symbols = list(raw.get("symbols") or project_config["universe"]["symbols"])
        start, end = str(raw.get("start", "2024-01-01")), str(raw.get("end", "2025-01-01"))
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        panel = add_common_factors(align_etf_frames(frames, date_mode="union"))
        columns = [str(column) for column in panel.columns if str(column).startswith(("roc_", "trend_score_", "ma_energy_", "volume_ratio_"))]
        csv = panel.to_csv(index=False).encode("utf-8-sig")
        metadata = {"task": "factor_compute", "rows": int(len(panel)), "symbols": symbols, "factor_columns": columns, "start": start, "end": end, "data_dir": str(data_dir)}
        artifacts = []
        for name, kind, content in (("factor_panel.csv", "factor_panel", csv), ("factor_compute_metadata.json", "factor_compute_metadata", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise

