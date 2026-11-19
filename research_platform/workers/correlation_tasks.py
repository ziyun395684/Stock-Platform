"""Celery adapter for cross-factor correlation analysis."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_correlation")
def factor_correlation(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute a Spearman correlation matrix over existing common factors."""
    _set_run_status(run_id, "running")
    try:
        import pandas as pd
        import yaml
        from data.panel import align_etf_frames, load_universe_frames
        from model_and_train.factor_library import add_common_factors
        from research_platform.backend.app.artifact_repository import record_artifact

        raw = dict(config or {})
        config_path = _relative_input(raw.get("config_path", "config.yaml"), "config.yaml")
        data_dir = _relative_input(raw.get("data_dir", "quotes2"), "quotes2")
        with (ROOT / config_path).open(encoding="utf-8") as handle:
            project_config = yaml.safe_load(handle)
        symbols = list(project_config["universe"]["symbols"])
        benchmark_symbol = str(raw.get("benchmark", project_config["universe"].get("benchmark", "510300.SH")))
        start = str(raw.get("start", "2024-01-01"))
        end = str(raw.get("end", "2024-03-31"))
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark = load_universe_frames([benchmark_symbol], data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)[benchmark_symbol]
        panel = align_etf_frames(frames, date_mode="union")
        panel = add_common_factors(panel, benchmark_frame=benchmark)
        factor_columns = raw.get("factor_columns") or ["roc_5", "roc_10", "roc_20", "roc_60", "trend_score_24", "trend_score_25", "ma_energy_38", "ma_energy_39", "ma_energy_41", "volume_ratio_5_20"]
        available = [column for column in factor_columns if column in panel.columns]
        if len(available) < 2:
            raise ValueError("可用于相关性分析的因子列少于 2 个")
        matrix = panel[available].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        rows = matrix.reset_index().rename(columns={"index": "factor"})
        csv_bytes = rows.to_csv(index=False).encode("utf-8-sig")
        metadata = {"task": "factor_correlation", "factor_columns": available, "rows": int(len(panel)), "start": start, "end": end, "data_dir": str(data_dir), "method": "spearman"}
        artifacts = []
        for name, kind, content in (("factor_correlation_matrix.csv", "factor_correlation_matrix", csv_bytes), ("factor_correlation_summary.json", "factor_correlation_summary", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
