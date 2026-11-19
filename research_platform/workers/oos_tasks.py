"""Celery adapter for rolling/non-overlapping out-of-sample Factor-common backtests."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_common_oos")
def factor_common_oos(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run existing non-overlapping OOS windows and persist audited outputs."""
    _set_run_status(run_id, "running")
    try:
        import pandas as pd
        from research.factor_baseline.run_double_oos import run
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.engine_contract import validate_backtest_result

        raw = dict(config or {})
        if raw.get("independent_oos") is True:
            if not raw.get("strategy_id") or not raw.get("strategy_version_id"):
                raise ValueError("independent OOS requires strategy and strategy version identifiers")
            if not raw.get("reference_end"):
                raise ValueError("independent OOS requires reference_end")
            if str(raw.get("start", "")) <= str(raw["reference_end"]):
                raise ValueError("independent OOS start must be after reference_end")
        normalized = {
            "config_path": str(_relative_input(raw.get("config_path", "config.yaml"), "config.yaml")),
            "data_dir": str(_relative_input(raw.get("data_dir", "quotes2"), "quotes2")),
            "symbols": tuple(str(value) for value in raw.get("symbols", [])),
            "benchmark_symbol": raw.get("benchmark_symbol"),
            "start": str(raw.get("start", "2024-01-01")),
            "end": str(raw.get("end", "2024-03-31")),
            "window_size": int(raw.get("window_size", 20)),
            "hold_days": int(raw.get("hold_days", 3)),
            "top_k": int(raw.get("top_k", 2)),
            "cost_rate": float(raw.get("cost_rate", 0.003)),
        }
        if normalized["window_size"] < 5 or normalized["window_size"] > 5000:
            raise ValueError("window_size 必须在 5 到 5000 之间")
        if normalized["hold_days"] < 1 or normalized["top_k"] < 1 or normalized["top_k"] > 10:
            raise ValueError("hold_days/top_k 配置无效")
        result = run(**normalized)
        validate_backtest_result(result)
        artifacts = []
        summary = {
            "task": "factor_common_oos",
            "independent_oos": bool(raw.get("independent_oos", False)),
            "strategy_id": raw.get("strategy_id"),
            "strategy_version_id": raw.get("strategy_version_id"),
            "reference_end": raw.get("reference_end"),
            "test_start": normalized["start"],
            "test_end": normalized["end"],
            "windows": int(len(result)),
            "config": {**normalized, "symbols": list(normalized["symbols"])},
            "mean_total_return_net": float(result["total_return_net"].mean()) if len(result) else None,
            "positive_windows": int((result["total_return_net"] > 0).sum()) if len(result) else 0,
            "mean_max_drawdown": float(result["max_drawdown"].mean()) if len(result) else None,
        }
        for name, kind, content in (
            ("factor_common_oos_windows.csv", "factor_common_oos_windows", result.to_csv(index=False).encode("utf-8-sig")),
            ("factor_common_oos_summary.json", "factor_common_oos_summary", json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8")),
        ):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **summary, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
