"""Celery adapters for the existing Factor Baseline runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .celery_app import celery_app
from .tasks import _set_run_status


ROOT = Path(__file__).resolve().parents[2]


def _relative_input(value: str, default: str) -> Path:
    candidate = Path(value or default)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("研究输入路径必须是仓库内相对路径")
    resolved = (ROOT / candidate).resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise ValueError("研究输入路径超出仓库边界")
    if not resolved.exists():
        raise FileNotFoundError(f"研究输入路径不存在: {candidate}")
    return candidate


def _factor_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    normalized = {
        "config_path": str(_relative_input(raw.get("config_path", "config.yaml"), "config.yaml")),
        "data_dir": str(_relative_input(raw.get("data_dir", "quotes2"), "quotes2")),
        "start": str(raw.get("start", "2020-01-01")),
        "end": str(raw.get("end", "2025-11-05")),
        "hold_days": int(raw.get("hold_days", 10)),
        "top_ks": tuple(int(x) for x in raw.get("top_ks", [1, 2])),
        "cost_rates": tuple(float(x) for x in raw.get("cost_rates", [0.003, 0.005])),
    }
    if normalized["hold_days"] < 1 or normalized["hold_days"] > 252:
        raise ValueError("hold_days 必须在 1 到 252 之间")
    if not normalized["top_ks"] or any(value < 1 or value > 10 for value in normalized["top_ks"]):
        raise ValueError("top_ks 必须是 1 到 10 的非空整数列表")
    if not normalized["cost_rates"] or any(value < 0 or value > 0.1 for value in normalized["cost_rates"]):
        raise ValueError("cost_rates 必须在 0 到 0.1 之间")
    if normalized["start"] >= normalized["end"]:
        raise ValueError("start 必须早于 end")
    return normalized


def _write_artifact(run_id: str, name: str, content: bytes) -> tuple[str, str, int]:
    output_dir = ROOT / "output" / "platform" / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return str(path.relative_to(ROOT)).replace("\\", "/"), digest, len(content)


@celery_app.task(bind=True, name="research_platform.factor_baseline")
def factor_baseline(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing Factor Baseline runner and register immutable artifacts."""
    _set_run_status(run_id, "running")
    try:
        from research.factor_baseline.run_factor_baselines import run
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.engine_contract import validate_backtest_result

        normalized = _factor_config(config)
        result = run(
            config_path=normalized["config_path"],
            data_dir=normalized["data_dir"],
            start=normalized["start"],
            end=normalized["end"],
            hold_days=normalized["hold_days"],
            top_ks=normalized["top_ks"],
            cost_rates=normalized["cost_rates"],
        )
        validate_backtest_result(result)
        csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
        summary = {
            "task": "factor_baseline",
            "rows": int(len(result)),
            "columns": list(result.columns),
            "config": {**normalized, "top_ks": list(normalized["top_ks"]), "cost_rates": list(normalized["cost_rates"])},
            "best_total_return_net": float(result["total_return_net"].max()) if len(result) else None,
        }
        json_bytes = json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        artifacts = []
        for name, kind, content in (
            ("factor_baseline_summary.csv", "factor_baseline_csv", csv_bytes),
            ("factor_baseline_summary.json", "factor_baseline_summary", json_bytes),
        ):
            relative_path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, relative_path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": relative_path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", "task": "factor_baseline", "rows": len(result), "artifacts": artifacts, "config": summary["config"]}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
