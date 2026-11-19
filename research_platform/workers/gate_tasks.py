"""Celery adapter for research-only strategy gate checks."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import _factor_config, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.strategy_gate")
def strategy_gate(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate explicit evidence gates without mutating frozen strategies."""
    _set_run_status(run_id, "running")
    try:
        from research.factor_baseline.run_factor_baselines import run
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.engine_contract import validate_backtest_result

        raw = dict(config or {})
        from research_platform.backend.app.selection_policy import validate_selection_policy
        selection_valid, violations = validate_selection_policy(raw)
        if not selection_valid:
            raise ValueError(f"禁止使用测试集选择配置: {violations}")
        normalized = _factor_config(raw)
        result = run(**normalized)
        validate_backtest_result(result)
        min_positive_ratio = float(raw.get("min_positive_ratio", 0.5))
        min_return = float(raw.get("min_return_net", 0.0))
        positive_ratio = float((result["total_return_net"] > 0).mean()) if len(result) else 0.0
        mean_return = float(result["total_return_net"].mean()) if len(result) else None
        checks = {
            "has_results": bool(len(result)),
            "positive_window_ratio": positive_ratio >= min_positive_ratio,
            "mean_net_return": mean_return is not None and mean_return >= min_return,
            "all_metrics_finite": bool(result[["total_return_net", "max_drawdown"]].replace([float("inf"), float("-inf")], float("nan")).notna().all().all()) if len(result) else False,
        }
        passed = all(checks.values())
        summary = {
            "task": "strategy_gate",
            "candidate_strategy": str(raw.get("candidate_strategy", "factor-common-v1")),
            "frozen_baseline": str(raw.get("frozen_baseline", "long_term_v1_quotes2")),
            "decision": "eligible_for_review" if passed else "hold",
            "automatic_promotion": False,
            "gate_passed": passed,
            "checks": checks,
            "positive_window_ratio": positive_ratio,
            "mean_net_return": mean_return,
            "thresholds": {"min_positive_ratio": min_positive_ratio, "min_return_net": min_return},
            "config": {**normalized, "top_ks": list(normalized["top_ks"]), "cost_rates": list(normalized["cost_rates"])},
        }
        artifacts = []
        for name, kind, content in (("strategy_gate_detail.csv", "strategy_gate_detail", result.to_csv(index=False).encode("utf-8-sig")), ("strategy_gate_summary.json", "strategy_gate_summary", json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **summary, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
