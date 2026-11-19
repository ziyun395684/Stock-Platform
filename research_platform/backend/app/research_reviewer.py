"""Evidence-based research review gates for Agent plans and results."""
from __future__ import annotations

import json
from typing import Any

from .schemas import ResearchReviewFinding, ResearchReviewRequest, ResearchReviewResponse
from .selection_policy import validate_selection_policy


def _finding(check: str, status: str, evidence: str, recommendation: str | None = None) -> ResearchReviewFinding:
    return ResearchReviewFinding(check=check, status=status, evidence=evidence, recommendation=recommendation)


def review_research(request: ResearchReviewRequest) -> ResearchReviewResponse:
    config = request.config
    metrics = request.metrics
    findings: list[ResearchReviewFinding] = []
    serialized = json.dumps(config, ensure_ascii=False).lower()

    future_markers = ("next_close", "next_open", "future_return", "forward_return", "未来收益")
    leaks = [marker for marker in future_markers if marker in serialized]
    findings.append(_finding(
        "data_leakage",
        "fail" if leaks else "pass",
        f"Detected markers: {leaks}" if leaks else "No configured future-data markers detected.",
        "Remove future-looking fields and rerun the review." if leaks else None,
    ))

    _, test_flags = validate_selection_policy(config)
    findings.append(_finding(
        "test_set_usage",
        "fail" if test_flags else "pass",
        f"Forbidden test-set selection settings: {test_flags}" if test_flags else "Configuration selection is isolated from the test set.",
        "Move configuration selection to validation data only." if test_flags else None,
    ))

    if any(key in config for key in ("oos", "rolling", "rolling_windows")):
        embargo = config.get("embargo_days")
        findings.append(_finding(
            "time_split_embargo",
            "pass" if isinstance(embargo, int) and embargo >= 0 else "warn",
            f"embargo_days={embargo!r}",
            "Record an explicit non-negative embargo_days value for rolling/OOS experiments." if not isinstance(embargo, int) else None,
        ))
    else:
        findings.append(_finding("time_split_embargo", "warn", "No rolling/OOS configuration evidence supplied."))

    windows = config.get("training_windows", [])
    seeds = config.get("seeds", [])
    costs = config.get("cost_rates", [])
    findings.append(_finding("window_seed_coverage", "pass" if len(set(windows)) >= 2 and len(set(seeds)) >= 2 else "warn", f"windows={windows}, seeds={seeds}", "Cover at least two windows and two seeds."))
    findings.append(_finding("cost_sensitivity", "pass" if len(set(costs)) >= 2 else "warn", f"cost_rates={costs}", "Include at least two cost rates."))

    regimes = metrics.get("market_regimes") or config.get("market_regimes") or []
    findings.append(_finding("market_regime_coverage", "pass" if len(set(regimes)) >= 2 else "warn", f"market_regimes={regimes}", "Report more than one market regime."))

    prediction_std = metrics.get("prediction_std")
    unique_ratio = metrics.get("unique_prediction_ratio")
    collapsed = (isinstance(prediction_std, (int, float)) and prediction_std < 1e-6) or (isinstance(unique_ratio, (int, float)) and unique_ratio < 0.05)
    findings.append(_finding("prediction_collapse", "fail" if collapsed else "pass" if prediction_std is not None or unique_ratio is not None else "warn", f"prediction_std={prediction_std!r}, unique_prediction_ratio={unique_ratio!r}", "Investigate output scaling and target normalization." if collapsed else None))

    for metric_name in ("max_drawdown", "turnover"):
        findings.append(_finding(metric_name, "pass" if metric_name in metrics else "warn", f"{metric_name}={metrics.get(metric_name)!r}", f"Record {metric_name} before strategy approval." if metric_name not in metrics else None))

    failed = sum(item.status == "fail" for item in findings)
    passed = failed == 0
    summary = "Review passed with evidence." if passed else f"Review failed with {failed} blocking finding(s)."
    return ResearchReviewResponse(passed=passed, findings=findings, summary=summary)
