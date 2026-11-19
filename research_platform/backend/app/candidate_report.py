"""Aggregate research runs into an auditable candidate-strategy report."""
from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from typing import Any


METRIC_NAMES = ("total_return_net", "max_drawdown", "sharpe", "turnover", "rank_ic", "mean_rank_ic")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _dimension(value: Any) -> str:
    return "unknown" if value is None or value == "" else str(value)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "positive_ratio": None}
    average = mean(values)
    variance = mean([(value - average) ** 2 for value in values])
    return {
        "count": len(values),
        "mean": average,
        "std": sqrt(variance),
        "positive_ratio": sum(value > 0 for value in values) / len(values),
    }


def summarize_observations(observations: list[dict[str, Any]], *, total_runs: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a report from normalized observations.

    Each observation contains optional ``window``, ``seed``, ``cost_rate`` and
    ``regime`` dimensions plus numeric metrics. The function deliberately does
    not rank or discard observations by return.
    """
    dimensions = {key: set() for key in ("window", "seed", "cost_rate", "regime")}
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        key = tuple(_dimension(observation.get(name)) for name in dimensions)
        for name, value in zip(dimensions, key):
            if value != "unknown":
                dimensions[name].add(value)
        for metric in METRIC_NAMES:
            number = _number(observation.get(metric))
            if number is not None:
                grouped[key][metric].append(number)

    groups = []
    for key in sorted(grouped):
        row = dict(zip(dimensions, key))
        row["metrics"] = {metric: _stats(values) for metric, values in sorted(grouped[key].items())}
        groups.append(row)

    returns = [_number(item.get("total_return_net")) for item in observations]
    returns = [value for value in returns if value is not None]
    coverage = {
        "windows": sorted(dimensions["window"]),
        "seeds": sorted(dimensions["seed"]),
        "cost_rates": sorted(dimensions["cost_rate"]),
        "market_regimes": sorted(dimensions["regime"]),
    }
    checks = {
        "all_windows_retained": len(coverage["windows"]) >= 2,
        "seed_stability_available": len(coverage["seeds"]) >= 2,
        "cost_sensitivity_available": len(coverage["cost_rates"]) >= 2,
        "market_regime_split_available": len(coverage["market_regimes"]) >= 2,
        "not_best_window_only": len(groups) > 1,
    }
    from .selection_policy import validate_selection_policy
    selection_valid, selection_violations = validate_selection_policy(config or {})
    checks["test_set_not_used_for_selection"] = selection_valid
    eligible = bool(returns) and all(checks.values())
    return {
        "report_type": "candidate_strategy_summary",
        "selection_policy": "aggregate all completed windows, seeds, costs, and regimes; never select by one best window",
        "total_runs": total_runs,
        "completed_observations": len(observations),
        "coverage": coverage,
        "checks": checks,
        "selection_policy_violations": selection_violations,
        "overall": _stats(returns),
        "groups": groups,
        "recommendation": {
            "status": "candidate_recommended" if eligible else "insufficient_evidence",
            "eligible": eligible,
            "reason": None if eligible else ("测试集被用于配置选择，禁止生成候选建议。" if not selection_valid else "需要至少两个窗口、两个 seed、两个成本率和两个市场阶段，并保留完整结果。"),
        },
    }
