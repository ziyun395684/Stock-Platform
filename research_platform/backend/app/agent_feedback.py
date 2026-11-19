"""Read-only Agent feedback from completed research runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .agent_repository import list_tasks
from .result_metrics import METRICS, artifact_root, average, collect_metrics, read_run_metrics
from .run_repository import get_run


def _artifact_root() -> Path:
    return artifact_root()


def _collect_metrics(value: Any, output: dict[str, list[float]]) -> None:
    collect_metrics(value, output)


def _read_run_metrics(run: dict[str, Any]) -> dict[str, list[float]]:
    # Keep the local wrapper for compatibility with existing tests and callers.
    return read_run_metrics(run, root=_artifact_root())


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _suggestions(objective: str, completed: int, failed: int, metrics: dict[str, float | None]) -> list[dict[str, Any]]:
    if failed:
        return [{"action": "retry_failed", "reason": "存在失败任务，先检查错误原因并仅重跑失败任务。"}]
    if completed == 0:
        return [{"action": "wait", "reason": "尚无完成的研究运行，等待 Worker 产出结果。"}]
    suggestions: list[dict[str, Any]] = []
    mean_return = metrics.get("total_return_net")
    drawdown = metrics.get("max_drawdown")
    if mean_return is not None and mean_return <= 0:
        suggestions.append({"action": "adjust_hypothesis", "reason": "平均净收益非正，建议更换因子组合或扩大样本外区间。"})
    if drawdown is not None and abs(drawdown) >= 0.3:
        suggestions.append({"action": "reduce_risk", "reason": "平均最大回撤偏高，建议降低 Top-K、换用更长训练窗口或增加风险过滤。"})
    if metrics.get("turnover") is not None and metrics["turnover"] >= 2:
        suggestions.append({"action": "test_costs", "reason": "换手率偏高，建议增加成本率网格并比较净收益稳定性。"})
    if not suggestions:
        suggestions.append({"action": "candidate_review", "reason": f"研究目标“{objective}”已完成基础运行，建议查看报告并进行候选策略评审。"})
    suggestions.append({"action": "iterate", "reason": "可复制当前实验，调整窗口、seed、成本率或因子任务后继续运行。"})
    return suggestions


def build_agent_feedback(session_id: str) -> dict[str, Any]:
    tasks = list_tasks(session_id)
    run_results: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {}
    failed = 0
    completed = 0
    objective = ""
    for task in tasks:
        payload = task.get("payload") or {}
        run_id = payload.get("run_id")
        run = get_run(run_id) if run_id else None
        status = run.get("status") if run else task.get("status", "planned")
        if status == "succeeded":
            completed += 1
            for name, values in _read_run_metrics(run).items():
                metric_values.setdefault(name, []).extend(values)
        if status == "failed":
            failed += 1
        if not objective:
            objective = str(payload.get("objective") or (payload.get("config") or {}).get("objective") or "")
        run_results.append({"task_id": task["id"], "task_type": task["task_type"], "run_id": run_id, "status": status, "error": (run or {}).get("error_message") or payload.get("error")})
    metrics = {name: _average(values) for name, values in metric_values.items()}
    suggestions = _suggestions(objective, completed, failed, metrics)
    return {"session_id": session_id, "outcome": "failed" if failed else ("completed" if tasks and completed == len(tasks) else "running"), "completed": completed, "failed": failed, "runs": run_results, "metrics": metrics, "suggestions": suggestions}
