"""Phase 10 read-only research comparison and report routes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ..agent_feedback import build_agent_feedback
from ..agent_repository import list_sessions
from ..config import settings
from ..repositories import list_experiments as db_list_experiments
from ..result_metrics import METRICS, average_metrics, read_run_metrics
from ..run_repository import list_runs
from ..memory_store import store

router = APIRouter(tags=["phase10"])


def _experiments() -> list[dict[str, Any]]:
    return db_list_experiments() if settings.storage_backend == "mysql" else store.list_experiments()


def _session_exists(session_id: str) -> bool:
    return settings.storage_backend == "mysql" and any(item["id"] == session_id for item in list_sessions())


def _run_list(experiment_id: str) -> list[dict[str, Any]]:
    return list_runs(experiment_id) if settings.storage_backend == "mysql" else []


@router.get("/experiments/compare")
def compare_experiments(ids: str) -> dict[str, Any]:
    requested = list(dict.fromkeys(item.strip() for item in ids.split(",") if item.strip()))
    if not requested:
        raise HTTPException(status_code=422, detail="ids must contain at least one experiment id")
    known = {item["id"]: item for item in _experiments()}
    rows: list[dict[str, Any]] = []
    for experiment_id in requested:
        experiment = known.get(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"experiment not found: {experiment_id}")
        runs = _run_list(experiment_id)
        metric_values: dict[str, list[float]] = {}
        completed = failed = 0
        for run in runs:
            if run.get("status") == "succeeded":
                completed += 1
                for name, values in read_run_metrics(run).items():
                    metric_values.setdefault(name, []).extend(values)
            elif run.get("status") == "failed":
                failed += 1
        rows.append({"id": experiment["id"], "experiment_id": experiment["experiment_id"], "objective": experiment["objective"], "runs": len(runs), "completed_runs": completed, "failed_runs": failed, "metrics": average_metrics(metric_values)})

    def best(metric: str, reverse: bool = True) -> str | None:
        candidates = [row for row in rows if row["metrics"].get(metric) is not None]
        if not candidates:
            return None
        if metric == "max_drawdown":
            return min(candidates, key=lambda row: abs(row["metrics"][metric]))["id"]
        return sorted(candidates, key=lambda row: row["metrics"][metric], reverse=reverse)[0]["id"]

    return {"experiments": rows, "comparison": {"best_return": best("total_return_net"), "lowest_drawdown": best("max_drawdown", reverse=False), "best_sharpe": best("sharpe")}}


def _markdown(session_id: str, feedback: dict[str, Any]) -> str:
    session = next((item for item in list_sessions() if item["id"] == session_id), None)
    objective = session.get("objective", "") if session else ""
    lines = ["# 研究结果报告", "", f"- 会话 ID：`{session_id}`", f"- 研究目标：{objective or '未记录'}", f"- 结果状态：{feedback['outcome']}", f"- 生成时间：{datetime.now(timezone.utc).isoformat()}", "", "## 运行概况", "", f"- 已完成：{feedback['completed']}", f"- 失败：{feedback['failed']}", ""]
    lines.extend(["## 汇总指标", "", "| 指标 | 数值 |", "| --- | ---: |"])
    if feedback["metrics"]:
        lines.extend(f"| {name} | {value if value is not None else '无'} |" for name, value in feedback["metrics"].items())
    else:
        lines.append("| 暂无 | 无 |")
    lines.extend(["", "## 任务明细", "", "| 任务类型 | 状态 | 错误 |", "| --- | --- | --- |"])
    lines.extend(f"| {run['task_type']} | {run['status']} | {run.get('error') or ''} |" for run in feedback["runs"])
    lines.extend(["", "## 下一步建议", ""])
    lines.extend(f"- **{suggestion['action']}**：{suggestion['reason']}" for suggestion in feedback["suggestions"])
    lines.extend(["", "> 本报告仅用于研究分析，不连接券商，不产生实盘交易指令。", ""])
    return "\n".join(lines)


@router.get("/agents/sessions/{session_id}/report/markdown", response_class=PlainTextResponse)
def agent_markdown_report(session_id: str) -> PlainTextResponse:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="agent session not found")
    return PlainTextResponse(_markdown(session_id, build_agent_feedback(session_id)), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="research-report-{session_id}.md"'})

