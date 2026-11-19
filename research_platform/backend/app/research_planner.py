"""Deterministic research-plan drafting for the governed Agent layer."""
from __future__ import annotations

from .schemas import ResearchGoal, ResearchPlan


def draft_research_plan(goal: ResearchGoal) -> ResearchPlan:
    objective = goal.objective.lower()
    tasks: list[str] = []

    if any(word in objective for word in ("因子", "factor", "信号", "signal", "质量", "quality")):
        tasks.extend(["factor_quality", "factor_correlation"])
    if any(word in objective for word in ("训练", "模型", "transformer", "lstm", "train")):
        tasks.append("factor_common_train")
    if any(word in objective for word in ("回测", "验证", "样本外", "oos", "backtest", "validate")):
        tasks.append("factor_common_oos")
    if any(word in objective for word in ("成本", "top-k", "topk", "敏感", "cost", "sensitivity")):
        tasks.append("factor_sensitivity")

    if not tasks:
        tasks = ["factor_quality", "factor_common_train", "factor_common_oos"]
    tasks = list(dict.fromkeys(tasks))
    return ResearchPlan(
        goal=goal,
        tasks=tasks,
        estimated_runs=len(tasks),
        requires_approval=True,
    )
