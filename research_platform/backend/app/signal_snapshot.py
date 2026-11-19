"""Deterministic daily signal snapshot and manual review checklist."""
from __future__ import annotations

from datetime import date
from typing import Any


def build_signal_snapshot(rows: list[dict[str, Any]], snapshot_date: str, top_k: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("signal snapshot requires at least one signal")
    try:
        if len(snapshot_date) != 10 or snapshot_date[4] != "-" or snapshot_date[7] != "-":
            raise ValueError
        date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise ValueError("snapshot_date 必须是 YYYY-MM-DD") from exc
    if top_k < 1:
        raise ValueError("top_k 必须为正数")
    normalized = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            raise ValueError("每条信号必须包含 symbol")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("每条信号必须包含数值 score") from exc
        normalized.append({"snapshot_date": snapshot_date, "symbol": symbol, "score": score, "source": row.get("source", "research")})
    normalized.sort(key=lambda item: (-item["score"], item["symbol"]))
    for rank, row in enumerate(normalized, start=1):
        row["rank"] = rank
        row["selected"] = rank <= top_k
        row["action"] = "REVIEW_BUY" if row["selected"] else "WATCH"
    checklist = [{"snapshot_date": snapshot_date, "symbol": row["symbol"], "rank": row["rank"], "review_required": True, "checks": {"data_freshness": "pending", "suspension": "pending", "liquidity": "pending", "corporate_action": "pending", "human_approval": "pending"}} for row in normalized if row["selected"]]
    return normalized, checklist
