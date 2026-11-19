"""Deterministic paper portfolio state transitions with no broker side effects."""
from __future__ import annotations

from typing import Any


def apply_paper_snapshot(
    previous: dict[str, Any] | None,
    signals: list[dict[str, Any]],
    snapshot_date: str,
    top_k: int,
    cost_rate: float = 0.003,
) -> dict[str, Any]:
    if top_k < 1 or cost_rate < 0 or cost_rate > 0.1:
        raise ValueError("top_k 或 cost_rate 无效")
    previous = previous or {"capital": 1.0, "holdings": [], "date": None}
    capital = float(previous.get("capital", 1.0))
    if capital <= 0:
        raise ValueError("paper portfolio capital must be positive")
    approved = [row for row in signals if row.get("approved") is True and str(row.get("symbol", "")).strip()]
    approved.sort(key=lambda row: (-float(row.get("score", 0)), str(row["symbol"])))
    selected = approved[:top_k]
    symbols = [str(row["symbol"]) for row in selected]
    old_symbols = set(previous.get("holdings", []))
    new_symbols = set(symbols)
    turnover = len(old_symbols.symmetric_difference(new_symbols)) / max(len(old_symbols | new_symbols), 1)
    returns = [float(row.get("return_1d", 0.0)) for row in selected]
    gross_return = sum(returns) / len(returns) if returns else 0.0
    net_return = gross_return - cost_rate * turnover
    new_capital = capital * (1.0 + net_return)
    return {"date": snapshot_date, "capital": new_capital, "holdings": symbols, "gross_return": gross_return, "net_return": net_return, "turnover": turnover, "cost_rate": cost_rate, "automatic_ordering": False}
