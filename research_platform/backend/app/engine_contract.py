"""Contract checks for tabular outputs returned by StockRNN backtest runners."""
from __future__ import annotations

from typing import Any


REQUIRED_BACKTEST_COLUMNS = frozenset({"total_return_net", "max_drawdown"})


def validate_backtest_result(result: Any) -> None:
    """Fail fast before an engine result can be persisted as a successful run."""
    columns = set(getattr(result, "columns", ()))
    missing = REQUIRED_BACKTEST_COLUMNS - columns
    if missing:
        raise ValueError(f"StockRNN 回测结果缺少必要字段: {sorted(missing)}")
    if len(result) == 0:
        raise ValueError("StockRNN 回测结果为空")
    for column in REQUIRED_BACKTEST_COLUMNS:
        values = result[column]
        if hasattr(values, "notna") and not bool(values.notna().all()):
            raise ValueError(f"StockRNN 回测结果字段包含空值: {column}")
        if hasattr(values, "replace"):
            cleaned = values.replace([float("inf"), float("-inf")], float("nan"))
            if hasattr(cleaned, "notna") and not bool(cleaned.notna().all()):
                raise ValueError(f"StockRNN 回测结果字段包含非有限值: {column}")
