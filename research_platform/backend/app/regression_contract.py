"""Regression comparisons for platform adapters and legacy StockRNN reports."""
from __future__ import annotations

from typing import Any, Iterable


def compare_tabular_results(
    expected: Any,
    actual: Any,
    *,
    key_columns: Iterable[str] = (),
    value_columns: Iterable[str] = (),
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare two pandas-like tables without depending on row order."""
    expected_columns = list(getattr(expected, "columns", ()))
    actual_columns = list(getattr(actual, "columns", ()))
    keys = list(key_columns)
    values = list(value_columns)
    required = set(keys + values)
    missing = sorted((required - set(expected_columns)) | (required - set(actual_columns)))
    if missing:
        return {"equal": False, "reason": "missing_columns", "columns": missing}
    if len(expected) != len(actual):
        return {"equal": False, "reason": "row_count", "expected": len(expected), "actual": len(actual)}
    if keys:
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
    mismatches: list[dict[str, Any]] = []
    for column in values:
        left = expected[column].tolist()
        right = actual[column].tolist()
        for index, (a, b) in enumerate(zip(left, right)):
            try:
                same = abs(float(a) - float(b)) <= tolerance
            except (TypeError, ValueError):
                same = a == b
            if not same:
                mismatches.append({"row": index, "column": column, "expected": a, "actual": b})
    return {"equal": not mismatches, "reason": None if not mismatches else "value_mismatch", "mismatches": mismatches[:20], "checked_rows": len(expected), "checked_columns": values}


def compare_factor_columns(expected: Any, actual: Any, factor_columns: Iterable[str]) -> dict[str, Any]:
    columns = list(factor_columns)
    return compare_tabular_results(expected, actual, key_columns=("trade_date", "symbol"), value_columns=columns)


def compare_backtest_reports(expected: Any, actual: Any) -> dict[str, Any]:
    keys = [column for column in ("factor_mode", "top_k", "start", "end", "hold_days", "cost_rate") if column in getattr(expected, "columns", ()) and column in getattr(actual, "columns", ())]
    values = [column for column in ("total_return_net", "annualized_return", "max_drawdown", "sharpe", "calmar", "hit_rate", "turnover", "signal_count", "n_trades", "skipped_overlap_signals") if column in getattr(expected, "columns", ()) and column in getattr(actual, "columns", ())]
    return compare_tabular_results(expected, actual, key_columns=keys, value_columns=values, tolerance=1e-8)

