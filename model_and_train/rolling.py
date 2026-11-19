"""Leakage-aware rolling out-of-sample date windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import pandas as pd


@dataclass(frozen=True)
class RollingWindow:
    index: int
    train_dates: tuple
    val_dates: tuple
    test_dates: tuple
    embargo_after_train: tuple
    embargo_after_val: tuple


def classify_market_regime(
    returns: pd.Series,
    volatility: pd.Series,
    high_volatility_threshold: float = 0.015,
    directional_threshold: float = 0.10,
) -> str:
    """Classify a window for coverage reporting without using labels/predictions."""
    cumulative_return = float(pd.to_numeric(returns, errors="coerce").dropna().sum() or 0.0)
    mean_volatility = float(
        pd.to_numeric(volatility, errors="coerce").dropna().mean() or 0.0
    )
    if mean_volatility >= high_volatility_threshold:
        return "high_volatility"
    if cumulative_return >= directional_threshold:
        return "up"
    if cumulative_return <= -directional_threshold:
        return "down"
    return "sideways"


def annotate_regime_report(report: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach a causal market-regime label to each rolling report row."""
    out = report.copy()
    if out.empty or not {"test_start", "test_end"}.issubset(out.columns):
        out["market_regime"] = "unknown"
        return out
    if not {"trade_date", "benchmark_return_1", "volatility_20"}.issubset(panel.columns):
        out["market_regime"] = "unknown"
        return out
    for index, row in out.iterrows():
        dates = pd.to_numeric(panel["trade_date"], errors="coerce")
        window = panel[dates.between(row["test_start"], row["test_end"])]
        out.at[index, "market_regime"] = classify_market_regime(
            window["benchmark_return_1"], window["volatility_20"]
        ) if len(window) else "unknown"
    return out


def rolling_windows(
    dates: Iterable,
    train_days: int,
    val_days: int,
    test_days: int,
    embargo_days: int = 3,
    step_days: int | None = None,
) -> Iterator[RollingWindow]:
    """Yield chronological train/validation/test windows with embargo gaps."""
    if min(train_days, val_days, test_days) <= 0:
        raise ValueError("滚动窗口长度必须大于 0")
    if embargo_days < 0:
        raise ValueError("embargo_days 不能为负数")
    unique = tuple(sorted(set(dates)))
    step = int(step_days or test_days)
    if step <= 0:
        raise ValueError("step_days 必须大于 0")
    required = train_days + embargo_days + val_days + embargo_days + test_days
    if len(unique) < required:
        raise ValueError(f"滚动窗口日期不足：需要至少 {required} 个，实际 {len(unique)} 个")

    index = 0
    start = 0
    while start + required <= len(unique):
        train_end = start + train_days
        val_start = train_end + embargo_days
        val_end = val_start + val_days
        test_start = val_end + embargo_days
        test_end = test_start + test_days
        yield RollingWindow(
            index=index,
            train_dates=unique[start:train_end],
            val_dates=unique[val_start:val_end],
            test_dates=unique[test_start:test_end],
            embargo_after_train=unique[train_end:val_start],
            embargo_after_val=unique[val_end:test_start],
        )
        index += 1
        start += step


def run_rolling_experiment(
    windows: Iterable[RollingWindow],
    train_fn: Callable[[tuple, tuple, RollingWindow], Any],
    predict_fn: Callable[[Any, tuple, RollingWindow], Any],
    evaluate_fn: Callable[[Any, Any, RollingWindow], dict] | None = None,
) -> list[dict]:
    """Execute leakage-aware rolling train/predict/evaluate callbacks."""
    results = []
    for window in windows:
        model = train_fn(window.train_dates, window.val_dates, window)
        prediction = predict_fn(model, window.test_dates, window)
        metrics = evaluate_fn(model, prediction, window) if evaluate_fn else {}
        results.append({
            "window": window.index,
            "train_start": window.train_dates[0],
            "train_end": window.train_dates[-1],
            "val_start": window.val_dates[0],
            "val_end": window.val_dates[-1],
            "test_start": window.test_dates[0],
            "test_end": window.test_dates[-1],
            "prediction": prediction,
            "metrics": metrics,
            "model": model,
        })
    return results
