"""Validation-only calibration for optional prediction filters."""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from .portfolio_backtest import PortfolioBacktest
from .ranking import rank_and_select


def fit_min_pred_return_threshold(
    validation_predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_k: int = 2,
    candidates: Iterable[float | None] | None = None,
    score_weights: dict[str, float] | None = None,
    cost_rate: float = 0.003,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    hold_days: int = 3,
    label_threshold: float = 0.05,
    min_trades: int = 1,
) -> dict:
    """Choose an optional minimum prediction threshold using validation only.

    ``None`` means no filter.  The returned threshold is intended to be
    applied unchanged to the following out-of-sample window.
    """
    required = {"trade_date", "symbol", "split", "future_return", "pred_return"}
    missing = required - set(validation_predictions.columns)
    if missing:
        raise ValueError(f"validation_predictions 缺少字段: {sorted(missing)}")
    splits = set(validation_predictions["split"].dropna().astype(str))
    if splits - {"val", "validation"}:
        raise ValueError("fit_min_pred_return_threshold 只能使用验证集数据")
    if min_trades < 0:
        raise ValueError("min_trades 不能为负数")
    candidate_values = list(candidates if candidates is not None else (None, 0.0, 0.02, 0.05))
    if not candidate_values:
        raise ValueError("candidates 不能为空")

    rows = []
    best = None
    for threshold in candidate_values:
        if threshold is not None:
            threshold = float(threshold)
        kwargs = {
            "top_k": int(top_k),
            "score_column": "pred_return",
            "score_weights": score_weights,
        }
        if threshold is not None:
            kwargs["min_pred_return"] = threshold
        signals = rank_and_select(validation_predictions.copy(), **kwargs)
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            hold_days=hold_days,
            label_threshold=label_threshold,
        ).run(prices, signals)
        report_dict = asdict(report)
        row = {
            "threshold": threshold,
            "eligible": bool(report.n_trades >= min_trades),
            "n_trades": int(report.n_trades),
            "total_return_net": float(report.total_return_net),
            "total_return_gross": float(report.total_return_gross),
            "max_drawdown": float(report.max_drawdown),
        }
        rows.append(row)
        if row["eligible"] and (best is None or row["total_return_net"] > best[0]):
            best = (row["total_return_net"], threshold, report_dict)

    if best is None:
        raise ValueError("所有候选门槛都没有达到最小交易数")
    return {
        "threshold": best[1],
        "report": best[2],
        "candidates": rows,
        "selection": "validation_total_return_net",
    }
