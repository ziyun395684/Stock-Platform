"""Reusable rolling-window strategy comparison helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import pandas as pd

from .portfolio_backtest import PortfolioBacktest
from .ranking import rank_and_select
from .signal_filter import fit_min_pred_return_threshold


def _report_row(window: object, top_k: int, report) -> dict:
    return {
        "window": window,
        "top_k": int(top_k),
        "n_trades": report.n_trades,
        "signal_count": report.signal_count,
        "total_return_gross": report.total_return_gross,
        "total_return_net": report.total_return_net,
        "annualized_return": report.annualized_return,
        "max_drawdown": report.max_drawdown,
        "sharpe": report.sharpe,
        "calmar": report.calmar,
        "hit_rate": report.hit_rate,
        "turnover": report.turnover,
        "average_holdings": report.average_holdings,
    }


def evaluate_window_topk(
    predictions_by_window: Mapping[object, pd.DataFrame],
    prices: pd.DataFrame,
    *,
    top_ks: Iterable[int] = (1, 2, 3, 5),
    score_column: str = "score",
    score_weights: dict[str, float] | None = None,
    cost_rate: float = 0.003,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    hold_days: int = 3,
    label_threshold: float = 0.05,
) -> pd.DataFrame:
    """Evaluate identical Top-K rules independently on each rolling window.

    ``predictions_by_window`` must contain only that window's test-date
    predictions.  The function deliberately does not fit weights or alter
    scores, so it can be used as a post-hoc audit of frozen rolling outputs.
    """
    required_prices = {"trade_date", "symbol", "open", "close"}
    missing_prices = required_prices - set(prices.columns)
    if missing_prices:
        raise ValueError(f"prices 缺少字段: {sorted(missing_prices)}")
    rows: list[dict] = []
    for window, predictions in predictions_by_window.items():
        required_predictions = {"trade_date", "symbol", score_column}
        missing_predictions = required_predictions - set(predictions.columns)
        if missing_predictions:
            raise ValueError(
                f"window={window} predictions 缺少字段: {sorted(missing_predictions)}"
            )
        for top_k in top_ks:
            signals = rank_and_select(
                predictions.copy(),
                top_k=int(top_k),
                score_column=score_column,
                score_weights=score_weights,
            )
            report = PortfolioBacktest(
                cost_rate=cost_rate,
                slippage_rate=slippage_rate,
                initial_capital=initial_capital,
                hold_days=hold_days,
                label_threshold=label_threshold,
            ).run(prices, signals)
            rows.append(_report_row(window, int(top_k), report))
    return pd.DataFrame(rows)


def evaluate_validation_calibrated_filter(
    validation_by_window: Mapping[object, pd.DataFrame],
    test_by_window: Mapping[object, pd.DataFrame],
    prices: pd.DataFrame,
    *,
    top_k: int = 2,
    candidates: Iterable[float | None] | None = None,
    score_weights_by_window: Mapping[object, dict[str, float]] | None = None,
    cost_rate: float = 0.003,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    hold_days: int = 3,
    label_threshold: float = 0.05,
    min_trades: int = 1,
) -> pd.DataFrame:
    """Fit a threshold on each validation window and apply it to its test window."""
    rows: list[dict] = []
    for window, validation in validation_by_window.items():
        if window not in test_by_window:
            raise ValueError(f"window={window} 缺少测试预测")
        score_weights = score_weights_by_window.get(window) if score_weights_by_window else None
        calibration = fit_min_pred_return_threshold(
            validation,
            prices,
            top_k=top_k,
            candidates=candidates,
            score_weights=score_weights,
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            hold_days=hold_days,
            label_threshold=label_threshold,
            min_trades=min_trades,
        )
        threshold = calibration["threshold"]
        kwargs = {"top_k": int(top_k), "score_column": "pred_return"}
        if threshold is not None:
            kwargs["min_pred_return"] = threshold
        kwargs["score_weights"] = score_weights
        signals = rank_and_select(test_by_window[window].copy(), **kwargs)
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            hold_days=hold_days,
            label_threshold=label_threshold,
        ).run(prices, signals)
        rows.append(
            {
                "window": window,
                "signal_filter_threshold": threshold,
                "validation_total_return_net": calibration["report"]["total_return_net"],
                **_report_row(window, int(top_k), report),
            }
        )
    return pd.DataFrame(rows)
