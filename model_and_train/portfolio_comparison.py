"""Comparable Top-K portfolio experiments."""
from __future__ import annotations

import pandas as pd

from .portfolio_backtest import PortfolioBacktest
from .ranking import rank_and_select


def compare_top_k(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    top_ks=(1, 2, 3, 5),
    cost_rate: float = 0.003,
    hold_days: int = 3,
    label_threshold: float = 0.05,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    exit_mode: str = "fixed",
) -> pd.DataFrame:
    """Run identical portfolio rules for several Top-K values."""
    if "score" not in predictions.columns:
        raise ValueError("predictions 必须包含 score 列")
    rows = []
    for top_k in top_ks:
        signals = rank_and_select(predictions, top_k=int(top_k), score_column="score")
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            hold_days=hold_days,
            label_threshold=label_threshold,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            exit_mode=exit_mode,
        ).run(prices, signals)
        rows.append({
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
            "top_k_hit_rate": report.top_k_hit_rate,
            "turnover": report.turnover,
            "average_holdings": report.average_holdings,
        })
    return pd.DataFrame(rows)
