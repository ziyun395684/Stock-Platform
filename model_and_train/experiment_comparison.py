"""Uniform comparison helpers for baseline and model signal tables."""
from __future__ import annotations

import pandas as pd

from .portfolio_backtest import PortfolioBacktest
from .ranking import rank_and_select


def build_model_variant_signals(
    predictions: pd.DataFrame,
    top_k: int = 2,
    validation_weights: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "symbol", "pred_return", "event_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions 缺少字段: {sorted(missing)}")
    supplied_weights = validation_weights or {
        "pred_return": 0.6,
        "event_probability": 0.4,
    }
    model_weights = {
        key: float(value)
        for key, value in supplied_weights.items()
        if key in {"pred_return", "event_probability"}
    }
    if not model_weights:
        model_weights = {"pred_return": 1.0}
    model_total = sum(model_weights.values())
    model_weights = {
        key: value / model_total for key, value in model_weights.items()
    }
    factor_weight = float(supplied_weights.get("legacy_factor_score", 0.0))
    total_supplied = factor_weight + sum(
        float(value) for key, value in supplied_weights.items()
        if key in {"pred_return", "event_probability"}
    )
    factor_weight = factor_weight / total_supplied if total_supplied else 0.0
    mixed_weights = {
        "legacy_factor_score": factor_weight,
        **{
            key: value * (1.0 - factor_weight)
            for key, value in model_weights.items()
        },
    }
    variants = {
        "Model-1": rank_and_select(
            predictions, top_k=top_k, score_column="event_probability"
        ),
        "Model-2": rank_and_select(
            predictions, top_k=top_k, score_column="pred_return"
        ),
        "Model-3": rank_and_select(
            predictions,
            top_k=top_k,
            score_weights=model_weights,
        ),
    }
    if "legacy_factor_score" in predictions.columns:
        variants["Model-4"] = rank_and_select(
            predictions,
            top_k=top_k,
            score_weights=mixed_weights,
        )
    elif "trend_score" in predictions.columns:
        variants["Model-4"] = variants["Model-3"].copy()
    else:
        variants["Model-4"] = variants["Model-3"].copy()
    return variants


def compare_experiments(
    prices: pd.DataFrame,
    signal_tables: dict[str, pd.DataFrame],
    cost_rate: float = 0.003,
    hold_days: int = 3,
    label_threshold: float = 0.05,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    backtest_kwargs: dict | None = None,
) -> pd.DataFrame:
    rows = []
    backtest_kwargs = dict(backtest_kwargs or {})
    for strategy, signals in signal_tables.items():
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            hold_days=hold_days,
            label_threshold=label_threshold,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            **backtest_kwargs,
        ).run(prices, signals)
        rows.append({
            "strategy": strategy,
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
        })
    return pd.DataFrame(rows)
