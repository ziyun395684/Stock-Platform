"""Simple buy-and-hold benchmark with portfolio-compatible metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_report(
    prices: pd.DataFrame,
    symbol: str,
    cost_rate: float = 0.003,
) -> dict:
    required = {"trade_date", "symbol", "open", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices 缺少字段: {sorted(missing)}")
    frame = prices[prices["symbol"] == symbol].sort_values("trade_date").copy()
    if frame.empty:
        raise ValueError(f"找不到基准标的: {symbol}")
    valid = frame[(frame["open"] > 0) & (frame["close"] > 0)].copy()
    if len(valid) < 2:
        return {"symbol": symbol, "n_trades": 0, "total_return_gross": 0.0,
                "total_return_net": 0.0, "max_drawdown": 0.0, "sharpe": None}
    buy = float(valid.iloc[0]["open"])
    sell = float(valid.iloc[-1]["close"])
    gross = sell / buy - 1.0
    net = gross - float(cost_rate)
    close_returns = valid["close"].pct_change().dropna()
    nav = (1.0 + close_returns).cumprod()
    peak = nav.cummax()
    drawdown = ((peak - nav) / peak).max() if len(nav) else 0.0
    std = close_returns.std(ddof=1)
    sharpe = float(np.sqrt(252.0) * close_returns.mean() / std) if std > 0 else None
    return {
        "symbol": symbol,
        "n_trades": 1,
        "total_return_gross": float(gross),
        "total_return_net": float(net),
        "max_drawdown": float(drawdown),
        "sharpe": sharpe,
    }
