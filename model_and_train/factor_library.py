"""Causal common factors shared by the legacy ETF strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .legacy_strategy_adapter import ma_energy_series, trend_score_series


def add_common_factors(
    panel: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    roc_windows: tuple[int, ...] = (5, 10, 20, 60),
    trend_periods: tuple[int, ...] = (24, 25),
    ma_energy_windows: tuple[int, ...] = (38, 39, 41),
) -> pd.DataFrame:
    """Add common legacy factors without using future observations."""
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    if any(window <= 0 for window in roc_windows + ma_energy_windows):
        raise ValueError("因子窗口必须大于 0")
    if any(period <= 1 for period in trend_periods):
        raise ValueError("趋势窗口必须大于 1")

    out = panel.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    grouped = out.groupby("symbol", sort=False)

    for window in roc_windows:
        out[f"roc_{window}"] = grouped["close"].transform(
            lambda series, window=window: series.pct_change(window, fill_method=None)
        )
    for period in trend_periods:
        out[f"trend_score_{period}"] = grouped["close"].transform(
            lambda series, period=period: trend_score_series(series, period)
        )
    for window in ma_energy_windows:
        out[f"ma_energy_{window}"] = grouped["close"].transform(
            lambda series, window=window: ma_energy_series(series, window)
        )

    volume_ma5 = grouped["volume"].transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )
    volume_ma20 = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=1).mean()
    )
    out["volume_ratio_5_20"] = volume_ma5 / volume_ma20.replace(0, np.nan)

    if benchmark_frame is not None:
        benchmark_required = {"trade_date", "close"}
        benchmark_missing = benchmark_required - set(benchmark_frame.columns)
        if benchmark_missing:
            raise ValueError(f"benchmark_frame 缺少字段: {sorted(benchmark_missing)}")
        benchmark = benchmark_frame[["trade_date", "close"]].copy()
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.sort_values("trade_date")
        for window in roc_windows:
            benchmark_returns = benchmark["close"].pct_change(window, fill_method=None)
            lookup = pd.Series(benchmark_returns.to_numpy(), index=benchmark["trade_date"])
            out[f"relative_return_to_benchmark_{window}"] = out.apply(
                lambda row, window=window: row[f"roc_{window}"] - lookup.get(row["trade_date"], np.nan),
                axis=1,
            )

    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
