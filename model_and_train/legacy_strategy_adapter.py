"""Pure-Pandas adapters for replaying legacy ETF strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd



def ma_energy_series(close: pd.Series, window: int) -> pd.Series:
    """Reproduce the legacy ``ma_energy`` factor without future data."""
    if window <= 0:
        raise ValueError("window 必须大于 0")
    ma = pd.to_numeric(close, errors="coerce").rolling(window).mean()
    diff = ma.diff()
    direction = pd.Series(np.nan, index=close.index, dtype="float64")
    direction.loc[diff > 0] = 1.0
    direction.loc[diff < 0] = -1.0
    direction.loc[diff == 0] = 0.0
    groups = direction.ne(direction.shift()).cumsum()
    valid_direction = direction.ne(0) & direction.notna()
    run_length = valid_direction.astype(int).groupby(groups).cumsum()
    energy = direction * run_length
    energy.loc[ma.isna()] = np.nan
    return energy


def trend_score_series(close: pd.Series, period: int = 25) -> pd.Series:
    """Reproduce the legacy annualized-log-trend times R-squared score."""
    if period <= 1:
        raise ValueError("period 必须大于 1")
    values = pd.to_numeric(close, errors="coerce")
    result = pd.Series(np.nan, index=close.index, dtype="float64")
    x = np.arange(period, dtype=float)
    sum_x = x.sum()
    denominator = period * (x**2).sum() - sum_x**2
    for end in range(period - 1, len(values)):
        window = values.iloc[end - period + 1 : end + 1]
        if window.isna().any() or (window <= 0).any():
            continue
        y = np.log(window.to_numpy(dtype=float))
        sum_y = y.sum()
        sum_xy = (y * x).sum()
        slope = (period * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / period
        predicted = slope * x + intercept
        ss_res = ((y - predicted) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        result.iloc[end] = (np.exp(slope * 250.0) - 1.0) * r_squared
    return result


def add_ma_energy_features(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = (39, 41),
) -> pd.DataFrame:
    """Add grouped ROC and moving-average energy features to an ETF panel."""
    required = {"trade_date", "symbol", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    out = panel.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    grouped = out.groupby("symbol", sort=False)
    for window in windows:
        out[f"ma_energy_{window}"] = grouped["close"].transform(
            lambda series, window=window: ma_energy_series(series, window)
        )
    out["roc_20"] = grouped["close"].transform(
        lambda series: series.pct_change(20, fill_method=None)
    )
    out["roc_21"] = grouped["close"].transform(
        lambda series: series.pct_change(21, fill_method=None)
    )
    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def build_ma_energy_signals(
    panel: pd.DataFrame,
    top_k: int = 2,
    buy_window: int = 41,
    sell_window: int = 39,
    sell_roc_window: int = 20,
    sell_roc_threshold: float = 0.2,
) -> pd.DataFrame:
    """Replay ``均线能量86_13.py`` as a current-StockRNN signal table.

    The legacy rules are:
    ``buy: ma_energy(close, 41) > 0``;
    ``sell: roc(close, 20) > 0.2 OR ma_energy(close, 39) < 0``;
    rank eligible symbols by ``roc(close, 21)`` and keep Top-K.
    """
    if sell_roc_window != 20:
        raise ValueError("当前适配器固定复现 roc(close,20) 卖出规则")
    out = add_ma_energy_features(panel, windows=(sell_window, buy_window))
    out["score"] = out["roc_21"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    out["buy_eligible"] = out[f"ma_energy_{buy_window}"] > 0
    out["sell_trigger"] = (
        (out["roc_20"] > sell_roc_threshold)
        | (out[f"ma_energy_{sell_window}"] < 0)
    )
    out["eligible"] = out["buy_eligible"] & ~out["sell_trigger"]
    if out["eligible"].sum() == 0:
        out["rank"] = np.nan
        out["selected"] = False
        out["weight"] = 0.0
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    # SelectBySignal in the legacy engine carries current holdings forward,
    # then removes sell-triggered symbols before SelectTopK is applied.
    selected_keys: set[tuple[object, str]] = set()
    rank_lookup: dict[tuple[object, str], int] = {}
    holdings: set[str] = set()
    for trade_date, day in out.groupby("trade_date", sort=True):
        buy_symbols = set(day.loc[day["buy_eligible"], "symbol"].astype(str))
        sell_symbols = set(day.loc[day["sell_trigger"], "symbol"].astype(str))
        candidates = (buy_symbols | holdings) - sell_symbols
        candidate_rows = day[day["symbol"].astype(str).isin(candidates)].copy()
        candidate_rows = candidate_rows.sort_values(
            ["score", "symbol"], ascending=[False, True], na_position="last"
        )
        selected_rows = candidate_rows.head(top_k)
        holdings = set(selected_rows["symbol"].astype(str))
        for rank, (_, row) in enumerate(candidate_rows.iterrows(), start=1):
            rank_lookup[(trade_date, str(row["symbol"]))] = rank
        selected_keys.update(
            (trade_date, str(symbol)) for symbol in selected_rows["symbol"].astype(str)
        )
    out["selected"] = [
        (date, str(symbol)) in selected_keys
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    out["rank"] = [
        rank_lookup.get((date, str(symbol)), np.nan)
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    selected_counts = out.groupby("trade_date")["selected"].transform("sum")
    out["weight"] = np.where(out["selected"], 1.0 / selected_counts, 0.0)
    return out.sort_values(["trade_date", "rank", "symbol"], na_position="last").reset_index(drop=True)


def build_global_asset_signals(
    panel: pd.DataFrame,
    top_k: int = 1,
    buy_roc_window: int = 20,
    buy_roc_threshold: float = 0.06,
    buy_energy_window: int = 41,
    sell_roc_window: int = 20,
    sell_roc_threshold: float = 0.2,
    sell_energy_window: int = 38,
    rank_roc_window: int = 21,
) -> pd.DataFrame:
    """Replay ``全球资产77_13.py`` with stateful legacy selection semantics.

    The legacy rules are:
    ``buy: roc(close, 20) > 0.06 OR ma_energy(close, 41) > 0``;
    ``sell: roc(close, 20) > 0.2 OR ma_energy(close, 38) < 0``;
    rank eligible symbols by ``roc(close, 21)`` and keep Top-K.
    """
    required = {"trade_date", "symbol", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if buy_roc_window != sell_roc_window:
        raise ValueError("当前适配器要求买入和卖出使用相同的 ROC 窗口")

    out = add_ma_energy_features(
        panel,
        windows=(buy_energy_window, sell_energy_window),
    )
    grouped = out.groupby("symbol", sort=False)
    roc_column = f"roc_{buy_roc_window}"
    rank_column = f"roc_{rank_roc_window}"
    out[roc_column] = grouped["close"].transform(
        lambda series: series.pct_change(buy_roc_window, fill_method=None)
    )
    out[rank_column] = grouped["close"].transform(
        lambda series: series.pct_change(rank_roc_window, fill_method=None)
    )
    out["score"] = out[rank_column].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    out["buy_eligible"] = (
        (out[roc_column] > buy_roc_threshold)
        | (out[f"ma_energy_{buy_energy_window}"] > 0)
    )
    out["sell_trigger"] = (
        (out[roc_column] > sell_roc_threshold)
        | (out[f"ma_energy_{sell_energy_window}"] < 0)
    )

    selected_keys: set[tuple[object, str]] = set()
    rank_lookup: dict[tuple[object, str], int] = {}
    holdings: set[str] = set()
    for trade_date, day in out.groupby("trade_date", sort=True):
        buy_symbols = set(day.loc[day["buy_eligible"], "symbol"].astype(str))
        sell_symbols = set(day.loc[day["sell_trigger"], "symbol"].astype(str))
        candidates = (buy_symbols | holdings) - sell_symbols
        candidate_rows = day[day["symbol"].astype(str).isin(candidates)].sort_values(
            ["score", "symbol"], ascending=[False, True], na_position="last"
        )
        selected_rows = candidate_rows.head(top_k)
        holdings = set(selected_rows["symbol"].astype(str))
        for rank, (_, row) in enumerate(candidate_rows.iterrows(), start=1):
            rank_lookup[(trade_date, str(row["symbol"]))] = rank
        selected_keys.update(
            (trade_date, str(symbol)) for symbol in selected_rows["symbol"].astype(str)
        )
    out["selected"] = [
        (date, str(symbol)) in selected_keys
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    out["rank"] = [
        rank_lookup.get((date, str(symbol)), np.nan)
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    selected_counts = out.groupby("trade_date")["selected"].transform("sum")
    out["weight"] = np.where(out["selected"], 1.0 / selected_counts, 0.0)
    return out.sort_values(["trade_date", "rank", "symbol"], na_position="last").reset_index(drop=True)


def build_composite_strategy_signals(
    panel: pd.DataFrame,
    top_k: int = 1,
    trend_period: int = 25,
    sell_roc_window: int = 18,
    sell_roc_threshold: float = 0.15,
    trend_weight: float = 0.30,
    momentum_weight: float = 0.18,
    use_composite_score: bool = True,
) -> pd.DataFrame:
    """Replay the legacy composite-ranking strategies.

    The composite score is
    ``trend_score(period) * trend_weight + (ROC(5)+ROC(10)) * momentum_weight
    + MA(volume,5) / MA(volume,20)``.  With ``use_composite_score=False`` the
    score is only the trend score, matching ``十年.py``.
    """
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    out = panel.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    grouped = out.groupby("symbol", sort=False)
    out[f"trend_score_{trend_period}"] = grouped["close"].transform(
        lambda series: trend_score_series(series, trend_period)
    )
    out[f"roc_{sell_roc_window}"] = grouped["close"].transform(
        lambda series: series.pct_change(sell_roc_window, fill_method=None)
    )
    out["roc_5"] = grouped["close"].transform(
        lambda series: series.pct_change(5, fill_method=None)
    )
    out["roc_10"] = grouped["close"].transform(
        lambda series: series.pct_change(10, fill_method=None)
    )
    volume_ma5 = grouped["volume"].transform(lambda series: series.rolling(5, min_periods=1).mean())
    volume_ma20 = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=1).mean())
    out["volume_ratio"] = volume_ma5 / volume_ma20.replace(0, np.nan)
    trend_column = f"trend_score_{trend_period}"
    if use_composite_score:
        out["score"] = (
            out[trend_column] * trend_weight
            + (out["roc_5"] + out["roc_10"]) * momentum_weight
            + out["volume_ratio"]
        )
    else:
        out["score"] = out[trend_column]
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    out["sell_trigger"] = out[f"roc_{sell_roc_window}"] > sell_roc_threshold

    selected_keys: set[tuple[object, str]] = set()
    rank_lookup: dict[tuple[object, str], int] = {}
    holdings: set[str] = set()
    for trade_date, day in out.groupby("trade_date", sort=True):
        sell_symbols = set(day.loc[day["sell_trigger"], "symbol"].astype(str))
        candidates = holdings | set(day["symbol"].astype(str))
        candidates -= sell_symbols
        candidate_rows = day[day["symbol"].astype(str).isin(candidates)].sort_values(
            ["score", "symbol"], ascending=[False, True], na_position="last"
        )
        selected_rows = candidate_rows.head(top_k)
        holdings = set(selected_rows["symbol"].astype(str))
        for rank, (_, row) in enumerate(candidate_rows.iterrows(), start=1):
            rank_lookup[(trade_date, str(row["symbol"]))] = rank
        selected_keys.update(
            (trade_date, str(symbol)) for symbol in selected_rows["symbol"].astype(str)
        )
    out["selected"] = [
        (date, str(symbol)) in selected_keys
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    out["rank"] = [
        rank_lookup.get((date, str(symbol)), np.nan)
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    selected_counts = out.groupby("trade_date")["selected"].transform("sum")
    out["weight"] = np.where(out["selected"], 1.0 / selected_counts, 0.0)
    return out.sort_values(["trade_date", "rank", "symbol"], na_position="last").reset_index(drop=True)


def build_long_term_strategy_signals(
    panel: pd.DataFrame,
    top_k: int = 1,
    trend_period: int = 24,
    buy_roc_window: int = 5,
    buy_roc_limit: float = 0.12,
    sell_roc_window: int = 59,
    sell_roc_limit: float = 0.24,
) -> pd.DataFrame:
    """Replay ``长期策略.py`` with stateful legacy selection semantics."""
    required = {"trade_date", "symbol", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    out = panel.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    grouped = out.groupby("symbol", sort=False)
    out["roc_5"] = grouped["close"].transform(
        lambda series: series.pct_change(buy_roc_window, fill_method=None)
    )
    out["roc_59"] = grouped["close"].transform(
        lambda series: series.pct_change(sell_roc_window, fill_method=None)
    )
    out["trend_score_24"] = grouped["close"].transform(
        lambda series: trend_score_series(series, trend_period)
    )
    out["score"] = out["trend_score_24"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    out["buy_eligible"] = out["roc_5"] < buy_roc_limit
    out["sell_trigger"] = out["roc_59"] > sell_roc_limit

    selected_keys: set[tuple[object, str]] = set()
    rank_lookup: dict[tuple[object, str], int] = {}
    holdings: set[str] = set()
    for trade_date, day in out.groupby("trade_date", sort=True):
        buy_symbols = set(day.loc[day["buy_eligible"], "symbol"].astype(str))
        sell_symbols = set(day.loc[day["sell_trigger"], "symbol"].astype(str))
        candidates = (buy_symbols | holdings) - sell_symbols
        candidate_rows = day[day["symbol"].astype(str).isin(candidates)].sort_values(
            ["score", "symbol"], ascending=[False, True], na_position="last"
        )
        selected_rows = candidate_rows.head(top_k)
        holdings = set(selected_rows["symbol"].astype(str))
        for rank, (_, row) in enumerate(candidate_rows.iterrows(), start=1):
            rank_lookup[(trade_date, str(row["symbol"]))] = rank
        selected_keys.update(
            (trade_date, str(symbol)) for symbol in selected_rows["symbol"].astype(str)
        )
    out["selected"] = [
        (date, str(symbol)) in selected_keys
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    out["rank"] = [
        rank_lookup.get((date, str(symbol)), np.nan)
        for date, symbol in zip(out["trade_date"], out["symbol"])
    ]
    selected_counts = out.groupby("trade_date")["selected"].transform("sum")
    out["weight"] = np.where(out["selected"], 1.0 / selected_counts, 0.0)
    return out.sort_values(["trade_date", "rank", "symbol"], na_position="last").reset_index(drop=True)
