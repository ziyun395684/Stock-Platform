"""Multi-ETF panel alignment and causal feature construction."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def load_universe_frames(
    symbols: Sequence[str],
    data_dir: str = "quotes",
    start: str = "1900-01-01",
    end: str = "2100-01-01",
    adjust: str = "none",
    enforce_min_years: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load one normalized CSV frame per configured ETF symbol."""
    from .access.factory import DataSourceFactory

    if not symbols:
        raise ValueError("symbols 不能为空")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol}.csv"
        adapter = DataSourceFactory.create_csv(str(path))
        frames[str(symbol)] = adapter.fetch(
            str(symbol), start, end, adjust,
            enforce_min_years=enforce_min_years,
        )
    return frames


def align_etf_frames(
    frames: Mapping[str, pd.DataFrame],
    date_mode: str = "common",
) -> pd.DataFrame:
    """Return a date-aligned panel while preserving symbol/date identity."""
    if date_mode not in {"common", "union"}:
        raise ValueError("date_mode 必须是 common 或 union")
    if not frames:
        raise ValueError("frames 不能为空")
    normalized = []
    common_dates = None
    for symbol, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        item = frame.copy()
        if "trade_date" not in item.columns:
            raise ValueError(f"{symbol}: 缺少 trade_date")
        item["trade_date"] = pd.to_numeric(item["trade_date"], errors="raise").astype(int)
        item["symbol"] = symbol
        dates = set(item["trade_date"].tolist())
        common_dates = dates if common_dates is None else common_dates & dates
        normalized.append(item)
    if not normalized or not common_dates:
        raise ValueError("ETF 数据没有公共交易日")
    out = pd.concat(normalized, ignore_index=True)
    if date_mode == "common":
        out = out[out["trade_date"].isin(common_dates)].copy()
    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _rank(group: pd.Series) -> pd.Series:
    return group.rank(method="average", pct=True).astype(float)


def build_panel_features(
    panel: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    extended_features: bool = False,
) -> pd.DataFrame:
    """Add causal per-symbol and cross-sectional features to a panel.

    The extended multi-horizon feature set is opt-in so the production/default
    input remains comparable with the validated legacy feature set.
    """
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    out = panel.sort_values(["symbol", "trade_date"]).copy()
    grouped = out.groupby("symbol", sort=False)
    out["ret_1"] = grouped["close"].pct_change(fill_method=None)
    out["ret_5"] = grouped["close"].pct_change(5, fill_method=None)
    out["ret_20"] = grouped["close"].pct_change(20, fill_method=None)
    daily_ret = out["ret_1"]
    out["volatility_20"] = daily_ret.groupby(out["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    vol5 = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    vol20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["volume_ratio"] = vol5 / vol20.replace(0, np.nan)
    rank_columns = ["ret_1", "ret_5", "ret_20", "volatility_20", "volume_ratio"]
    if extended_features:
        out["ret_10"] = grouped["close"].pct_change(10, fill_method=None)
        out["ret_60"] = grouped["close"].pct_change(60, fill_method=None)
        ma20 = grouped["close"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        out["ma_gap_20"] = out["close"] / ma20.replace(0, np.nan) - 1.0
        rolling_high20 = grouped["close"].transform(
            lambda s: s.rolling(20, min_periods=5).max()
        )
        out["drawdown_20"] = out["close"] / rolling_high20.replace(0, np.nan) - 1.0
        out["volatility_5"] = daily_ret.groupby(out["symbol"]).transform(
            lambda s: s.rolling(5, min_periods=3).std()
        )
        rank_columns[2:2] = ["ret_10"]
        rank_columns.insert(4, "ret_60")
        rank_columns.extend(["ma_gap_20", "drawdown_20", "volatility_5"])
    for col in rank_columns:
        out[f"{col}_rank"] = out.groupby("trade_date")[col].transform(_rank)
    if benchmark is not None:
        if not {"trade_date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark 缺少 trade_date 或 close")
        bench = benchmark.sort_values("trade_date").copy()
        bench["benchmark_return_1"] = pd.to_numeric(
            bench["close"], errors="coerce"
        ).pct_change(fill_method=None)
        bench_returns = bench.set_index("trade_date")["benchmark_return_1"]
        out["benchmark_return_1"] = out["trade_date"].map(bench_returns)
        out["relative_benchmark_return"] = (
            out["ret_1"] - out["benchmark_return_1"]
        )
        if extended_features:
            bench["benchmark_return_20"] = pd.to_numeric(
                bench["close"], errors="coerce"
            ).pct_change(20, fill_method=None)
            out["benchmark_return_20"] = out["trade_date"].map(
                bench.set_index("trade_date")["benchmark_return_20"]
            )
            out["relative_benchmark_return_20"] = (
                out["ret_20"] - out["benchmark_return_20"]
            )
    from .legacy_factors import add_legacy_factor_features

    out = add_legacy_factor_features(out)
    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
