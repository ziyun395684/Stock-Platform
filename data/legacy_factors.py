"""Causal features reproduced from the legacy ETF ranking strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd


LEGACY_FACTOR_COLUMNS = [
    "legacy_trend_score",
    "legacy_roc_5",
    "legacy_roc_10",
    "legacy_roc_18",
    "legacy_volume_ratio",
    "legacy_factor_score",
]


def add_legacy_factor_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the original strategy's causal factors to a symbol/date panel.

    The score mirrors the legacy expression::

        trend_score(close, 25) * 0.30
        + (roc(close, 5) + roc(close, 10)) * 0.18
        + ma(volume, 5) / ma(volume, 20)

    ``roc(close, 18)`` is retained separately because the legacy strategy uses
    it as a sell/exclusion rule.  All calculations are grouped by symbol and
    therefore cannot borrow observations from another ETF.
    """
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")

    out = panel.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.sort_values(["symbol", "trade_date"]).copy()
    grouped = out.groupby("symbol", sort=False)

    out["legacy_trend_score"] = grouped["close"].transform(
        lambda series: series.pct_change(25, fill_method=None)
    )
    out["legacy_roc_5"] = grouped["close"].transform(
        lambda series: series.pct_change(5, fill_method=None)
    )
    out["legacy_roc_10"] = grouped["close"].transform(
        lambda series: series.pct_change(10, fill_method=None)
    )
    out["legacy_roc_18"] = grouped["close"].transform(
        lambda series: series.pct_change(18, fill_method=None)
    )
    volume_ma5 = grouped["volume"].transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )
    volume_ma20 = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=1).mean()
    )
    out["legacy_volume_ratio"] = volume_ma5 / volume_ma20.replace(0, np.nan)

    component_columns = LEGACY_FACTOR_COLUMNS[:-1]
    out[component_columns] = out[component_columns].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    out["legacy_factor_score"] = (
        out["legacy_trend_score"] * 0.30
        + (out["legacy_roc_5"] + out["legacy_roc_10"]) * 0.18
        + out["legacy_volume_ratio"]
    )
    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
