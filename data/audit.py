"""Quote data quality and adjustment-convention audit helpers."""
from __future__ import annotations

from typing import Mapping

import pandas as pd


def audit_quote_frame(
    frame: pd.DataFrame,
    symbol: str | None = None,
    adjust_type: str = "none",
) -> dict:
    """Return deterministic date, price, suspension, and schema audit fields."""
    required = {"trade_date", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required - set(frame.columns))
    dates = pd.to_numeric(frame.get("trade_date", pd.Series(dtype="float64")), errors="coerce")
    price_columns = [column for column in ("open", "high", "low", "close") if column in frame]
    price_invalid = (
        frame[price_columns].isna().any(axis=1) | (frame[price_columns] <= 0).any(axis=1)
        if price_columns
        else pd.Series(False, index=frame.index)
    )
    volume = pd.to_numeric(frame.get("volume", pd.Series(0, index=frame.index)), errors="coerce")
    suspension = price_invalid | volume.fillna(0).le(0)
    return {
        "symbol": symbol,
        "adjust_type": adjust_type,
        "rows": int(len(frame)),
        "min_trade_date": int(dates.min()) if dates.notna().any() else None,
        "max_trade_date": int(dates.max()) if dates.notna().any() else None,
        "duplicate_trade_dates": int(dates.duplicated().sum()),
        "date_monotonic": bool(dates.dropna().is_monotonic_increasing),
        "missing_columns": missing_columns,
        "amount_missing_in_input": "amount" not in frame.columns,
        "suspension_rows": int(suspension.sum()),
        "invalid_price_rows": int(price_invalid.sum()),
        "zero_volume_rows": int(volume.fillna(0).le(0).sum()),
    }


def audit_universe_frames(
    frames: Mapping[str, pd.DataFrame],
    adjust_type: str = "none",
) -> dict:
    """Audit each ETF and report the common trading-date intersection."""
    audits = [audit_quote_frame(frame, str(symbol), adjust_type) for symbol, frame in frames.items()]
    date_sets = [
        set(pd.to_numeric(frame.get("trade_date", pd.Series(dtype="float64")), errors="coerce").dropna())
        for frame in frames.values()
    ]
    common = set.intersection(*date_sets) if date_sets else set()
    return {
        "symbols": [str(symbol) for symbol in frames],
        "common_trade_dates": int(len(common)),
        "per_symbol": audits,
    }
