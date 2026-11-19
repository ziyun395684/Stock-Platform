"""Data ingestion shared schema constants.

Defines the unified output schema that every DataSourceAdapter must produce.
Per requirement //8 / Task 1.1: output fields =
    trade_date, open, high, low, close, volume, amount
"""
from __future__ import annotations

from typing import Final

# Unified output columns, in this exact order (requirement Task 1.1).
OUTPUT_COLUMNS: Final[list[str]] = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

# Column name mapping: source-native names -> unified names. Adapters normalize
# their native DataFrames through these aliases.
_COLUMN_ALIASES: Final[dict[str, str]] = {
    # akshare / common
    "日期": "trade_date",
    "date": "trade_date",
    "trade_date": "trade_date",
    "开盘": "open",
    "open": "open",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "收盘": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "vol": "volume",
    "成交额": "amount",
    "amount": "amount",
}

# Minimum amount of history required to train, in calendar years (requirement 5.1).
MIN_YEARS: Final[int] = 5

# For a stock/ETF, roughly 250 trading days per year.
TRADING_DAYS_PER_YEAR: Final[float] = 250.0

# Class names for source types accepted by DataSourceFactory.
SOURCE_AKSHARE: Final[str] = "akshare"
SOURCE_TUSHARE: Final[str] = "tushare"
SOURCE_CSV: Final[str] = "csv"
SOURCE_YAHOO: Final[str] = "yahoo"