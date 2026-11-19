"""数据接入模块 (Task 1.1).

统一的数据源访问层：所有数据源适配器产出相同的统一 Schema
``trade_date, open, high, low, close, volume, amount``，
其余代码只依赖 `fetch(code, start, end, adjust)` 这一统一接口。
"""
from __future__ import annotations

from .base import (
    DataSourceAdapter,
    DataSourceConfigError,
    DataSourceError,
    InsufficientDataError,
)
from .akshare_adapter import AkshareAdapter
from .csv_adapter import CsvAdapter
from .factory import DataSourceFactory
from .schema import OUTPUT_COLUMNS
from .tushare_adapter import TushareAdapter

__all__ = [
    "DataSourceAdapter",
    "DataSourceError",
    "DataSourceConfigError",
    "InsufficientDataError",
    "AkshareAdapter",
    "CsvAdapter",
    "TushareAdapter",
    "DataSourceFactory",
    "OUTPUT_COLUMNS",
]