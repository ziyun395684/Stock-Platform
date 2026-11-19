"""DataSourceAdapter abstract base and shared validations.

Task 1.1 数据接入模块. Every concrete adapter implements `fetch(code, start, end,
adjust)` and returns a unified DataFrame whose columns are exactly
`schema.OUTPUT_COLUMNS` ordered as `trade_date,open,high,low,close,volume,amount`.
"""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Optional

import pandas as pd

from . import schema


class DataSourceError(Exception):
    """Base error for any data-source problem."""


class InsufficientDataError(DataSourceError):
    """Raised when fetched history is shorter than the required MIN_YEARS."""


class DataSourceConfigError(DataSourceError):
    """Raised when the adapter is not configured (e.g. missing token/back-end)."""


class DataSourceAdapter(abc.ABC):
    """Abstract base class for unified daily-bar fetch across sources.

    Type annotation for `adjust`:
        'none' -> unadjusted prices,
        'hfq'  -> 前复权 (forward adjusted),
        'qfq'  -> 后复权 (backward adjusted).
    """

    source_name: str = "abstract"

    def __init__(self) -> None:
        self._normalize_adjust("none")
        self._min_years = schema.MIN_YEARS

    # ------------------------------------------------------------------ API
    @abc.abstractmethod
    def fetch(
        self,
        code: str,
        start: str,
        end: str,
        adjust: str = "none",
    ) -> pd.DataFrame:
        """Fetch daily bars for `code` in window [start, end] and normalize.

        Returns a DataFrame with columns exactly `schema.OUTPUT_COLUMNS` and a
        dtype-cast, index-free, time-ascending ordering. Raises
        InsufficientDataError when fewer than MIN_YEARS of bars are available.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ utils
    def validate_start_end(self, start: str, end: str) -> tuple[str, str]:
        """Validate and canonicalize start/end dates (YYYYMMDD or YYYY-MM-DD)."""
        for d in (start, end):
            try:
                dt = datetime.strptime(d, "%Y%m%d")
            except ValueError:
                try:
                    datetime.strptime(d, "%Y-%m-%d")
                except ValueError as exc:
                    raise DataSourceError(
                        f"非法日期 {d!r}，期望 YYYYMMDD 或 YYYY-MM-DD 格式"
                    ) from exc
                else:
                    dt = datetime.strptime(d, "%Y-%m-%d")
            raw = dt.strftime("%Y%m%d")
            if d != raw and d != dt.strftime("%Y-%m-%d"):
                raise DataSourceError(f"非法日期 {d!r}")
        if start > end:
            raise DataSourceError(f"start({start}) 不能晚于 end({end})")
        return start, end

    def normalize_adjust(self, adjust: str) -> str:
        """Validate the adjustment mode and record it."""
        return self._normalize_adjust(adjust)

    def _normalize_adjust(self, adjust: str) -> str:
        if adjust not in ("none", "hfq", "qfq"):
            raise DataSourceError(
                f"非法复权参数 {adjust!r}，可选: none/hfq/qfq"
            )
        self._adjust = adjust
        return adjust

    def check_min_years(self, df: pd.DataFrame, code: str) -> None:
        """Reject histories shorter than MIN_YEARS with a clear message.

        The check is calendar-span-based (requirement 5.1: 至少5年), so a dense
        recent series that starts less than ~MIN_YEARS back from its last bar is
        rejected even if it has many rows (e.g. minute data aggregated up).
        """
        if df.empty:
            raise InsufficientDataError(f"[{code}] 无任何数据，数据不足5年，拒绝训练")
        # trade_date is int YYYYMMDD; parse as a string with explicit format so
        # ints are NOT reinterpreted as epoch-nanoseconds (pandas default).
        dates = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        first = dates.min()
        last = dates.max()
        span_years = (last - first).days / 365.25
        if span_years + 1e-9 < self._min_years:
            raise InsufficientDataError(
                f"[{code}] 数据跨度 {span_years:.2f} 年 < 要求的 {self._min_years} "
                f"年（不足5年，拒绝训练）"
            )

    @staticmethod
    def normalize_frame(df: pd.DataFrame, code: str) -> pd.DataFrame:
        """Map native column names, subset to unified columns, cast types, sort.

        Rows are sorted ascending by trade_date so downstream windows/labels are
        time-consistent regardless of source ordering.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=schema.OUTPUT_COLUMNS)
        renamed = df.rename(columns=schema._COLUMN_ALIASES)
        missing = [c for c in schema.OUTPUT_COLUMNS if c not in renamed.columns]
        if missing:
            raise DataSourceError(
                f"[{code}] 数据源缺少必需字段 {missing}，无法输出统一Schema"
            )
        out = renamed[schema.OUTPUT_COLUMNS].copy()
        # trade_date -> int YYYYMMDD; tolerate both YYYYMMDD and YYYY-MM-DD forms.
        out["trade_date"] = (
            out["trade_date"].astype(str).str.replace("-", "", regex=False)
            .str.split(" ").str[0].astype(int)
        )
        for c in ("open", "high", "low", "close", "volume", "amount"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.sort_values("trade_date").reset_index(drop=True)
        return out