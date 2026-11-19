"""Local CSV data-source adapter (Task 1.1).

Reads historical OHLCV daily bars from a local CSV file and normalizes them
into the unified ``OUTPUT_COLUMNS`` schema.
从本地 CSV 文件中读取历史的 OHLCV 日线数据，并将其标准化为统一的 OUTPUT_COLUMNS 格式（或列名结构）

Contracts (requirement Task 1.1 / §5.1):
    * unified output columns: trade_date, open, high, low, close, volume, amount
    * output sorted ascending by trade_date (row order in the file is arbitrary)
    * if the covered calendar span is shorter than MIN_YEARS lines -> error
    * clear, actionable error messages

File conventions accepted:
    * column aliases resolved through ``_COLUMN_ALIASES`` (supports ``date``/``trade_date``,
      Chinese names, etc.); extra columns (e.g. ``adj_factor``, ``symbol``) are ignored.
    * ``amount`` may be absent: it is derived as ``close * volume`` when possible,
      otherwise padded with 0 so every row still carries all 7 unified columns.
    * left/right padding rows outside the requested [start, end] window are dropped.
"""
from __future__ import annotations

import os

from .base import (
    DataSourceAdapter,
    DataSourceConfigError,
)
from .schema import SOURCE_CSV


class CsvAdapter(DataSourceAdapter):
    """Reads unified daily bars from a local CSV file."""

    source_name: str = SOURCE_CSV

    def __init__(self, filepath: str) -> None:
        super().__init__()
        if not filepath or not isinstance(filepath, str):
            raise DataSourceConfigError(
                f"[csv] filepath must be a non-empty string, got {filepath!r}"
            )
        if not os.path.isfile(filepath):
            raise DataSourceConfigError(f"[csv] data file not found: {filepath}")
        self.filepath = filepath

    # ------------------------------------------------------------------ fetch
    def fetch(self, code, start, end, adjust="none", enforce_min_years=True):
        """Load the CSV, filter to [start, end] and normalize.

        Any ``adjust`` is accepted for API parity but a CSV has no adjustment
        resolution built in; normalized prices are taken verbatim from the file.
        """
        start, end = self.validate_start_end(start, end)
        self.normalize_adjust(adjust)

        df = self._read_csv()
        df = self._filter_window(df, start, end)
        df = self._ensure_amount(df)
        df = self.normalize_frame(df, code)
        if enforce_min_years:
            self.check_min_years(df, code)
        return df

    # ------------------------------------------------------------- helpers
    def _read_csv(self):
        """Read the CSV into a DataFrame, surfacing a clear error on failure."""
        import pandas as pd

        try:
            df = pd.read_csv(self.filepath)
        except Exception as exc:  # noqa: BLE001 - surface clear message
            raise DataSourceConfigError(
                f"[csv] failed to read {self.filepath}: {exc}"
            ) from exc
        if df is None or df.empty:
            raise DataSourceConfigError(
                f"[csv] file is empty or has no rows: {self.filepath}"
            )
        return df

    def _filter_window(self, df, start, end):
        """Drop rows whose trade_date falls outside [start, end).

        Compares on the normalized YYYYMMDD string form so it works whether the
        date column is ``date`` or ``trade_date`` (native or unified), and int,
        str or datetime. If no date column is found, no filtering happens
        (normalize_frame will raise a clear schema error later).
        """
        import pandas as pd

        if df.empty:
            return df
        date_col = "trade_date" if "trade_date" in df.columns else (
            "date" if "date" in df.columns else None
        )
        if date_col is None:
            return df

        lo = str(start).replace("-", "")
        hi = str(end).replace("-", "")

        def _key(row_date):
            text = str(pd.to_datetime(row_date)).split(" ")[0].replace("-", "")
            return text.zfill(8)

        try:
            mask = [lo <= _key(x) <= hi for x in df[date_col].tolist()]
        except Exception:  # noqa: BLE001 - nothing comparable -> keep everything
            return df

        return df[mask].copy()

    def _ensure_amount(self, df):
        """Guarantee an ``amount`` column.

        Prefers a native ``amount`` alias; otherwise derives ``close * volume``;
        elsewhere pads with 0 (all 7 unified columns still present afterwards).
        """
        import pandas as pd

        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            return df

        if "close" in df.columns and "volume" in df.columns:
            df["amount"] = (
                pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            )
            return df

        df["amount"] = 0.0
        return df
