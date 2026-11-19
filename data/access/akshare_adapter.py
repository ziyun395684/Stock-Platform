"""Akshare daily-bar adapter.

Task 1.1 数据接入模块. Wraps AkShare's eastmoney endpoints so the rest of the
pipeline talks only to the unified schema.

Routing (category-specific, eastmoney back-end):
  stock -> ak.stock_zh_a_hist   (A股个股)
  etf   -> ak.fund_etf_hist_em  (ETF基金)
  index -> ak.index_zh_a_hist   (指数, no adjust/复权)
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base import DataSourceAdapter, DataSourceError
from . import schema

logger = logging.getLogger(__name__)

_ADJUST_TO_EM = {"none": "", "hfq": "hfq", "qfq": "qfq"}


class AkshareAdapter(DataSourceAdapter):
    """AkShare data source (eastmoney back-end)."""

    source_name = schema.SOURCE_AKSHARE

    def __init__(self, category: str = "stock") -> None:
        super().__init__()
        if category not in ("stock", "etf", "index"):
            raise DataSourceError(
                f"Akshare 非法类型 {category!r}，可选 stock/etf/index"
            )
        self.category = category

    # ------------------------------------------------------------------ fetch
    def fetch(
        self,
        code: str,
        start: str,
        end: str,
        adjust: str = "none",
    ) -> pd.DataFrame:
        start, end = self.validate_start_end(start, end)
        self.normalize_adjust(adjust)
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DataSourceError(
                "未安装 akshare，请先 `pip install akshare` 或改用本地CSV适配器"
            ) from exc

        native = self._pull(ak, code, start, end)
        df = self.normalize_frame(native, code)
        # eastmoney already yields <5yr in-progress - still enforce the gate.
        self.check_min_years(df, code)
        logger.debug("akshare fetched %s %s rows [%s]", code, len(df), adjust)
        return df

    def _pull(self, ak, code: str, start: str, end: str) -> pd.DataFrame:
        if self.category == "stock":
            return ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end,
                adjust=_ADJUST_TO_EM[self._adjust],
            )
        if self.category == "etf":
            return ak.fund_etf_hist_em(
                symbol=code, period="daily",
                start_date=start, end_date=end,
                adjust=_ADJUST_TO_EM[self._adjust],
            )
        # index: no adjust/复权 concept in eastmoney
        return ak.index_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end,
        )