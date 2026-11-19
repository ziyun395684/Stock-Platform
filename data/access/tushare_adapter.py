"""Tushare daily-bar adapter (Task 1.1).

Tushare Pro is the requirement's recommended primary source (V1.2 §10.2), but it
requires an installed `tushare` package and a valid token. To keep the rest of
the pipeline working without those, this adapter raises a clear, actionable
`DataSourceConfigError` when the dependency or token is unavailable, and only
fetches when actually configured.

Per requirement "异常提示清晰", the error messages explain exactly what to do
(pip install tushare / set TUSHARE_TOKEN) rather than failing opaquely.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .base import DataSourceAdapter, DataSourceConfigError
from . import schema

logger = logging.getLogger(__name__)


class TushareAdapter(DataSourceAdapter):
    """Tushare Pro daily bars (dropped / clear-config-error when unavailable)."""

    source_name = schema.SOURCE_TUSHARE

    def __init__(self, token: str | None = None) -> None:
        super().__init__()
        # Fall back to the environment (e.g. .env key TUSHARE_TOKEN) if not passed.
        self._token = token or os.environ.get("TUSHARE_TOKEN") or ""
        self._pro = self._resolve_pro()

    # ------------------------------------------------------------------ deps
    def _resolve_pro(self):
        """Return a configured tushare pro API, failing clearly if unavailable."""
        try:
            import tushare as ts
        except ImportError as exc:
            raise DataSourceConfigError(
                "tushare 未安装，无法使用 Tushare 数据源。请执行 `pip install tushare`，"
                "或改用 akshare / 本地CSV 数据源。"
            ) from exc
        if not self._token:
            raise DataSourceConfigError(
                "未配置 TUSHARE_TOKEN，无法使用 Tushare 数据源。请在 .env 中设置 "
                "TUSHARE_TOKEN（tushare Pro 个人中心获取），或改用 akshare / 本地CSV。"
            )
        try:
            return ts.pro_api(self._token)
        except Exception as exc:  # noqa: BLE001 - token rejection surfaces clearly
            raise DataSourceConfigError(
                f"tushare 初始化失败，请检查 TUSHARE_TOKEN 是否正确: {exc}"
            ) from exc

    # ------------------------------------------------------------------ fetch
    def fetch(self, code, start, end, adjust="none"):
        start, end = self.validate_start_end(start, end)
        self.normalize_adjust(adjust)
        if self._pro is None:  # pragma: no cover - _resolve_pro raises instead
            raise DataSourceConfigError("tushare 数据源不可用，请先配置")
        native = self._pull(code, start, end)
        df = self.normalize_frame(native, code)
        self.check_min_years(df, code)
        logger.debug("tushare fetched %s %s rows [%s]", code, len(df), adjust)
        return df

    def _pull(self, code: str, start: str, end: str) -> pd.DataFrame:
        # ts.pro_bar returns columns: ts_code, trade_date, open, high, low,
        # close, pre_close, change, pct_chg, vol, amount. '-' adjust variants
        # are supported by pro_bar; `adj` maps none/qfq/hfq -> ""/"qfq"/"hfq".
        adj_map = {"none": "", "qfq": "qfq", "hfq": "hfq"}
        return self._pro.pro_bar(
            ts_code=code,
            adj=adj_map[self._adjust],
            start_date=start,
            end_date=end,
        )