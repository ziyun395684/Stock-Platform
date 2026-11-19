"""DataSourceFactory - runtime selection of a data-source adapter (Task 1.1).

Lets the pipeline select the source by name without touching concrete classes.
Per requirement Task 1.1 ③: a factory that maps source names (schema.SOURCE_*) to
adapter instances, with clear errors for unknown sources.
让流水线（Pipeline）仅通过名称来选择数据源，而不需要直接接触或依赖具体的实现类
"""
from __future__ import annotations

import logging
from typing import Type

from .akshare_adapter import AkshareAdapter
from .base import DataSourceConfigError, DataSourceError
from .csv_adapter import CsvAdapter
from .schema import (
    SOURCE_AKSHARE,
    SOURCE_CSV,
    SOURCE_TUSHARE,
    SOURCE_YAHOO,
)
from .tushare_adapter import TushareAdapter

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """Builds a concrete DataSourceAdapter from a source name + kwargs."""

    _REGISTRY: dict[str, Type] = {
        SOURCE_AKSHARE: AkshareAdapter,
        SOURCE_TUSHARE: TushareAdapter,
        SOURCE_CSV: CsvAdapter,
        # SOURCE_YAHOO: not implemented in M1 - reported clearly if requested.
    }

    @classmethod
    def create(cls, source: str, **kwargs):
        """Return a configured adapter for ``source`` (see schema.SOURCE_*)."""
        if not source:
            raise DataSourceConfigError("未指定数据源 source")
        source = source.strip().lower()
        adapter_cls = cls._REGISTRY.get(source)
        if adapter_cls is None:
            known = ", ".join(sorted(cls._REGISTRY))
            if source == SOURCE_YAHOO:
                raise DataSourceConfigError(
                    "Yahoo 数据源尚未实现（M1 未含 SOURCE_YAHOO）。"
                    f"可用数据源: {known}"
                )
            raise DataSourceConfigError(
                f"未知数据源 {source!r}，可用数据源: {known}"
            )
        try:
            return adapter_cls(**kwargs)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - construction failure surfaces clearly
            raise DataSourceConfigError(
                f"数据源 {source!r} 初始化失败: {exc}"
            ) from exc

    @classmethod
    def create_csv(cls, filepath: str) -> CsvAdapter:
        """Shorthand: build the local-CSV adapter."""
        return cls.create(SOURCE_CSV, filepath=filepath)

    @classmethod
    def create_akshare(cls, category: str = "stock") -> AkshareAdapter:
        """Shorthand: build the AkShare adapter for a category."""
        return cls.create(SOURCE_AKSHARE, category=category)