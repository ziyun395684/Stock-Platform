"""数据清洗组件 (Task 1.2).

`clean_data()` 对统一 Schema 的日线 DataFrame 做清洗，返回 `(df, stats)`：

    1. 停牌标记 `is_suspend`：OHLC 缺失/为空 或 成交量为 0 的交易日视为停牌。
    2. 异常值过滤：价格 <= 0，或日涨跌幅（收盘价环比）> 100% 的异常交易日剔除。
    3. 缺失前向填充：非停牌行的 OHLCV 缺失用前向填充补齐。
    4. 连续缺失丢弃：连续缺失/停牌段长度 >= 总行数 20% 的整段剔除。

`stats` 为清洗统计 dict，逐项记录剔除原因与数量（对应验收：输出清洗日志，记录剔除原因）。
输入要求：统一 Schema 的 DataFrame（trade_date 升序）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .access.schema import OUTPUT_COLUMNS

# 连续缺失/停牌段达到总行数的该比例，则该段剔除 (Todo Task 1.2).
DEFAULT_MISSING_THRESHOLD: float = 0.20

_PRICE_COLS: tuple = ("open", "high", "low", "close")
# pandas does NOT treat a tuple as a multi-column key - must index with a list.
_PRICE_COL_LIST = list(_PRICE_COLS)
_ALL_NUM_COLS = _PRICE_COL_LIST + ["volume", "amount"]


def clean_data(
    df: pd.DataFrame,
    code: Optional[str] = None,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Clean a unified-schema daily DataFrame.

    Returns:
        (df_clean, stats): df_clean has schema.OUTPUT_COLUMNS + 'is_suspend'.
        stats logs each removal reason (清洗统计 / 清洗日志).
    """
    stats: dict = {
        "code": code or "?",
        "total_rows_input": int(len(df)),
        "removed_reasons": [],
    }

    work = df.copy()
    if work.empty:
        stats.update(_empty_stats(work))
        return work, stats

    # ``amount`` is optional for local ETF CSVs.  Keep the unified schema by
    # deriving it from close * volume before numeric cleaning begins.
    if "amount" not in work.columns:
        close = pd.to_numeric(work.get("close", 0), errors="coerce").fillna(0.0)
        volume = pd.to_numeric(work.get("volume", 0), errors="coerce").fillna(0.0)
        work["amount"] = close * volume

    # ---- (1) is_suspend marker: missing OHLC *or* zero volume = 停牌日.
    price_missing = work[_PRICE_COL_LIST].isna().all(axis=1) | (
        work[_PRICE_COL_LIST].fillna(0) <= 0
    ).all(axis=1)
    vol_zero = work.get("volume", pd.Series(0, index=work.index)).fillna(0) <= 0
    is_suspend = (price_missing | vol_zero).astype(int)
    work["is_suspend"] = is_suspend

    # ---- (2) price anomaly filter.
    rm_anomaly = _price_anomaly_mask(work)
    dropped_anomaly = int(rm_anomaly.sum())
    stats["rows_dropped_price_anomaly"] = dropped_anomaly
    if dropped_anomaly:
        stats["removed_reasons"].append(
            f"价格异常({dropped_anomaly})：price<=0 或 |涨跌幅|>100%"
        )
    work = work[~rm_anomaly].copy()
    if work.empty:
        stats.update(_empty_stats(work))
        return work, stats

    # ---- (3) forward-fill OHLCV missing (only on rows kept).
    before_fill = work[_ALL_NUM_COLS].isna().sum().sum()
    for col in _ALL_NUM_COLS:
        work[col] = work[col].ffill()
    stats["filled_forward"] = int(before_fill)

    # ---- (4) consecutive missing/suspend runs >= threshold -> drop whole runs.
    work, dropped_consec = _drop_long_missing_runs(
        work, code=code, threshold=missing_threshold
    )
    stats["rows_dropped_consecutive_missing"] = int(dropped_consec)
    if dropped_consec:
        stats["removed_reasons"].append(
            f"连续缺失/停牌段(>=20%)剔除({dropped_consec})"
        )

    # Post-drop NaN caused by leading consecutive-missing segments: forward-fill
    # any remaining gaps of pure missing (no suspension semantics left).
    for col in _ALL_NUM_COLS:
        work[col] = work[col].ffill()

    stats["total_rows_output"] = int(len(work))
    return work.reset_index(drop=True), stats


# ------------------------------------------------------------------ helpers
def _price_anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """Mask of rows to *drop*: price<=0 or |pct_change|>100%."""
    drop = pd.Series(False, index=df.index)
    for c in _PRICE_COLS:
        drop = drop | (df[c] <= 0)
    close = df["close"]
    pct = close.pct_change(fill_method=None).abs()
    drop = drop | (pct > 1.0)  # 涨跌幅 > 100%
    return drop.fillna(False)


def _drop_long_missing_runs(
    df: pd.DataFrame, code, threshold: float
) -> tuple[pd.DataFrame, int]:
    """Drop runs of rows whose close is missing or marked is_suspend, when the
    run length >= threshold * total rows. Returns (df, n_dropped)."""
    if df.empty:
        return df, 0
    total = len(df)
    n_dropped = 0
    keep_idx = pd.Series(True, index=df.index)
    # a row is "missing/suspend" if close is NaN or is_suspend == 1
    missing = df["close"].isna() | (df["is_suspend"] == 1)
    missing = missing.astype(int)

    run_start = None
    run_len = 0
    for idx, m in missing.items():
        if m == 1:
            if run_start is None:
                run_start = idx
            run_len += 1
        else:
            if run_start is not None:
                if run_len / total >= threshold:
                    keep_idx.loc[run_start:idx] = False
                    n_dropped += run_len
                run_start = None
                run_len = 0
    # trailing run
    if run_start is not None and run_len / total >= threshold:
        keep_idx.loc[run_start:] = False
        n_dropped += run_len
    return df[keep_idx].copy(), n_dropped


def _empty_stats(df: pd.DataFrame) -> dict:
    return {
        "rows_dropped_price_anomaly": 0,
        "filled_forward": 0,
        "rows_dropped_consecutive_missing": 0,
        "total_rows_output": 0,
        "removed_reasons": ["清洗后无有效数据"],
    }
