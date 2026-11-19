"""公共停牌工具 (Task 1.4).

提供与「停牌判定」和「T+1~T+3 持仓窗口」有关的可复用函数，供标签生成器
`data/labels.py` 统一调用（对应 Todo「统一调用 suspend_utils.py」）。

设计要点：
  * 所有函数基于 **时间升序** 的统一 Schema 日线（含 `is_suspend` 列）的
    **位置索引**工作：调用方传入 `df` 与当前样本行位置 `t`，函数只读取
    `t` 之后的交易日窗口（T+1/T+2/T+3），**不引入未来数据**。
  * `is_suspend` 列：若 `df` 已有（由 data.clean 生成）则直接使用；
    否则根据「OHLC 缺失/<=0 或 成交量<=0」即时生成，保证公共调用一致性。
  * 全部为 O(N) 逐行计算；测试覆盖 TC-LABEL-01/02 相关停牌边界。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

_PRICE_COLS: tuple = ("open", "high", "low", "close")
_PRICE_COL_LIST: list = list(_PRICE_COLS)

# 持仓窗口长度（T+1~T+3，共 3 个交易日）。
DEFAULT_HOLD_DAYS: int = 3


# --------------------------------------------------------------------------
# 停牌判定
# --------------------------------------------------------------------------
def ensure_is_suspend(df: pd.DataFrame) -> pd.Series:
    """返回与 df 对齐的 `is_suspend` 标记。

    若 df 已含 `is_suspend` 列则透传，否则按统一规则生成：
    OHLC 缺失/<=0（全列） 或 成交量<=0 视为停牌日。
    """
    if "is_suspend" in df.columns:
        return df["is_suspend"].astype(int)
    work = df
    price_missing = work[_PRICE_COL_LIST].isna().all(axis=1) | (
        work[_PRICE_COL_LIST].fillna(0) <= 0
    ).all(axis=1)
    vol_zero = work.get("volume", pd.Series(0, index=work.index)).fillna(0) <= 0
    return (price_missing | vol_zero).astype(int)


# --------------------------------------------------------------------------
# 持仓窗口访问（位置索引，禁用未来泄露）
# --------------------------------------------------------------------------
def _rows_in_window(df: pd.DataFrame, t: int, days: int = DEFAULT_HOLD_DAYS) -> Sequence[int]:
    """样本位置 t 的持仓窗口（T+1..T+days）行位置列表；越界部分忽略。"""
    n = len(df)
    end = min(t + days, n - 1)
    if t + 1 > end:
        return ()
    return tuple(range(t + 1, end + 1))


def hold_window_suspension_count(
    df: pd.DataFrame,
    t: int,
    days: int = DEFAULT_HOLD_DAYS,
    is_suspend: Optional[pd.Series] = None,
) -> int:
    """T+1~T+days 持仓窗口内停牌天数。越界行视为非停牌（窗口已到数据末尾）。"""
    susp = ensure_is_suspend(df) if is_suspend is None else is_suspend
    rows = _rows_in_window(df, t, days)
    return int(susp.iloc[list(rows)].sum()) if rows else 0


def next_open_price(
    df: pd.DataFrame,
    t: int,
    is_suspend: Optional[pd.Series] = None,
) -> Optional[float]:
    """T+1 日开盘价（真实买入基准价）。

    T+1 越界、该日停牌、或 open<=0/缺失 -> 返回 None（对应「次日停牌剔除」）。
    """
    n = len(df)
    if t + 1 >= n:
        return None
    susp = ensure_is_suspend(df) if is_suspend is None else is_suspend
    if int(susp.iloc[t + 1]) == 1:
        return None
    o = df["open"].iloc[t + 1]
    if o is None or not np.isfinite(o) or o <= 0:
        return None
    return float(o)


def hold_window_last_valid_close(
    df: pd.DataFrame,
    t: int,
    days: int = DEFAULT_HOLD_DAYS,
    is_suspend: Optional[pd.Series] = None,
) -> Optional[float]:
    """持仓窗口最后一个**有效交易日**收盘价（卖出价）。

    只取窗口内非停牌、close 有效（>0 且非 NaN）的最后一天的 close；
    若无任何有效交易日 -> None。
    """
    susp = ensure_is_suspend(df) if is_suspend is None else is_suspend
    rows = _rows_in_window(df, t, days)
    last_valid: Optional[float] = None
    for i in rows:
        if int(susp.iloc[i]) == 1:
            continue
        c = df["close"].iloc[i]
        if c is not None and np.isfinite(c) and c > 0:
            last_valid = float(c)
    return last_valid


def hold_window_return(
    df: pd.DataFrame,
    t: int,
    days: int = DEFAULT_HOLD_DAYS,
    is_suspend: Optional[pd.Series] = None,
) -> Optional[float]:
    """计算样本位置 t 的持仓累计收益率 = 最后有效收盘 / T+1开盘 - 1。

    T+1 开盘不可得（缺失/停牌/越界）或窗口无有效卖出价 -> None。
    """
    buy = next_open_price(df, t, is_suspend)
    if buy is None:
        return None
    sell = hold_window_last_valid_close(df, t, days, is_suspend)
    if sell is None:
        return None
    return sell / buy - 1.0