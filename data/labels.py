"""双模式标签生成器 (Task 1.4).

`generate_labels(df, label_threshold, label_mode)` 把升序日线（含因子/停牌列）
转换为二分类标签样本，返回 `LabelResult`。

业务规则（需求 §2.2.2 / §2.2.4）：
    * 买入基准价 = T+1 开盘价 P_{T+1}^{open}；卖出价 = 持仓窗口最后有效交易日收盘价。
    * 累计收益率 R = 卖出价 / T+1开盘价 - 1；label = 1 当且仅当 R >= label_threshold
      （**恰好等于阈值归入正例**,≥条件），默认 +5%。
    * T+1 开盘缺失/停牌/<=0 -> 样本**剔除**。
    * T+1~T+3 停牌 >1 交易日 -> 样本**剔除**；恰好 1 日仍保留，卖出价取窗口内
      最后有效交易日的收盘价。

双模式（label_mode）：
    * "tolerant"（默认，需求约定）：持仓窗口停牌 <=1 日均保留；
    * "strict"（严格模式）：持仓窗口**出现任意停牌**即剔除，要求 T+1~T+3 完整成交。

样本对齐与 weight：
    * 生成结果的 sample 行与 y / weight / 特征行严格对齐（同一过滤后的 DataFrame
      切片），满足「无效样本 X/y/weight 同步过滤」。
    * weight 默认恒为 1.0；Task 1.5 序列切分阶段再按 low_vol 施加 0.5 权重。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .suspend_utils import (
    DEFAULT_HOLD_DAYS,
    ensure_is_suspend,
    hold_window_last_valid_close,
    hold_window_return,
    hold_window_suspension_count,
    next_open_price,
)

DEFAULT_LABEL_THRESHOLD: float = 0.05
VALID_LABEL_MODES: tuple = ("tolerant", "strict")


@dataclass
class LabelResult:
    """标签生成结果（行与 df 对其后的样本严格对齐）。"""

    df: pd.DataFrame                 # 仅含有效样本行的 DataFrame（含 label/y、return、weight、buy/sell）
    y: pd.Series                     # 标签二分类(0/1)，index 与 df 一致
    weight: pd.Series                # 样本权重，默认 1.0
    valid_index: pd.Index            # 有效样本相对原始 df 的位置(iloc 位置)
    label_mode: str = "tolerant"
    label_threshold: float = DEFAULT_LABEL_THRESHOLD
    stats: dict = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        if len(self.y) == 0:
            return 0.0
        return float(self.y.mean())

    def summary(self) -> dict:
        return {
            "n_valid": int(len(self.y)),
            "n_positive": int(self.y.sum()),
            "n_negative": int(len(self.y) - self.y.sum()),
            "positive_rate": round(self.positive_rate, 4),
            "label_mode": self.label_mode,
            "label_threshold": self.label_threshold,
        }


def _validate_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("generate_labels: df must be a non-empty DataFrame")
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"generate_labels: 缺少必要列 '{c}'")
    if "trade_date" in df.columns and not df["trade_date"].is_monotonic_increasing:
        # 时间升序校验（容忍重复日期的普通检查）。
        pass


def generate_labels(
    df: pd.DataFrame,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
    label_mode: str = "tolerant",
) -> LabelResult:
    """生成二分类标签样本。

    Args:
        df: 统一 Schema 升序日线 DataFrame（含 open/high/low/close，最好含
            is_suspend 与因子列；因子列用于对齐特征，不参与标签计算）。
        label_threshold: 正例阈值，默认 0.05（>= 归正例）。
        label_mode: "tolerant"（默认）或 "strict"。

    Returns:
        LabelResult：有效样本 df + y + weight，以及样本损耗统计 `stats`。
    """
    _validate_input(df)
    if label_threshold is None or not np.isfinite(label_threshold):
        raise ValueError("generate_labels: label_threshold 必须为有限数值")
    if label_mode not in VALID_LABEL_MODES:
        raise ValueError(
            f"generate_labels: label_mode 必须为 {VALID_LABEL_MODES}，收到 {label_mode!r}"
        )

    n = len(df)
    is_suspend = ensure_is_suspend(df)

    stats: dict = {
        "code": df.attrs.get("code", "?"),
        "total_rows_input": n,
        "label_threshold": float(label_threshold),
        "label_mode": label_mode,
        "hold_days": DEFAULT_HOLD_DAYS,
        "removed_reasons": [],
    }
    stats["dropped_next_open_missing"] = 0        # T+1 开盘缺失/停牌/<=0
    stats["dropped_suspension_gt1"] = 0           # 持仓停牌 >1
    stats["dropped_suspension_any_strict"] = 0    # strict 模式任意停牌
    stats["dropped_no_valid_sell"] = 0            # 窗口无有效卖出价

    keeps: list[bool] = []
    sample_rows: list[int] = []
    labels: list[int] = []
    rets: list[float] = []
    buy_prices: list[float] = []
    sell_prices: list[float] = []

    for t in range(n):
        if label_mode == "strict" and int(is_suspend.iloc[t]) == 1:
            # strict 模式：样本行本身停牌也剔除（要求完整 T+1~T+3 连续交易，
            # 且样本日需为有效交易日）。
            stats["dropped_suspension_any_strict"] += 1
            continue
        if label_mode == "tolerant" and int(is_suspend.iloc[t]) == 1:
            # tolerant 模式：样本日停牌视为无效样本（无可作为基准的交易日）。
            stats["dropped_next_open_missing"] += 1
            continue

        buy = next_open_price(df, t, is_suspend)
        if buy is None:
            stats["dropped_next_open_missing"] += 1
            continue

        susp_count = hold_window_suspension_count(df, t, is_suspend=is_suspend)
        if label_mode == "strict":
            if susp_count >= 1:
                stats["dropped_suspension_any_strict"] += 1
                continue
        else:
            if susp_count > 1:
                stats["dropped_suspension_gt1"] += 1
                continue

        sell = hold_window_last_valid_close(df, t, is_suspend=is_suspend)
        if sell is None:
            stats["dropped_no_valid_sell"] += 1
            continue

        ret = sell / buy - 1.0
        sample_rows.append(t)
        labels.append(1 if ret >= label_threshold else 0)
        rets.append(ret)
        buy_prices.append(buy)
        sell_prices.append(sell)
        keeps.append(True)

    # 构建与样本严格对齐的结果（X/y/weight 同步过滤：行均来自 sample_rows）。
    keep_mask = pd.Series(False, index=df.index)
    keep_mask.iloc[sample_rows] = True
    sample_df = df.iloc[sample_rows].copy()
    sample_df["label"] = pd.Series(labels, index=sample_df.index)
    sample_df["return"] = pd.Series(rets, index=sample_df.index)
    sample_df["buy_price"] = pd.Series(buy_prices, index=sample_df.index)
    sample_df["sell_price"] = pd.Series(sell_prices, index=sample_df.index)

    y = pd.Series(labels, index=sample_df.index, dtype=int)
    weight = pd.Series(1.0, index=sample_df.index, dtype=float)

    n_valid = len(sample_rows)
    stats["total_rows_valid"] = n_valid
    stats["total_rows_dropped"] = n - n_valid
    stats["n_positive"] = int(y.sum())
    stats["n_negative"] = n_valid - int(y.sum())
    stats["positive_rate"] = round(stats["n_positive"] / n_valid, 4) if n_valid else 0.0

    if stats["dropped_next_open_missing"]:
        stats["removed_reasons"].append(
            f"T+1开盘缺失/停牌/<=0({stats['dropped_next_open_missing']})"
        )
    if stats["dropped_suspension_gt1"]:
        stats["removed_reasons"].append(
            f"持仓停牌>1个交易日剔除({stats['dropped_suspension_gt1']})"
        )
    if stats["dropped_suspension_any_strict"]:
        stats["removed_reasons"].append(
            f"strict模式:持仓含停牌剔除({stats['dropped_suspension_any_strict']})"
        )
    if stats["dropped_no_valid_sell"]:
        stats["removed_reasons"].append(
            f"窗口无有效卖出价({stats['dropped_no_valid_sell']})"
        )

    return LabelResult(
        df=sample_df,
        y=y,
        weight=weight,
        valid_index=sample_df.index.copy(),
        label_mode=label_mode,
        label_threshold=float(label_threshold),
        stats=stats,
    )