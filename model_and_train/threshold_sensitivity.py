"""Task 3.2b 阈值灵敏度分析.

对应 Todo Task3.2b / 需求 §5.3、§8.5：

    * 对 `infer_threshold` 在 0.3-0.7 范围扫描（步长 0.05），输出每个
      阈值下：信号数量、持仓胜率、盈亏比、F1、净值曲线。
    * 支持**最小信号间隔**（如连续 3 天信号只取第一次），降低交易频率
      与摩擦成本（需求§8.5）。
    * 输出为 JSON 安全 dict（经 `to_json_safe`），每行格式与 Task3.2
      `threshold_scan` 对齐；净值曲线数据供 Task4.1 绘图。

口径（与标签/回测完全一致）：
    * 信号日 T：样本日；持仓收益 ret_actual 为该样本的业务口径累计收益
      （T+1 开盘买入 -> 持仓窗口最后有效交易日收盘卖出）。
    * 持仓胜率（Hit Rate，需求§10.3）= 预测为上涨事件且实际累计收益
      ≥ label_threshold 的样本 / 所有预测为上涨事件的样本。
    * 盈亏比（需求§10.3）= 命中信号的平均收益 / 未命中信号的平均亏损。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .evaluation import classification_metrics, to_json_safe

logger = logging.getLogger(__name__)

#: 默认阈值扫描范围 0.3-0.7，步长 0.05（需求§8.5）。
DEFAULT_SCAN_START: float = 0.30
DEFAULT_SCAN_END: float = 0.70
DEFAULT_SCAN_STEP: float = 0.05


def _scan_thresholds(start: float, end: float, step: float) -> List[float]:
    return [round(t, 4) for t in np.arange(start, end + 1e-9, step)]


def _apply_min_signal_interval(
    dates: np.ndarray,
    signal: np.ndarray,
    min_interval: int,
) -> np.ndarray:
    """最小信号间隔：连续 min_interval 个交易日内只保留第一个信号。

    按日期升序处理；返回过滤后的信号掩码。min_interval<=0 时原样返回。
    """
    if min_interval is None or int(min_interval) <= 0:
        return signal
    interval = int(min_interval)
    out = np.zeros_like(signal, dtype=bool)
    last_signal_pos: Optional[int] = None
    for i in range(len(dates)):
        if not signal[i]:
            continue
        if last_signal_pos is None or (i - last_signal_pos) > interval:
            out[i] = True
            last_signal_pos = i
    return out


def _hit_rate_and_plr(
    signal: np.ndarray,
    y_true: np.ndarray,
    ret_actual: Optional[np.ndarray],
    label_threshold: float,
) -> Dict[str, Optional[float]]:
    """持仓胜率与盈亏比。

    ret_actual 缺失时只用 label 计算胜率（盈亏比返回 None）。
    """
    n_sig = int(np.sum(signal))
    if n_sig == 0:
        return {"hit_rate": None, "profit_loss_ratio": None, "n_signals": 0}

    if ret_actual is not None:
        rets = ret_actual[signal]
        hit = np.mean(rets >= label_threshold) if len(rets) else None
        winners = rets[rets > 0]
        losers = rets[rets < 0]
        plr: Optional[float] = None
        if len(winners) and len(losers):
            plr = float(np.mean(winners) / abs(np.mean(losers)))
    else:
        hits = y_true[signal]
        hit = float(np.mean(hits == 1)) if len(hits) else None
        plr = None

    return {
        "hit_rate": None if hit is None else round(float(hit), 4),
        "profit_loss_ratio": None if plr is None else round(plr, 4),
        "n_signals": n_sig,
    }


def _nav_curve(
    dates: np.ndarray,
    signal: np.ndarray,
    ret_actual: np.ndarray,
    cost_rate: float,
) -> Dict[str, list]:
    """按信号顺序构造净值曲线（扣摩擦成本后）。

    每次信号按业务口径持有（ret_actual 已含停牌顺延逻辑），
    净值 = cumprod(1 + ret_actual - cost_rate)，起点 1.0。
    返回 {"dates": [...], "nav": [...]}。
    """
    sig_dates = dates[signal]
    sig_rets = ret_actual[signal]
    net_rets = sig_rets - cost_rate
    nav = np.cumprod(1.0 + net_rets)
    return {
        "dates": [str(d) for d in sig_dates.tolist()],
        "nav": [round(float(v), 6) for v in nav.tolist()],
    }


def threshold_sensitivity_analysis(
    dates: Sequence,
    y_true: Sequence[int],
    y_score: Sequence[float],
    ret_actual: Optional[Sequence[float]] = None,
    *,
    scan_start: float = DEFAULT_SCAN_START,
    scan_end: float = DEFAULT_SCAN_END,
    scan_step: float = DEFAULT_SCAN_STEP,
    min_signal_interval: int = 0,
    cost_rate: float = 0.003,
    label_threshold: float = 0.05,
) -> Dict[str, object]:
    """阈值灵敏度分析（Task3.2b）。

    Args:
        dates: 样本日期（升序，与 y_true/y_score 对齐）。
        y_true: 真实标签 0/1。
        y_score: 集成（或子模型）预测概率。
        ret_actual: 各样本业务口径累计收益（与标签生成同口径）；
            缺失时盈亏比/净值曲线返回 None/空。
        scan_start/scan_end/scan_step: 阈值扫描范围（默认 0.3-0.7 / 0.05）。
        min_signal_interval: 最小信号间隔（交易日数，0=不限制）。
        cost_rate: 单次交易摩擦成本（买入+卖出合计，需求§8.6 建议 0.3%-0.5%）。
        label_threshold: 上涨事件阈值（默认 +5%）。

    Returns:
        dict（JSON 安全）：
            thresholds / rows（每行含 threshold、n_signals、hit_rate、
            profit_loss_ratio、f1、auc、nav 等）/ 汇总建议。
    """
    dates = np.asarray(list(dates))
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    ret_actual = None if ret_actual is None else np.asarray(ret_actual, dtype="float64")
    if not (len(dates) == len(y_true) == len(y_score)):
        raise ValueError("dates / y_true / y_score 长度必须一致")
    if ret_actual is not None and len(ret_actual) != len(dates):
        raise ValueError("ret_actual 长度必须与 dates 一致")

    thresholds = _scan_thresholds(scan_start, scan_end, scan_step)
    rows: List[Dict] = []
    for t in thresholds:
        raw_signal = (y_score >= t)
        signal = _apply_min_signal_interval(dates, raw_signal, min_signal_interval)
        metrics = classification_metrics(y_true, y_score, threshold=t)
        hl = _hit_rate_and_plr(signal, y_true, ret_actual, label_threshold)
        row = {
            "threshold": t,
            "n_signals": hl["n_signals"],
            "hit_rate": hl["hit_rate"],
            "profit_loss_ratio": hl["profit_loss_ratio"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        }
        if ret_actual is not None:
            row["nav"] = _nav_curve(dates, signal, ret_actual, cost_rate)
        rows.append(row)

    # 汇总：推荐阈值（F1 最优；并列取信号数较少者以降低摩擦成本）。
    best = None
    valid = [r for r in rows if r["hit_rate"] is not None or r["n_signals"] > 0]
    if valid:
        best = min(valid, key=lambda r: (-r["f1"], r["n_signals"]))

    return to_json_safe(
        {
            "scan": {
                "start": scan_start,
                "end": scan_end,
                "step": scan_step,
                "min_signal_interval": int(min_signal_interval),
                "cost_rate": cost_rate,
                "label_threshold": label_threshold,
            },
            "rows": rows,
            "recommended_threshold": None if best is None else best["threshold"],
        }
    )
