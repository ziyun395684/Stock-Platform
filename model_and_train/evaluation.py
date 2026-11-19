"""Task 3.2 分类指标评估模块.

对应 Todo Task3.2 / 需求 §5.3：

    * 指标：Accuracy / Precision / Recall / F1 / AUC / 混淆矩阵 / 阈值扫描。
    * 输出预处理：`np.nan/np.inf -> None`，numpy 类型转为原生 Python 类型
      （保证 JSON 序列化不报错）。
    * 与 sklearn 对齐：直接用 sklearn.metrics 计算，测试做数值对齐。

另含 `to_json_safe()` 通用递归转换器，供 Task3.2b / Task3.3 / Task3.4
及阶段四 JSON 报告共用。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: 默认阈值扫描范围（需求 §8.5：0.3-0.7，步长 0.05）。
DEFAULT_THRESHOLDS: tuple = tuple(np.round(np.arange(0.30, 0.71, 0.05), 2))


def to_json_safe(value: Any) -> Any:
    """递归把 numpy / NaN / Inf 转成 JSON 安全类型。

    * np.nan / ±np.inf -> None
    * numpy 标量 -> 原生 Python 类型
    * numpy 数组 / pandas Index -> list
    * dict / list / tuple -> 递归处理
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (int, np.integer, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [to_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    # pandas 类型兜底
    if hasattr(value, "item"):
        return to_json_safe(value.item())
    return value


def classification_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """计算二分类全套指标（阈值扫描单点版本）。

    Args:
        y_true: 真实标签 0/1。
        y_score: 预测概率（0-1）。
        threshold: 正例判定阈值。

    Returns:
        dict：accuracy / precision / recall / f1 / auc / confusion /
        threshold / n_samples / n_positive。所有值均为原生类型，
        无正例或单类别时 auc 返回 None。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("y_true / y_score 必须为等长一维数组")

    pred = (y_score >= threshold).astype(int)
    n = int(len(y_true))
    n_pos = int(np.sum(y_true == 1))

    acc = float(accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    auc: Optional[float] = None
    if 0 < n_pos < n:
        auc = float(roc_auc_score(y_true, y_score))

    return to_json_safe(
        {
            "threshold": threshold,
            "n_samples": n,
            "n_positive": n_pos,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    )


def threshold_scan(
    y_true: Sequence[int],
    y_score: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """阈值扫描（Task3.2 阈值扫描部分）。

    返回每个阈值一行指标 dict，格式与 `classification_metrics` 对齐。
    """
    thresholds = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLDS)
    rows = []
    for t in thresholds:
        row = classification_metrics(y_true, y_score, threshold=float(t))
        # 阈值扫描行额外给出信号数（预测为正的数量）。
        pred = (np.asarray(y_score) >= float(t)).astype(int)
        row["n_signals"] = int(np.sum(pred == 1))
        rows.append(row)
    return rows


def metrics_table(rows: Sequence[Dict[str, Any]]) -> str:
    """把阈值扫描行格式化为可读文本表（供日志/报表）。"""
    header = f"{'threshold':>9} {'n_signals':>9} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}"
    lines = [header]
    for r in rows:
        auc = "  -  " if r.get("auc") is None else f"{r['auc']:.4f}"
        lines.append(
            f"{r['threshold']:>9.2f} {r['n_signals']:>9d} "
            f"{r['accuracy']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['f1']:>6.3f} {auc}"
        )
    return "\n".join(lines)
