"""Task 4.2 结构化报表输出.

对应 Todo Task4.2 / 需求 §5.3：

    * CSV：`date,ret_actual,label,prob_ensemble,prob_20,prob_60,prob_120`
      （各窗口概率按 trade_date 对齐，禁止下标跨窗口对齐）。
    * JSON 报告必填字段：`label_mode, enable_lowvol_weight,
      label_threshold, infer_threshold, adjust_type, random_seed,
      submodel_status, pos_statistics`。
    * `pos_statistics` 与 `submodel_status` 一致性校验：两者的 look_back
      窗口集合必须一致（pos 有的窗口 status 必须有，反之亦然），
      不一致标记 `consistency_check: FAIL`。
    * JSON 预处理：NaN/Inf -> None、numpy 类型转原生类型（复用
      `evaluation.to_json_safe`）。
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import to_json_safe

logger = logging.getLogger(__name__)

#: CSV 输出列（需求§4.2：date,ret_actual,label,prob_ensemble,prob_20,prob_60,prob_120）
CSV_COLUMNS: tuple = (
    "date",
    "ret_actual",
    "label",
    "prob_ensemble",
    "prob_20",
    "prob_60",
    "prob_120",
)


def check_consistency(
    pos_statistics: Dict,
    submodel_status: Dict,
) -> Dict:
    """`pos_statistics` 与 `submodel_status` 一致性校验（TC-CONS）。

    规则：两者的 look_back 窗口键集合必须一致（str 归一化比较）；
    不一致返回 consistency_check=FAIL，并列出差异。

    Returns:
        {"consistency_check": "PASS"|"FAIL", "pos_windows": [...],
         "status_windows": [...], "missing_in_status": [...],
         "missing_in_pos": [...]}
    """
    pos_windows = {str(k) for k in (pos_statistics or {}).keys()}
    status_windows = {str(k) for k in (submodel_status or {}).keys()}
    missing_in_status = sorted(pos_windows - status_windows)
    missing_in_pos = sorted(status_windows - pos_windows)
    ok = not missing_in_status and not missing_in_pos
    return {
        "consistency_check": "PASS" if ok else "FAIL",
        "pos_windows": sorted(pos_windows),
        "status_windows": sorted(status_windows),
        "missing_in_status": missing_in_status,
        "missing_in_pos": missing_in_pos,
    }


def _align_window_probs(
    dates: Sequence,
    prob_by_window: Dict[int, Sequence[float]],
    look_backs: Sequence[int],
) -> Dict[int, np.ndarray]:
    """把各窗口概率按公共 trade_date 对齐（缺值填 NaN）。

    返回 {lb: np.ndarray}，长度与公共日期一致。
    """
    common = sorted(set(dates))
    aligned: Dict[int, np.ndarray] = {}
    for lb in look_backs:
        p = np.asarray(prob_by_window.get(lb, []), dtype="float64")
        if len(p) == 0:
            aligned[lb] = np.full(len(common), np.nan)
            continue
        if len(p) != len(dates):
            raise ValueError(
                f"窗口 {lb}: 概率长度 {len(p)} 与日期长度 {len(dates)} 不一致"
            )
        idx = {d: i for i, d in enumerate(dates)}
        aligned[lb] = np.array([p[idx[d]] if d in idx else np.nan for d in common])
    return aligned


def build_report_csv(
    dates: Sequence,
    ret_actual: Sequence[float],
    label: Sequence[int],
    prob_ensemble: Sequence[float],
    prob_by_window: Optional[Dict[int, Sequence[float]]] = None,
    look_backs: Sequence[int] = (20, 60, 120),
    path: str = "report.csv",
) -> str:
    """输出预测结果 CSV（Task4.2）。

    Args:
        dates: 样本日期（升序）。
        ret_actual: 实际业务口径收益。
        label: 真实标签 0/1。
        prob_ensemble: 集成概率。
        prob_by_window: {lb: 概率数组}（与 dates 对齐；可缺失）。
        look_backs: 窗口集合（默认 20/60/120）。
        path: CSV 输出路径。

    Returns:
        实际写入路径。
    """
    dates = list(dates)
    n = len(dates)
    if not (len(ret_actual) == len(label) == len(prob_ensemble) == n):
        raise ValueError("dates / ret_actual / label / prob_ensemble 长度必须一致")

    common = sorted(set(dates))
    aligned = _align_window_probs(dates, prob_by_window or {}, look_backs)
    date_idx = {d: i for i, d in enumerate(common)}

    rows = []
    for d in common:
        i = date_idx[d]
        row = {
            "date": d,
            "ret_actual": float(ret_actual[i]),
            "label": int(label[i]),
            "prob_ensemble": float(prob_ensemble[i]),
        }
        for lb in look_backs:
            row[f"prob_{lb}"] = aligned[lb][date_idx[d]]
        rows.append(row)

    df = pd.DataFrame(rows)
    # 保证列顺序与 CSV_COLUMNS 一致（prob_20/60/120 缺失列自动补 NaN）。
    for lb in look_backs:
        col = f"prob_{lb}"
        if col not in df.columns:
            df[col] = np.nan
    df = df[["date", "ret_actual", "label", "prob_ensemble"]
            + [f"prob_{lb}" for lb in look_backs]]
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("报表 CSV 已写出: %s (%d 行)", path, len(df))
    return path


def build_report_json(
    *,
    label_mode: str,
    enable_lowvol_weight: bool,
    label_threshold: float,
    infer_threshold: float,
    adjust_type: str,
    random_seed: Optional[int],
    submodel_status: Dict,
    pos_statistics: Dict,
    extra: Optional[Dict] = None,
    path: Optional[str] = None,
) -> Dict:
    """输出结构化 JSON 报告（Task4.2）。

    必填字段全部校验存在；并执行 pos_statistics 与 submodel_status
    一致性校验（consistency_check=PASS/FAIL）。

    Returns:
        dict（JSON 安全）。path 给定时同时写入文件。
    """
    # 必填字段（random_seed 允许为 None，不参与缺失校验）。
    raw_required = {
        "label_mode": label_mode,
        "enable_lowvol_weight": enable_lowvol_weight,
        "label_threshold": label_threshold,
        "infer_threshold": infer_threshold,
        "adjust_type": adjust_type,
        "submodel_status": submodel_status,
        "pos_statistics": pos_statistics,
    }
    missing = [k for k, v in raw_required.items() if v is None]
    if missing:
        raise ValueError(f"JSON 报告缺少必填字段: {missing}")

    required = {
        "label_mode": label_mode,
        "enable_lowvol_weight": bool(enable_lowvol_weight),
        "label_threshold": float(label_threshold),
        "infer_threshold": float(infer_threshold),
        "adjust_type": adjust_type,
        "random_seed": random_seed,
        "submodel_status": submodel_status,
        "pos_statistics": pos_statistics,
    }

    consistency = check_consistency(pos_statistics, submodel_status)
    report: Dict = {
        **required,
        **consistency,
    }
    if extra:
        report.update(extra)
    report = to_json_safe(report)

    if path:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("报表 JSON 已写出: %s", path)
    return report
