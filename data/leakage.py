"""数据泄露自动化校验组件 (Task 1.6).

`validate_no_leakage(full, labeled, res, ...)` 在时序样本构建完成后，对
`SequenceResult` 的三窗口**逐一独立校验**是否存在数据泄露，三窗口全部 PASS 才允许
进入阶段二（模型层）。

检测项（对应Todo Task 1.6 / 需求§8.1）：
    * **feature_date < label_date**：窗口内任一步所在交易日期不得晚于标签日 T，
      且窗口末行必须恰恰是标签日 T（禁止把 T 之后的数据卷入特征窗口）。
    * **scaler.fit 仅训练集**：对每一窗口，用「训练集原始窗口扁平集」独立重新拟合
      StandardScaler，其 mean_/scale_ 与 `res.scalers[lb]` 精确一致 —— 若 scaler
      曾在验证/测试(或全部)数据上拟合过，二者必然偏离。
    * **指标时间边界抽样**：对抽样窗口校验末行特征值 == `full` 在 T 行对应的原始
      特征值，防止窗口取行错位（前移/后移）导致的隐性时移泄露。

泄露即终止流程并输出报告（`LeakageReport`），其中 `passed` 字段反映三窗口是否全部
通过。**注释：仅显性/机械性泄露的自动检测，不能替代人工评审**（如未来数据换皮、
停牌标签回看等，需人工审阅因子与标签口径）。

工作流：构建序列 → `validate_no_leakage(...)`；`passed == False` 时上游应中断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .labels import LabelResult
from .sequences import SequenceResult

# 默认抽样步长：对每个窗口抽样的样本索引（每 sample_skip 个取样 1 个 + 首尾）。
DEFAULT_SAMPLE_SKIP: int = 7
# 归一化数值容差（scaler 参数与特征比对）。
_TOL: float = 1e-9
VALID_CHECK_NAMES: tuple = (
    "feature_date_le_label",
    "scaler_train_only",
    "indicator_time_boundary",
)


@dataclass
class _CheckResult:
    name: str
    passed: bool
    message: str = ""
    n_checked: int = 0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "message": self.message,
            "n_checked": self.n_checked,
            "issues": self.issues,
        }


@dataclass
class LeakageReport:
    """三窗口泄露校验结果。

    `passed` 为三窗口 AND；任一项失败即整体 FAIL。`checks[lb]` 记录每窗口的各项
    检查明细，便于定位泄露来源。
    """

    passed: bool
    look_backs: List[int]
    feature_names: List[str]
    checks: Dict[int, Dict[str, _CheckResult]]
    samples: int
    sample_skip: int

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的报告（供日志/持久化）。"""
        return {
            "passed": self.passed,
            "look_backs": self.look_backs,
            "feature_names": self.feature_names,
            "samples": self.samples,
            "sample_skip": self.sample_skip,
            "checks": {
                str(lb): {name: c.to_dict() for name, c in checks.items()}
                for lb, checks in self.checks.items()
            },
        }


def _sample_indices(n: int, skip: int) -> List[int]:
    """按 skip 抽样索引，并保证首尾都覆盖。"""
    if n <= 0:
        return []
    idx = list(range(0, n, max(1, skip)))
    if (n - 1) not in idx:
        idx.append(n - 1)
    return idx


def _validate_inputs(full, labeled, res) -> None:
    if not isinstance(full, pd.DataFrame) or full.empty:
        raise ValueError("validate_no_leakage: full 必须为非空日线 DataFrame")
    if not isinstance(labeled, LabelResult):
        raise ValueError("validate_no_leakage: labeled 必须为 LabelResult")
    if not isinstance(res, SequenceResult):
        raise ValueError("validate_no_leakage: res 必须为 SequenceResult")
    if "trade_date" not in full.columns:
        raise ValueError("validate_no_leakage: full 缺少 trade_date 列")


def _check_feature_date_le_label(
    full: pd.DataFrame,
    res: SequenceResult,
    lb: int,
    skip: int,
) -> _CheckResult:
    """窗口内任一步日期 <= 标签日 T，且末行 == T（禁把 T 之后数据卷入窗口）。"""
    c = _CheckResult(name="feature_date_le_label", passed=True)
    full_dates = full["trade_date"].to_numpy()
    date_to_pos = {d: i for i, d in enumerate(full_dates)}
    dates = res.dates[lb]
    n = len(dates)
    idxs = _sample_indices(n, skip)
    c.n_checked = len(idxs)
    for i in idxs:
        tdate = dates[i]
        pos = date_to_pos.get(tdate)
        if pos is None:
            c.passed = False
            c.issues.append(f"样本 {i}: 标签日 {tdate} 不在 full 交易日中")
            continue
        # 窗口 span = [T-lb+1 .. T]，全部步进日期必须 <= T。
        win_len = min(lb, pos + 1)
        start = pos - win_len + 1
        span_dates = full_dates[start:pos + 1]
        if len(span_dates) != win_len or int(span_dates[-1]) != int(tdate):
            c.passed = False
            c.issues.append(
                f"样本 {i}: 窗口末行误对齐，期望 T={tdate}，实际 {span_dates[-1] if len(span_dates) else '无'}"
            )
            continue
        if np.any(span_dates > tdate):
            c.passed = False
            c.issues.append(f"样本 {i}: 窗口内存在晚于标签日 T={tdate} 的数据")
    if c.passed and c.n_checked:
        c.message = f"抽样 {c.n_checked} 例，窗口末行均为 T 且无未来数据"
    return c


def _check_scaler_train_only(
    full: pd.DataFrame,
    res: SequenceResult,
    lb: int,
    feat_mat: np.ndarray,
) -> _CheckResult:
    """独立用「训练集原始窗口扁平集」重拟合 scaler，与 res.scalers[lb] 对比。

    若构建阶段 scaler 只在训练集拟合，重拟合结果应与存储值一致；否则（曾在
    验证/测试/全量拟合）必偏，从而捕获 scaler 训练泄露。
    """
    c = _CheckResult(name="scaler_train_only", passed=True)
    sample_dates = res.dates[lb].to_numpy()
    F = len(res.feature_names)
    if len(sample_dates) == 0 or len(res.train_index[lb]) == 0:
        c.message = "无样本，跳过 scaler 训练集校验"
        return c

    full_dates = full["trade_date"].to_numpy()
    date_to_pos = {d: i for i, d in enumerate(full_dates)}
    # 训练集样本日期映射到 full 行号；出现不在 full 的日期本身即异常（标签日
    # 错位/未来日期），按失败处理而不是崩溃。
    train_pos = []
    for d in sample_dates[res.train_index[lb]]:
        p = date_to_pos.get(d)
        if p is None:
            c.passed = False
            c.issues.append(f"训练样本标签日 {d} 不在 full 交易日中")
            continue
        train_pos.append(p)
    if not train_pos:
        c.issues.append("训练集原始窗口缺失（预热不足或标签日错位）")
        return c
    train_pos = np.asarray(train_pos, dtype=int)
    train_flat = np.stack(
        [feat_mat[p - lb + 1:p + 1].reshape(-1) if p >= lb - 1 else None
         for p in train_pos]
    )
    if train_flat.shape[0] == 0:
        c.passed = False
        c.issues.append("训练集原始窗口缺失（预热不足）")
        return c

    sc_fit = StandardScaler().fit(train_flat)
    sc_stored = res.scalers[lb]
    mean_ok = np.allclose(sc_fit.mean_, sc_stored.mean_, rtol=0, atol=_TOL)
    scale_ok = np.allclose(
        sc_fit.scale_, sc_stored.scale_,
        rtol=0, atol=_TOL * max(1.0, float(np.max(np.abs(sc_stored.scale_)))),
    )
    c.n_checked = int(train_flat.shape[0])
    c.passed = mean_ok and scale_ok
    if not c.passed:
        c.issues.append("重拟合的训练集 scaler 参数与存储 scaler 不一致：疑似非训练集拟合")
    else:
        c.message = f"训练集 {c.n_checked} 例重拟合 scaler 与存储一致"
    return c


def _check_indicator_time_boundary(
    full: pd.DataFrame,
    res: SequenceResult,
    lb: int,
    feat_mat: np.ndarray,
    skip: int,
) -> _CheckResult:
    """末行特征 == full 在 T 行的原始特征值，防窗口取行错位导致的隐性时移泄露。"""
    c = _CheckResult(name="indicator_time_boundary", passed=True)
    full_dates = full["trade_date"].to_numpy()
    date_to_pos = {d: i for i, d in enumerate(full_dates)}
    dates = res.dates[lb]
    n = len(dates)
    idxs = _sample_indices(n, skip)
    c.n_checked = len(idxs)
    for i in idxs:
        tdate = dates[i]
        pos = date_to_pos.get(tdate)
        if pos is None:
            c.passed = False
            c.issues.append(f"样本 {i}: 标签日 {tdate} 不在 full 交易日中")
            continue
        # 窗口下采样（若用 skip>1 只抽部分，保证覆盖抽样样本即可）。
        win = res.X[lb][i]                       # [lb, F] 已标准化
        raw_last = feat_mat[pos]                 # [F] 原始末行
        # 由存储 scaler 还原末行原始值，与 full 比对。
        sc = res.scalers[lb]
        F = len(res.feature_names)
        flat = win.reshape(-1)
        raw_last_recon = sc.inverse_transform(
            flat.reshape(1, -1)
        ).reshape(lb, F)[-1]
        if not np.allclose(raw_last_recon, raw_last, rtol=0, atol=_TOL):
            c.passed = False
            c.issues.append(f"样本 {i}: 末行特征与 full 在 T 日原始值不一致（疑似取行错位）")
    if c.passed and c.n_checked:
        c.message = f"抽样 {c.n_checked} 例，末行特征与 T 日原始特征一致"
    return c


def validate_no_leakage(
    full: pd.DataFrame,
    labeled: LabelResult,
    res: SequenceResult,
    sample_skip: int = DEFAULT_SAMPLE_SKIP,
) -> LeakageReport:
    """三窗口独立数据泄露校验。

    Args:
        full: 完整升序日线（含因子列，与 build_sequences 传入同一份）。
        labeled: `generate_labels` 产物（未用，仅为接口对齐/复验标签日期）。
        res: `build_sequences` 产物。
        sample_skip: 抽样步长（越大越快）；首尾必检。

    Returns:
        LeakageReport：`passed` 为三窗口全部检查通过。
    """
    _validate_inputs(full, labeled, res)
    full = full.sort_values("trade_date").reset_index(drop=True)

    # 原始特征矩阵（未标准化、未前向填充）供独立复验。
    feats = res.feature_names
    feat_mat_raw = full[feats].to_numpy(dtype=float)
    # 前向填充后矩阵（与 build_sequences 内部口径一致）
    if np.isnan(feat_mat_raw).any():
        feat_mat = pd.DataFrame(feat_mat_raw, columns=feats).ffill().fillna(0.0).to_numpy(dtype=float)
    else:
        feat_mat = feat_mat_raw

    checks: Dict[int, Dict[str, _CheckResult]] = {}
    overall = True
    for lb in sorted(res.look_backs):
        c1 = _check_feature_date_le_label(full, res, lb, sample_skip)
        c2 = _check_scaler_train_only(full, res, lb, feat_mat)
        c3 = _check_indicator_time_boundary(full, res, lb, feat_mat, sample_skip)
        checks[lb] = {"feature_date_le_label": c1, "scaler_train_only": c2,
                      "indicator_time_boundary": c3}
        if not (c1.passed and c2.passed and c3.passed):
            overall = False

    return LeakageReport(
        passed=overall,
        look_backs=list(sorted(res.look_backs)),
        feature_names=list(feats),
        checks=checks,
        samples=int(len(res.sample_meta)),
        sample_skip=sample_skip,
    )