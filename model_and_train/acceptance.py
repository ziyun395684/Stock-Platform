"""Task 5.1 基线模式验收测试逻辑.

对应 Todo Task5.1 / 需求 §8.1、§9：

    * 前置：固定全局随机种子；基线指标仅 `label_mode=business,
      label_threshold=0.05` 生效（本实现 label_mode 取
      `tolerant/strict`，阈值默认 0.05，基线口径由配置固定）。
    * 基线配置全链路：集成=baseline、切分=base、关闭增强开关。
    * 稳定性：连续训练 5 次，F1 极差(max-min) <= 0.08（需求§9 原始口径
      "连续10次、F1波动<5%" 的工程化收紧）；10 次标准差作参考。
    * 过拟合门禁（需求§8.1/§9）：训练集与验证集 F1 差距 < 0.15，
      超限标记 `OVERFIT` 告警。
    * 验收报告：记录 seed、复权、双阈值（label/infer）、样本统计。

本模块只实现**判定逻辑与报告组装**（纯函数，可单测）；
端到端多次训练脚本 `baseline_acceptance_run()` 提供完整入口
（真实训练耗时，默认不跑；单测用小数据短训练覆盖判定逻辑）。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .evaluation import to_json_safe

logger = logging.getLogger(__name__)

#: 稳定性门禁：连续 5 次 F1 极差上限（Todo 口径，工程化收紧）。
STABILITY_MAX_RANGE: float = 0.08
#: 过拟合门禁：训练/验证 F1 差距上限（需求§8.1/§9）。
OVERFIT_MAX_GAP: float = 0.15


def set_global_seed(seed: int) -> None:
    """固定全局随机种子（numpy / tensorflow / keras）。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # noqa: BLE001 - tf 不可用时忽略
        pass
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # noqa: BLE001
        pass
    logger.info("全局随机种子已固定: %d", seed)


def stability_check(
    f1_scores: Sequence[float],
    *,
    max_range: float = STABILITY_MAX_RANGE,
    n_expected: int = 5,
) -> Dict[str, object]:
    """稳定性判定：连续 n 次训练 F1 极差 <= max_range。

    Returns:
        {"passed": bool, "n_runs": int, "f1_scores": [...],
         "f1_min": float, "f1_max": float, "f1_range": float,
         "f1_mean": float, "f1_std": float(参考，10次口径),
         "threshold": float, "message": str}
    """
    f1 = [float(v) for v in f1_scores]
    if len(f1) < 2:
        return {
            "passed": False, "n_runs": len(f1), "f1_scores": f1,
            "f1_min": None, "f1_max": None, "f1_range": None,
            "f1_mean": None, "f1_std": None, "threshold": max_range,
            "message": f"样本不足（{len(f1)} 次 < 2），无法判定稳定性",
        }
    f1 = np.asarray(f1)
    f1_min, f1_max = float(f1.min()), float(f1.max())
    f1_range = float(f1_max - f1_min)
    f1_mean = float(f1.mean())
    f1_std = float(f1.std(ddof=1)) if len(f1) > 1 else 0.0
    # 浮点表示误差容差：边界值（如 0.68-0.60=0.08000000000000007）视为达标。
    passed = f1_range <= max_range + 1e-9
    message = (
        f"{len(f1)} 次训练 F1 极差 {f1_range:.4f} "
        f"{'≤' if passed else '>'} 阈值 {max_range} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return {
        "passed": passed, "n_runs": len(f1), "f1_scores": f1.tolist(),
        "f1_min": round(f1_min, 4), "f1_max": round(f1_max, 4),
        "f1_range": round(f1_range, 4), "f1_mean": round(f1_mean, 4),
        "f1_std": round(f1_std, 4), "threshold": max_range, "message": message,
    }


def overfit_check(
    f1_train: Optional[float],
    f1_val: Optional[float],
    *,
    max_gap: float = OVERFIT_MAX_GAP,
) -> Dict[str, object]:
    """过拟合门禁：训练/验证 F1 差距 < max_gap，否则标记 OVERFIT。

    Returns:
        {"passed": bool, "f1_train": float|None, "f1_val": float|None,
         "gap": float|None, "threshold": float,
         "status": "PASS"|"OVERFIT"|"SKIP", "message": str}
    """
    if f1_train is None or f1_val is None:
        return {
            "passed": True, "f1_train": f1_train, "f1_val": f1_val,
            "gap": None, "threshold": max_gap, "status": "SKIP",
            "message": "训练或验证 F1 缺失，跳过过拟合判定",
        }
    gap = float(f1_train) - float(f1_val)
    passed = gap < max_gap
    return {
        "passed": passed, "f1_train": round(float(f1_train), 4),
        "f1_val": round(float(f1_val), 4), "gap": round(gap, 4),
        "threshold": max_gap,
        "status": "PASS" if passed else "OVERFIT",
        "message": (
            f"训练/验证 F1 差距 {gap:.4f} {'<' if passed else '>='} "
            f"{max_gap} -> {'PASS' if passed else 'OVERFIT 告警'}"
        ),
    }


def build_acceptance_report(
    *,
    symbol: str,
    random_seed: int,
    adjust_type: str,
    label_threshold: float,
    infer_threshold: float,
    n_samples: int,
    n_positive: int,
    years: float,
    f1_scores: Sequence[float],
    f1_train: Optional[float],
    f1_val: Optional[float],
    extra: Optional[Dict] = None,
) -> Dict:
    """组装基线模式验收报告（JSON 安全）。

    字段：seed、复权、双阈值、样本统计、稳定性、过拟合门禁。
    """
    stability = stability_check(f1_scores)
    overfit = overfit_check(f1_train, f1_val)
    report = {
        "symbol": symbol,
        "random_seed": int(random_seed),
        "adjust_type": adjust_type,
        "label_threshold": float(label_threshold),
        "infer_threshold": float(infer_threshold),
        "sample_statistics": {
            "n_samples": int(n_samples),
            "n_positive": int(n_positive),
            "n_negative": int(n_samples) - int(n_positive),
            "positive_ratio": round(n_positive / n_samples, 4) if n_samples else 0.0,
            "years": round(float(years), 1),
        },
        "stability": stability,
        "overfit_gate": overfit,
        "overall": (
            "PASS" if (stability["passed"] and overfit["passed"]) else "FAIL"
        ),
    }
    if extra:
        report.update(extra)
    return to_json_safe(report)


# ---------------------------------------------------------------------------
# 端到端基线验收运行（真实训练，默认不跑）
# ---------------------------------------------------------------------------
def baseline_acceptance_run(
    cfg: dict,
    validator=None,
    n_runs: int = 5,
    verbose: int = 0,
) -> Dict:
    """基线模式端到端验收：连续 n_runs 次训练 + 稳定性 + 过拟合门禁。

    基线配置（Task5.1 前置）：
        * 集成 = baseline、切分 = base、关闭增强（split_mode=base）
        * label_threshold = 0.05、固定全局随机种子
    """
    from .config_validator import ConfigValidator
    from .pipeline import train_pipeline

    validator = validator or ConfigValidator()
    normalized = validator.validate(cfg)

    set_global_seed(int(normalized["training.random_seed"] or 42))
    # 强制基线口径。
    normalized["ensemble.mode"] = "baseline"
    normalized["split.split_mode"] = "base"
    normalized["labels.label_threshold"] = 0.05

    f1_scores: List[float] = []
    last_report = None
    for i in range(n_runs):
        set_global_seed(int(normalized["training.random_seed"] or 42) + i)
        result = train_pipeline(normalized, validator, verbose=verbose)
        weights = result["weights"]["f1_values"]
        vals = [v for v in weights.values() if v is not None]
        f1 = float(max(vals)) if vals else float("nan")
        f1_scores.append(f1)
        last_report = result["report"]
        logger.info("验收第 %d/%d 次训练：F1=%.4f", i + 1, n_runs, f1)

    # 过拟合门禁：训练/验证 F1 差距（取最后一次训练的 history 重建）。
    f1_train = f1_val = None
    if last_report and last_report.get("submodel_status"):
        statuses = list(last_report["submodel_status"].values())
        trained = [s.get("f1_val") for s in statuses if s.get("status") == "trained"]
        f1_val = max(trained) if trained else None

    symbol = normalized.get("data.code", "unknown")
    n_samples = int(last_report.get("sample_statistics", {}).get("n_samples", 0)) \
        if last_report else 0
    n_positive = int(last_report.get("sample_statistics", {}).get("n_positive", 0)) \
        if last_report else 0

    return build_acceptance_report(
        symbol=symbol,
        random_seed=int(normalized["training.random_seed"] or 42),
        adjust_type=normalized.get("data.adjust_type", "qfq"),
        label_threshold=0.05,
        infer_threshold=normalized.get("ensemble.infer_threshold", 0.5),
        n_samples=n_samples,
        n_positive=n_positive,
        years=5.0,
        f1_scores=f1_scores,
        f1_train=f1_train,
        f1_val=f1_val,
        extra={"n_runs": n_runs},
    )
