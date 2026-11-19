"""Task 4.4 增量推理流水线.

对应 Todo Task4.4 / 需求 §4.4：

    * `incremental_factor_update()`：行情连续性校验，断档降级全量重算；
    * `load_model_and_predict()`：严格对齐 `feature_names` 特征顺序；
    * 边界：缓存损坏/复权变更触发全量重建；输入 NaN/Inf 返回空信号 +
      告警不崩溃；
    * `incremental_train()`：追加新交易日数据续训，沿用法
      `trained_feature_names` 校验特征一致，进度写入训练日志并保存为
      新 version（禁止覆盖旧版本）；新增数据不足时跳过并告警；
    * 子任务：多窗口推理结果按 `trade_date` 外连接 Join，缺值填 NaN。

本模块设计为**模型无关**：输入子模型 + 特征矩阵 + 日期，输出对齐概率，
不依赖具体 keras 模型，便于单测与后续接入 EnsembleModel。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 增量训练最少新增交易日数（不足则跳过并告警）。
DEFAULT_MIN_INCREMENT_DAYS: int = 10


# ---------------------------------------------------------------------------
# 特征对齐
# ---------------------------------------------------------------------------
def validate_feature_names(
    trained_feature_names: Sequence[str],
    new_feature_names: Sequence[str],
    model_name: str = "model",
) -> List[str]:
    """严格校验新特征列与训练特征列一致（顺序敏感，Task4.4 验收）。

    Raises:
        ValueError: 特征名/顺序不一致（缺列或顺序不同）。
    """
    trained = list(trained_feature_names)
    new = list(new_feature_names)
    if trained != new:
        missing = [c for c in trained if c not in new]
        extra = [c for c in new if c not in trained]
        order_diff = [c for c in trained if c in new] != [c for c in new if c in trained]
        raise ValueError(
            f"{model_name} 特征不一致：缺列 {missing}，多余列 {extra}，"
            f"顺序不同={order_diff}；训练特征={trained}，新特征={new}"
        )
    return trained


# ---------------------------------------------------------------------------
# 因子增量更新
# ---------------------------------------------------------------------------
def incremental_factor_update(
    new_rows: pd.DataFrame,
    prev_rows: Optional[pd.DataFrame] = None,
    *,
    max_gap_days: int = 5,
    factor_fn=None,
) -> Dict:
    """增量因子更新（行情连续性校验）。

    Args:
        new_rows: 新交易日数据（统一 Schema，升序）。
        prev_rows: 之前已处理的数据（用于连续性校验与因子计算预热）；
            None 或为空时视为全量首建。
        max_gap_days: 交易日最大间隔（超过视为断档）。
        factor_fn: 因子计算函数；None 时用 data.features.compute_factors。

    Returns:
        {"status": "incremental"|"full_rebuild"|"empty", "reason": str,
         "factors": pd.DataFrame|None, "n_new_rows": int}

    断档/连续性校验失败 -> status="full_rebuild"，调用方应全量重算。
    """
    if new_rows is None or len(new_rows) == 0:
        return {"status": "empty", "reason": "无新增数据", "factors": None, "n_new_rows": 0}

    work = new_rows.copy()
    if "trade_date" not in work.columns:
        raise ValueError("new_rows 缺少 trade_date 列")
    work = work.sort_values("trade_date").reset_index(drop=True)

    if prev_rows is None or len(prev_rows) == 0:
        # 无历史：只能全量首建。
        return _compute_factors(work, factor_fn, "full_rebuild",
                                "无历史数据，全量重建因子")

    prev = prev_rows.sort_values("trade_date")["trade_date"].to_numpy()
    new_dates = work["trade_date"].to_numpy()
    prev_last = prev[-1]

    # 连续性校验：新数据首日与历史末日必须衔接（间隔 <= max_gap_days）。
    gap = None
    if len(new_dates):
        try:
            gap = int(new_dates[0] - prev_last)
        except TypeError:
            gap = None
    if gap is not None and gap > max_gap_days:
        logger.warning(
            "行情断档：历史末日 %s 与新增首日 %s 间隔 %s 天 > %s，降级全量重算",
            prev_last, new_dates[0], gap, max_gap_days,
        )
        return _compute_factors(work, factor_fn, "full_rebuild",
                                f"行情断档（间隔 {gap} 天 > {max_gap_days}）")

    # 连续性 OK：拼接历史 + 新增（新增行用于因子滚动窗口预热），
    # 但只返回新增行上的因子（增量）。
    full = pd.concat([prev_rows, work], ignore_index=True)
    return _compute_factors(full, factor_fn, "incremental",
                            f"增量更新 {len(work)} 行", only_new=work["trade_date"])


def _compute_factors(
    df: pd.DataFrame,
    factor_fn,
    status: str,
    reason: str,
    only_new: Optional[pd.Series] = None,
) -> Dict:
    """统一因子计算入口。"""
    fn = factor_fn or _default_factors
    try:
        out = fn(df)
    except Exception as exc:  # noqa: BLE001 - 因子失败降级全量重建
        logger.warning("因子计算失败(%s)，返回空：%s", type(exc).__name__, exc)
        return {"status": "full_rebuild", "reason": f"因子计算异常: {exc}",
                "factors": None, "n_new_rows": 0}
    if only_new is not None:
        out = out[out["trade_date"].isin(set(only_new.tolist()))].reset_index(drop=True)
    return {"status": status, "reason": reason, "factors": out,
            "n_new_rows": int(len(out))}


def _default_factors(df: pd.DataFrame) -> pd.DataFrame:
    """默认因子：仅用 data.features（若可用），否则返回原数据。"""
    try:
        from data.features import compute_factors

        out, _ = compute_factors(df)
        return out
    except Exception:  # noqa: BLE001
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 多窗口推理 + 日期外连接
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """多窗口推理结果（trade_date 外连接，缺值填 NaN）。"""

    dates: List = field(default_factory=list)
    prob_by_window: Dict[int, List[float]] = field(default_factory=dict)
    prob_ensemble: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"trade_date": self.dates})
        for lb, probs in self.prob_by_window.items():
            df[f"prob_{lb}"] = probs
        if self.prob_ensemble:
            df["prob_ensemble"] = self.prob_ensemble
        return df


def join_window_predictions(
    window_predictions: Dict[int, pd.DataFrame],
    look_backs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """多窗口推理结果按 trade_date 外连接 Join，缺值填 NaN（Task4.4 子任务）。

    Args:
        window_predictions: {lb: DataFrame(trade_date, prob)}（每窗口样本
            日期可能不同，禁止下标对齐）。
        look_backs: 窗口顺序（默认按 dict 键排序）。

    Returns:
        DataFrame：trade_date + 各窗口 prob 列（外连接全日期并集）。
    """
    if not window_predictions:
        raise ValueError("window_predictions 为空")
    lbs = list(look_backs) if look_backs else sorted(window_predictions.keys())
    merged: Optional[pd.DataFrame] = None
    for lb in lbs:
        wp = window_predictions[lb]
        if "trade_date" not in wp.columns or "prob" not in wp.columns:
            raise ValueError(f"窗口 {lb} 预测结果缺少 trade_date/prob 列")
        sub = wp[["trade_date", "prob"]].rename(columns={"prob": f"prob_{lb}"})
        merged = sub if merged is None else pd.merge(
            merged, sub, on="trade_date", how="outer"
        )
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    return merged


def load_model_and_predict(
    predict_fn,
    X: np.ndarray,
    dates: Sequence,
    trained_feature_names: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    *,
    model_name: str = "model",
) -> Dict:
    """加载模型并预测（严格特征对齐 + NaN 防护，Task4.4）。

    Args:
        predict_fn: 可调用 fn(X) -> 概率数组（如 model.predict）。
        X: 特征矩阵 [n, look_back, F]（已标准化或原始，取决于 predict_fn）。
        dates: 与 X 对齐的 trade_date [n]。
        trained_feature_names: 模型训练时保存的 feature_names。
        feature_names: 当前输入特征名（与 trained 严格比对）；None 跳过校验。
        model_name: 模型标识（错误信息用）。

    Returns:
        {"dates": [...], "prob": [...], "n": int, "warnings": [...]}；
        输入含 NaN/Inf 时返回空信号 + 告警（不崩溃）。
    """
    X = np.asarray(X, dtype="float64")
    dates = np.asarray(list(dates))
    if X.ndim != 3:
        raise ValueError(f"{model_name}: X 必须为 [n, look_back, F]，收到 {X.shape}")
    if X.shape[0] != len(dates):
        raise ValueError(f"{model_name}: X 行数 {X.shape[0]} 与 dates {len(dates)} 不一致")

    # 严格特征顺序对齐。
    if feature_names is not None:
        validate_feature_names(trained_feature_names, feature_names, model_name)

    # NaN/Inf 输入 -> 空信号 + 告警，不崩溃。
    if not np.isfinite(X).all():
        logger.warning("%s: 输入包含 NaN/Inf，返回空信号", model_name)
        return {"dates": [], "prob": [], "n": 0,
                "warnings": [f"{model_name}: 输入含 NaN/Inf，已返回空信号"]}

    prob = np.asarray(predict_fn(X.astype("float32")), dtype="float64").reshape(-1)
    if prob.shape[0] != len(dates):
        raise ValueError(f"{model_name}: 预测结果长度 {prob.shape[0]} 与 dates 不一致")
    return {"dates": dates.tolist(), "prob": prob.tolist(), "n": int(len(dates)),
            "warnings": []}


# ---------------------------------------------------------------------------
# 增量训练
# ---------------------------------------------------------------------------
def incremental_train(
    model,
    new_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    min_increment_days: int = DEFAULT_MIN_INCREMENT_DAYS,
    save_dir: str = "model_and_train/checkpoints",
    symbol: str = "unknown",
    look_back: Optional[int] = None,
    label_threshold: float = 0.05,
    label_mode: str = "tolerant",
) -> Dict:
    """增量训练：追加新交易日数据续训，保存为新 version（禁止覆盖旧版本）。

    Args:
        model: 已加载的 `BaseTimeSeriesModel` 子类实例（含 trained
            feature_names，训练时由 build_sequences 的 feature_names 注入）。
        new_df: 新增数据（统一 Schema 升序，含因子列）。
        feature_columns: 新数据中与训练对齐的特征列。
        min_increment_days: 新增数据不足该天数 -> 跳过并告警。
        save_dir / symbol / look_back: 模型命名（lstm_transformer_{lb}_{symbol}_v{n}.keras）。
        label_threshold / label_mode: 新标签口径。

    Returns:
        {"status": "trained"|"skipped"|"error", "reason": str,
         "new_version": int|None, "path": str|None, "best_val_f1": float|None}
    """
    if new_df is None or len(new_df) < min_increment_days:
        logger.warning(
            "incremental_train: 新增数据 %s 行 < %s，跳过增量训练",
            0 if new_df is None else len(new_df), min_increment_days,
        )
        return {"status": "skipped",
                "reason": f"新增数据不足 {min_increment_days} 行",
                "new_version": None, "path": None, "best_val_f1": None}

    try:
        # 特征一致性校验（沿用 trained feature_names）。
        trained_feats = getattr(model, "feature_names", None)
        if trained_feats is None:
            return {"status": "error", "reason": "模型缺少 trained feature_names",
                    "new_version": None, "path": None, "best_val_f1": None}
        validate_feature_names(trained_feats, feature_columns, model_name="incremental_train")

        # 用新数据追加标签训练。
        from data.labels import generate_labels
        from .train_pipeline import model_filename, next_model_version

        labeled = generate_labels(
            new_df, label_threshold=label_threshold, label_mode=label_mode
        )
        if len(labeled.y) == 0:
            return {"status": "skipped", "reason": "新数据无有效标签样本",
                    "new_version": None, "path": None, "best_val_f1": None}

        X = new_df[trained_feats].to_numpy(dtype="float64")
        look_back = look_back or model.look_back
        if len(X) <= look_back:
            return {"status": "skipped",
                    "reason": f"新数据行数 {len(X)} <= look_back {look_back}，无法构建窗口",
                    "new_version": None, "path": None, "best_val_f1": None}

        # 标签按 trade_date 对齐（generate_labels 会剔除无效样本，行位置
        # 与 new_df 不对齐，禁止用 iloc 下标跨表取标签）。
        label_by_date = dict(zip(labeled.df["trade_date"], labeled.y))

        # 滑动窗口构建（与 data.sequences 同构但简化：仅用新数据）。
        new_dates = new_df["trade_date"].to_numpy()
        wins, yy = [], []
        for t in range(look_back, len(X)):
            d = new_dates[t]
            if d not in label_by_date:
                continue  # 该样本日无有效标签（与主流程同一剔除口径）
            wins.append(X[t - look_back + 1:t + 1])
            yy.append(int(label_by_date[d]))
        if not wins:
            return {"status": "skipped", "reason": "新数据无有效标签样本",
                    "new_version": None, "path": None, "best_val_f1": None}
        Xw = np.stack(wins).astype("float32")
        yw = np.asarray(yy, dtype="float32")

        # 续训（从当前权重继续）并保存新版本（不覆盖旧版本）。
        version = next_model_version(save_dir, look_back, symbol)
        save_path = os.path.join(save_dir, model_filename(look_back, symbol, version))
        result = model.train(
            Xw, yw, config=model.config, save_path=save_path,
            resume_from=getattr(model, "_latest_path", None),
            verbose=0,
        )
        logger.info(
            "incremental_train: look_back=%d 保存为新版本 v%d -> %s (F1=%.4f)",
            look_back, version, save_path, result["best_val_f1"],
        )
        return {"status": "trained", "reason": "ok",
                "new_version": version, "path": save_path,
                "best_val_f1": result["best_val_f1"]}
    except Exception as exc:  # noqa: BLE001 - 增量失败不阻断主流程
        logger.warning("incremental_train 失败: %s", exc)
        return {"status": "error", "reason": str(exc),
                "new_version": None, "path": None, "best_val_f1": None}
