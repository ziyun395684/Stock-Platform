"""Task 2.4 三周期子模型批量训练脚本.

对 `data.sequences.build_sequences` 产出的 `SequenceResult`，为每个
look_back 窗口（20/60/120）独立训练一个 LSTMAttentionModel：

    * **默认串行**；`parallel_training=True` 且存在多 GPU 时用
      `tf.distribute.MirroredStrategy` 并行，否则告警并降级串行。
    * 读取配置 `min_positive_count / min_positive_ratio`：正例不足的窗口
      **自动跳过**，记录到 `submodel_status[lb]`（status="skipped"）。
    * 命名：`lstm_transformer_{look_back}_{symbol}_{version}.keras`；
      version 递增（扫描 output_dir 已有最大版本 +1），**禁止覆盖旧版本**。
    * 输出：训练日志、各窗口 F1、子模型状态、正例统计。

集成衔接：`load_submodels()` 按版本加载可用的子模型（缺失窗口自动忽略），
供阶段三 `EnsembleModel` 使用 —— 部分子模型缺失/禁用时集成仍可正常运行。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

import tensorflow as tf

from .config import DEFAULT_CONFIG, ModelConfig, merge_config
from .lstm_transformer import LSTMAttentionModel

logger = logging.getLogger(__name__)

#: 模型文件命名：lstm_transformer_{look_back}_{symbol}_{version}.keras
MODEL_FILE_PATTERN = re.compile(
    r"^lstm_transformer_(\d+)_(.+?)_(\d+)\.(keras|h5)$"
)


def model_filename(look_back: int, symbol: str, version: int, ext: str = ".keras") -> str:
    """生成模型文件名：lstm_transformer_{look_back}_{symbol}_{version}.keras。"""
    return f"lstm_transformer_{look_back}_{symbol}_{version}{ext}"


def next_model_version(
    output_dir: str, look_back: int, symbol: str, ext: str = ".keras"
) -> int:
    """扫描 output_dir，返回该 look_back+symbol 的下一个版本号（禁止覆盖）。"""
    max_version = 0
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            m = MODEL_FILE_PATTERN.match(fname)
            if not m:
                continue
            lb, sym, ver, fext = m.group(1), m.group(2), m.group(3), m.group(4)
            if int(lb) == int(look_back) and sym == symbol and f".{fext}" == ext:
                max_version = max(max_version, int(ver))
    return max_version + 1


def _positive_statistics(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    total = int(len(y))
    positive = int(np.sum(y == 1))
    return {
        "total": total,
        "positive": positive,
        "negative": total - positive,
        "ratio": round(positive / total, 4) if total else 0.0,
    }


def _build_and_train(
    look_back: int,
    feature_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    cfg: ModelConfig,
    save_path: str,
    strategy: Optional[tf.distribute.Strategy],
    seed: Optional[int],
    verbose: int,
) -> Dict:
    def _run() -> Dict:
        model = LSTMAttentionModel(
            look_back=look_back,
            feature_dim=feature_dim,
            config=cfg,
        )
        return model.train(
            X_train,
            y_train,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            config=cfg,
            save_path=save_path,
            seed=seed,
            verbose=verbose,
        )

    if strategy is not None:
        with strategy.scope():
            return _run()
    return _run()


def train_submodels(
    seq_res,
    symbol: str,
    config: Optional[ModelConfig] = None,
    output_dir: str = "model_and_train/checkpoints",
    parallel_training: Optional[bool] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
) -> Dict:
    """为三周期窗口批量训练子模型（Task 2.4）。

    Args:
        seq_res: `data.sequences.SequenceResult`（三窗口独立张量与划分）。
        symbol: 标的代码（用于模型命名）。
        config: 训练/门禁配置；None 用默认。
        output_dir: 模型输出目录（自动创建）。
        parallel_training: 是否多 GPU 并行；None 时读 config.parallel_training。
        seed: 随机种子。
        verbose: keras fit verbose。

    Returns:
        dict：
            submodel_status: {lb: {status: "trained"|"skipped", reason?,
                                   f1_val?, path?, version?, ...}}
            pos_statistics: {lb: {total, positive, negative, ratio}}
            model_files: 已训练模型路径列表
            parallel_training: 实际生效的并行标志
    """
    cfg = merge_config(DEFAULT_CONFIG, config)
    parallel = cfg.parallel_training if parallel_training is None else bool(parallel_training)
    os.makedirs(output_dir, exist_ok=True)

    strategy: Optional[tf.distribute.Strategy] = None
    if parallel:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) >= 2:
            strategy = tf.distribute.MirroredStrategy()
            logger.info("Task2.4: 多 GPU 并行训练，GPU 数量=%d", len(gpus))
        else:
            logger.warning(
                "Task2.4: parallel_training=True 但可用 GPU=%d（<2），降级为串行",
                len(gpus),
            )
            parallel = False

    look_backs = sorted(int(lb) for lb in seq_res.look_backs)
    submodel_status: Dict = {}
    pos_statistics: Dict = {}
    model_files: List[str] = []
    feature_dim = int(seq_res.X[look_backs[0]].shape[-1]) if look_backs else 0

    for lb in look_backs:
        X = np.asarray(seq_res.X[lb], dtype="float32")
        y = np.asarray(seq_res.y[lb], dtype="float32")
        pos_stats = _positive_statistics(y)
        pos_statistics[lb] = pos_stats

        # 正例门禁：不满足自动跳过，记录 submodel_status。
        skip_reason = None
        if pos_stats["positive"] < cfg.min_positive_count:
            skip_reason = (
                f"正例数 {pos_stats['positive']} < min_positive_count {cfg.min_positive_count}"
            )
        elif pos_stats["ratio"] < cfg.min_positive_ratio:
            skip_reason = (
                f"正例占比 {pos_stats['ratio']} < min_positive_ratio {cfg.min_positive_ratio}"
            )
        if skip_reason:
            submodel_status[lb] = {
                "status": "skipped",
                "reason": skip_reason,
                **pos_stats,
            }
            logger.warning("Task2.4: look_back=%d 跳过：%s", lb, skip_reason)
            continue

        version = next_model_version(output_dir, lb, symbol)
        save_path = os.path.join(output_dir, model_filename(lb, symbol, version))
        train_idx = np.asarray(seq_res.train_index[lb], dtype=int)
        val_idx = np.asarray(seq_res.val_index[lb], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx] if len(val_idx) else None
        y_val = y[val_idx] if len(val_idx) else None
        # 样本权重（如低波动减权 0.5）仅取训练子集；不外部 clip。
        w_all = np.asarray(seq_res.weight[lb], dtype="float32")
        sample_weight = w_all[train_idx] if w_all.size else None

        result = _build_and_train(
            look_back=lb,
            feature_dim=feature_dim,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            cfg=cfg,
            save_path=save_path,
            strategy=strategy,
            seed=seed,
            verbose=verbose,
        )
        submodel_status[lb] = {
            "status": "trained",
            "f1_val": result["best_val_f1"],
            "path": save_path,
            "version": version,
            "epochs_run": result["epochs_run"],
            "class_weight": result["class_weight"],
            **pos_stats,
        }
        model_files.append(save_path)
        logger.info(
            "Task2.4: look_back=%d 训练完成，val F1=%.4f，保存 %s",
            lb,
            result["best_val_f1"],
            save_path,
        )

    return {
        "symbol": symbol,
        "output_dir": output_dir,
        "parallel_training": parallel,
        "submodel_status": submodel_status,
        "pos_statistics": pos_statistics,
        "model_files": model_files,
    }


def load_submodels(
    output_dir: str,
    symbol: str,
    look_backs: List[int],
) -> Dict[int, LSTMAttentionModel]:
    """按版本加载已训练子模型（缺失窗口自动忽略）。

    供阶段三 `EnsembleModel` 使用：部分子模型缺失/禁用时，仅加载
    可用的窗口，集成逻辑据此正常运行（Task 2.4 验收项）。
    配置从模型侧车 .meta.json 还原（BaseTimeSeriesModel.load）。
    """
    loaded: Dict[int, LSTMAttentionModel] = {}
    for lb in look_backs:
        candidates = []
        if os.path.isdir(output_dir):
            for fname in os.listdir(output_dir):
                m = MODEL_FILE_PATTERN.match(fname)
                if m and int(m.group(1)) == int(lb) and m.group(2) == symbol:
                    candidates.append((int(m.group(3)), os.path.join(output_dir, fname)))
        if not candidates:
            logger.warning("load_submodels: look_back=%d 无可用模型，跳过", lb)
            continue
        candidates.sort(key=lambda t: t[0])
        _, latest = candidates[-1]
        model = LSTMAttentionModel.load(latest)
        loaded[lb] = model
        logger.info("load_submodels: 加载 look_back=%d <- %s", lb, latest)
    return loaded
