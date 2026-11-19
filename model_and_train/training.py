"""Task 2.3 训练策略：get_class_weights / train_model.

对应需求§4.3 与 Todo Task 2.3：
    * `get_class_weights()`：sklearn 平衡类别权重；单类别返回 None 并告警；
      `max_class_weight=5.0` 截断上限。
    * `train_model()`：BinaryCrossentropy + Adam(clipnorm=1.0) +
      ReduceLROnPlateau + EarlyStopping + F1 最优保存（ModelCheckpoint，
      monitor=val_f1）+ 断点续训（resume_from 加载权重继续）。

    **注释（红线）**：class_weight 与 sample_weight 由 keras 在损失内部
    相乘，本模块**不对 sample_weight 做外部 clip**；class_weight 仅按
    max_class_weight 截断上限。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

import keras
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .base import BaseTimeSeriesModel, register_custom_object
from .config import DEFAULT_CONFIG, ModelConfig, merge_config

logger = logging.getLogger(__name__)


@keras.utils.register_keras_serializable(package="StockRNN")
class F1Score(keras.metrics.Metric):
    """二分类 F1 指标（阈值 0.5），用于训练监控与模型选择。"""

    def __init__(self, name: str = "f1", threshold: float = 0.5, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.reshape(y_pred, (-1,))
        y_true = tf.reshape(y_true, (-1,))
        pred_pos = tf.cast(y_pred >= self.threshold, tf.float32)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            self.tp.assign_add(tf.reduce_sum(pred_pos * y_true * sw))
            self.fp.assign_add(tf.reduce_sum(pred_pos * (1.0 - y_true) * sw))
            self.fn.assign_add(tf.reduce_sum((1.0 - pred_pos) * y_true * sw))
        else:
            self.tp.assign_add(tf.reduce_sum(pred_pos * y_true))
            self.fp.assign_add(tf.reduce_sum(pred_pos * (1.0 - y_true)))
            self.fn.assign_add(tf.reduce_sum((1.0 - pred_pos) * y_true))

    def result(self) -> tf.Tensor:
        precision = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        return 2.0 * precision * recall / (precision + recall + keras.backend.epsilon())

    def reset_state(self) -> None:
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(name=self.name, threshold=self.threshold)
        return config

    @classmethod
    def from_config(cls, config: dict) -> "F1Score":
        return cls(**config)


register_custom_object("F1Score", F1Score)


def get_class_weights(
    y: np.ndarray,
    max_class_weight: float = 5.0,
) -> Optional[Dict[int, float]]:
    """计算类别权重（sklearn balanced），上限按 max_class_weight 截断。

    Args:
        y: 训练标签 [n]（0/1）。
        max_class_weight: 权重上限（默认 5.0，Todo Task2.3）。

    Returns:
        {0: w0, 1: w1}；**训练集只有单一类别时返回 None 并告警**。
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        msg = (
            "get_class_weights: 训练集仅含单一类别 "
            f"{classes.tolist()}，返回 None（不做类别加权）"
        )
        logger.warning(msg)
        import warnings

        warnings.warn(msg, UserWarning)
        return None
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weights = np.clip(weights, None, float(max_class_weight))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def train_model(
    model: BaseTimeSeriesModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[ModelConfig] = None,
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    class_weight: Optional[Dict[int, float]] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
    extra_callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """训练单个子模型（Task 2.3）。

    Args:
        model: 已 build_model 的子模型实例。
        X_train / y_train: 训练数据 [n, look_back, F] / [n]。
        sample_weight: 训练样本权重（如低波动减权 0.5；**不外部 clip**）。
        X_val / y_val: 验证数据（用于早停、LR 衰减与 F1 最优保存）。
        config: 训练配置（None 用默认）。
        save_path: 最优模型保存路径（.keras 优先，兼容 .h5）。
        resume_from: 断点续训：若存在则加载该权重/模型继续训练。
        class_weight: 类别权重；None 时自动 get_class_weights（单类别
            None 则不做类别加权）。
        seed: 随机种子。
        verbose: keras fit verbose。
        extra_callbacks: 附加回调。

    Returns:
        dict：history / best_val_f1 / class_weight / epochs_run / status。
    """
    if model.model is None:
        raise RuntimeError("train_model: model 必须先 build_model")
    cfg = merge_config(DEFAULT_CONFIG, config)
    _seed_all(seed if seed is not None else cfg.random_seed)

    X_train = np.asarray(X_train, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    has_val = X_val is not None and y_val is not None

    # 断点续训：加载已有权重继续训练（不覆盖历史版本文件）。
    if resume_from and os.path.exists(resume_from):
        logger.info("断点续训：从 %s 加载权重继续训练", resume_from)
        model.model.load_weights(resume_from)

    # 类别权重：单类别返回 None -> 不做类别加权。
    if class_weight is None:
        class_weight = get_class_weights(y_train, max_class_weight=cfg.max_class_weight)

    # **TF 2.15 组合缺陷规避**：keras 2.15 的 data_adapter 在同时传入
    # sample_weight 与 class_weight 时，会对批量样本执行
    # `tf.reshape(sw, sw.shape + [1]*rank_delta)`，而批量 sw.shape 为
    # 部分已知形状 (None,) 导致 ValueError（Cannot convert a partially
    # known TensorShape）。因此按 keras 内部"class_weight × sample_weight"
    # 的乘法语义在**外部合并**为单一 sample_weight 传入：
    #    * 语义等价：keras 官方文档 class_weight 与 sample_weight 相乘，
    #      此处合并后结果一致；
    #    * 仍遵守红线：外部只做乘法，**不对合并结果做任何 clip**。
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype="float32")
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None and class_weight is not None:
        merged = sample_weight * np.asarray(
            [class_weight[int(c)] for c in y_train], dtype="float32"
        )
        fit_kwargs["sample_weight"] = merged
    elif sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    else:
        fit_kwargs["class_weight"] = class_weight

    model.model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm
        ),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy", F1Score(name="f1")],
    )

    callbacks: list = []
    if has_val:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.min_learning_rate,
                verbose=verbose,
            )
        )
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=cfg.early_stop_patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        )
    if save_path:
        save_path = save_path if save_path.lower().endswith((".keras", ".h5")) else save_path + ".keras"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_f1" if has_val else "loss",
                mode="max" if has_val else "min",
                save_best_only=True,
                save_weights_only=False,
                verbose=verbose,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    history = model.model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if has_val else None,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks or None,
        verbose=verbose,
        **fit_kwargs,
    )

    hist = history.history
    best_val_f1 = float(max(hist.get("val_f1", [0.0])))

    # 训练结束：EarlyStopping(restore_best_weights=True) 已把 F1 最优权重恢复
    # 到内存模型；此处用基类 save() 再写一遍模型 + .meta.json 侧车，
    # 保证 load 依赖的侧车存在（ModelCheckpoint 只写模型文件）。
    if save_path:
        model.save(save_path)

    return {
        "history": hist,
        "best_val_f1": best_val_f1,
        "class_weight": class_weight,
        "epochs_run": int(len(hist.get("loss", []))),
        "status": "ok",
    }
