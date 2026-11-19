"""Shared LSTM-Transformer model with return and event prediction heads.

The existing LSTMAttentionModel remains unchanged for the legacy single-symbol
binary task. This module reuses its backbone for cross-ETF training and exposes
two predictions: pred_return and event_probability.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import keras
import tensorflow as tf
from keras import layers

from .base import BaseTimeSeriesModel
from .config import ModelConfig
from . import layers as custom_layers
from .factor_guidance import (
    build_factor_guidance_schedule,
    validate_factor_guidance_schedule,
)


def _stable_prediction_std(predictions: tf.Tensor) -> tf.Tensor:
    """Compute batch std with an epsilon so zero-variance batches stay finite."""
    centered = predictions - tf.reduce_mean(predictions)
    variance = tf.reduce_mean(tf.square(centered))
    return tf.sqrt(variance + tf.constant(1e-8, dtype=predictions.dtype))


def allocate_guidance_epochs(total_epochs: int, stages: int) -> list[int]:
    """Split the configured epoch budget across factor-guidance stages."""
    total = int(total_epochs)
    count = int(stages)
    if total <= 0 or count <= 0:
        raise ValueError("total_epochs 和 stages 必须大于 0")
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def build_date_grouped_dataset(X, targets, sample_weights, group_ids):
    """Build a finite dataset with one complete trading date per batch."""
    features = np.asarray(X, dtype="float32")
    groups = np.asarray(group_ids, dtype="float32").reshape(-1)
    if len(features) != len(groups):
        raise ValueError("group_ids 与 X 长度不一致")
    normalized_targets = {
        key: np.asarray(value, dtype="float32") for key, value in targets.items()
    }
    normalized_weights = {
        key: np.asarray(value, dtype="float32").reshape(-1)
        for key, value in sample_weights.items()
    }
    unique_groups = np.unique(groups)

    def generator():
        for group in unique_groups:
            indices = np.flatnonzero(groups == group)
            yield (
                features[indices],
                {key: value[indices] for key, value in normalized_targets.items()},
                {key: value[indices] for key, value in normalized_weights.items()},
            )

    target_signature = {
        key: tf.TensorSpec(shape=(None,) + value.shape[1:], dtype=tf.float32)
        for key, value in normalized_targets.items()
    }
    weight_signature = {
        key: tf.TensorSpec(shape=(None,), dtype=tf.float32)
        for key in normalized_weights
    }
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,) + features.shape[1:], dtype=tf.float32),
            target_signature,
            weight_signature,
        ),
    )


@keras.saving.register_keras_serializable(package="StockRNN")
class FactorGuidedHuberLoss(keras.losses.Loss):
    """Huber loss with an optional causal factor-teacher target."""

    def __init__(
        self,
        guidance_weight: float = 0.0,
        name: str = "factor_guided_huber",
        reduction: str = "auto",
    ):
        super().__init__(name=name, reduction=reduction)
        self.guidance_weight = float(guidance_weight)
        self._huber = keras.losses.Huber(reduction="none")

    def call(self, y_true, y_pred):
        target = y_true[..., :1]
        loss = self._huber(target, y_pred)
        if self.guidance_weight > 0.0 and y_true.shape[-1] is not None and y_true.shape[-1] >= 2:
            teacher = y_true[..., 1:2]
            loss = loss + self.guidance_weight * self._huber(teacher, y_pred)
        return loss

    def get_config(self):
        return {
            **super().get_config(),
            "guidance_weight": self.guidance_weight,
        }


@keras.saving.register_keras_serializable(package="StockRNN")
class RankingAwareHuberLoss(keras.losses.Loss):
    """Huber regression plus same-date pairwise ranking loss.

    ``y_true`` is ``[target_return, date_group_id]`` when ranking is enabled.
    The group id prevents comparisons across different trading dates.
    """

    def __init__(self, ranking_weight: float = 0.0, name: str = "ranking_aware_huber", reduction: str = "auto"):
        super().__init__(name=name, reduction=reduction)
        self.ranking_weight = float(ranking_weight)
        self._huber = keras.losses.Huber(reduction="none")

    def call(self, y_true, y_pred):
        target = y_true[..., :1]
        base_loss = self._huber(target, y_pred)
        if self.ranking_weight <= 0.0:
            return base_loss
        returns = tf.squeeze(target, axis=-1)
        groups = tf.squeeze(y_true[..., -1:], axis=-1)
        predictions = tf.squeeze(y_pred, axis=-1)
        target_diff = returns[:, None] - returns[None, :]
        pred_diff = predictions[:, None] - predictions[None, :]
        same_group = tf.equal(groups[:, None], groups[None, :])
        upper = tf.linalg.band_part(tf.ones_like(target_diff), 0, -1) - tf.eye(tf.shape(target_diff)[0])
        valid = same_group & (tf.abs(target_diff) > 1e-8) & (upper > 0)
        pair_loss = tf.nn.softplus(-tf.sign(target_diff) * pred_diff)
        pair_loss = tf.boolean_mask(pair_loss, valid)
        ranking_loss = tf.cond(
            tf.size(pair_loss) > 0,
            lambda: tf.reduce_mean(pair_loss),
            lambda: tf.constant(0.0, dtype=predictions.dtype),
        )
        return base_loss + self.ranking_weight * ranking_loss

    def get_config(self):
        return {**super().get_config(), "ranking_weight": self.ranking_weight}


@keras.saving.register_keras_serializable(package="StockRNN")
class FactorGuidedRankingHuberLoss(keras.losses.Loss):
    """Huber regression with both factor guidance and same-date ranking loss."""

    def __init__(
        self,
        guidance_weight: float = 0.0,
        ranking_weight: float = 0.0,
        name: str = "factor_guided_ranking_huber",
        reduction: str = "auto",
    ):
        super().__init__(name=name, reduction=reduction)
        self.guidance_weight = float(guidance_weight)
        self.ranking_weight = float(ranking_weight)
        self._huber = keras.losses.Huber(reduction="none")

    def call(self, y_true, y_pred):
        target = y_true[..., :1]
        loss = self._huber(target, y_pred)
        if self.guidance_weight > 0.0 and y_true.shape[-1] is not None and y_true.shape[-1] >= 3:
            teacher = y_true[..., 1:2]
            loss = loss + self.guidance_weight * self._huber(teacher, y_pred)
        if self.ranking_weight <= 0.0:
            return loss
        returns = tf.squeeze(target, axis=-1)
        groups = tf.squeeze(y_true[..., -1:], axis=-1)
        predictions = tf.squeeze(y_pred, axis=-1)
        target_diff = returns[:, None] - returns[None, :]
        pred_diff = predictions[:, None] - predictions[None, :]
        same_group = tf.equal(groups[:, None], groups[None, :])
        upper = tf.linalg.band_part(tf.ones_like(target_diff), 0, -1) - tf.eye(tf.shape(target_diff)[0])
        valid = same_group & (tf.abs(target_diff) > 1e-8) & (upper > 0)
        pair_loss = tf.nn.softplus(-tf.sign(target_diff) * pred_diff)
        pair_loss = tf.boolean_mask(pair_loss, valid)
        ranking_loss = tf.cond(
            tf.size(pair_loss) > 0,
            lambda: tf.reduce_mean(pair_loss),
            lambda: tf.constant(0.0, dtype=predictions.dtype),
        )
        return loss + self.ranking_weight * ranking_loss

    def get_config(self):
        return {
            **super().get_config(),
            "guidance_weight": self.guidance_weight,
            "ranking_weight": self.ranking_weight,
        }


@keras.saving.register_keras_serializable(package="StockRNN")
class DispersionAwareHuberLoss(keras.losses.Loss):
    """Huber loss with an optional lower-bound penalty on batch dispersion."""

    def __init__(
        self,
        dispersion_weight: float = 0.0,
        target_std: float = 0.0,
        name: str = "dispersion_aware_huber",
        reduction: str = "auto",
    ):
        super().__init__(name=name, reduction=reduction)
        self.dispersion_weight = float(dispersion_weight)
        self.target_std = float(target_std)
        self._huber = keras.losses.Huber(reduction="none")

    def call(self, y_true, y_pred):
        target = y_true[..., :1]
        loss = self._huber(target, y_pred)
        if self.dispersion_weight <= 0.0 or self.target_std <= 0.0:
            return loss
        predictions = tf.squeeze(y_pred, axis=-1)
        std = _stable_prediction_std(predictions)
        shortfall = tf.nn.relu(tf.cast(self.target_std, predictions.dtype) - std)
        return loss + self.dispersion_weight * tf.square(shortfall)

    def get_config(self):
        return {
            **super().get_config(),
            "dispersion_weight": self.dispersion_weight,
            "target_std": self.target_std,
        }


@keras.saving.register_keras_serializable(package="StockRNN")
class CompositeReturnLoss(keras.losses.Loss):
    """Combine factor guidance, pairwise ranking and dispersion penalties."""

    def __init__(
        self,
        guidance_weight: float = 0.0,
        ranking_weight: float = 0.0,
        dispersion_weight: float = 0.0,
        target_std: float = 0.0,
        name: str = "composite_return_loss",
        reduction: str = "auto",
    ):
        super().__init__(name=name, reduction=reduction)
        self.guidance_weight = float(guidance_weight)
        self.ranking_weight = float(ranking_weight)
        self.dispersion_weight = float(dispersion_weight)
        self.target_std = float(target_std)
        self._huber = keras.losses.Huber(reduction="none")

    def call(self, y_true, y_pred):
        target = y_true[..., :1]
        loss = self._huber(target, y_pred)
        width = y_true.shape[-1]
        if self.guidance_weight > 0.0 and width is not None and width >= 2:
            loss = loss + self.guidance_weight * self._huber(y_true[..., 1:2], y_pred)
        if self.ranking_weight > 0.0:
            returns = tf.squeeze(target, axis=-1)
            groups = tf.squeeze(y_true[..., -1:], axis=-1)
            predictions = tf.squeeze(y_pred, axis=-1)
            target_diff = returns[:, None] - returns[None, :]
            pred_diff = predictions[:, None] - predictions[None, :]
            same_group = tf.equal(groups[:, None], groups[None, :])
            upper = tf.linalg.band_part(tf.ones_like(target_diff), 0, -1) - tf.eye(tf.shape(target_diff)[0])
            valid = same_group & (tf.abs(target_diff) > 1e-8) & (upper > 0)
            pair_loss = tf.boolean_mask(tf.nn.softplus(-tf.sign(target_diff) * pred_diff), valid)
            ranking_loss = tf.cond(
                tf.size(pair_loss) > 0,
                lambda: tf.reduce_mean(pair_loss),
                lambda: tf.constant(0.0, dtype=predictions.dtype),
            )
            loss = loss + self.ranking_weight * ranking_loss
        if self.dispersion_weight > 0.0 and self.target_std > 0.0:
            predictions = tf.squeeze(y_pred, axis=-1)
            shortfall = tf.nn.relu(tf.cast(self.target_std, predictions.dtype) - _stable_prediction_std(predictions))
            loss = loss + self.dispersion_weight * tf.square(shortfall)
        return loss

    def get_config(self):
        return {
            **super().get_config(),
            "guidance_weight": self.guidance_weight,
            "ranking_weight": self.ranking_weight,
            "dispersion_weight": self.dispersion_weight,
            "target_std": self.target_std,
        }


class ValidationRankICCallback(keras.callbacks.Callback):
    """Expose raw-return validation Rank IC as a Keras callback metric."""

    def __init__(self, X_val: np.ndarray, val_meta) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val, dtype="float32")
        self.val_meta = val_meta[["trade_date", "future_return"]].reset_index(drop=True)

    def on_epoch_end(self, epoch, logs=None):
        del epoch
        logs = logs if logs is not None else {}
        raw = self.model.predict(self.X_val, verbose=0)
        if isinstance(raw, dict):
            pred_return = np.asarray(raw["pred_return"]).reshape(-1)
        else:
            pred_return = np.asarray(raw[0]).reshape(-1)
        from .panel_evaluation import mean_daily_rank_ic

        frame = self.val_meta.copy()
        frame["pred_return"] = pred_return
        rank_ic = mean_daily_rank_ic(frame)
        logs["val_rank_ic"] = float(rank_ic) if rank_ic is not None else 0.0


class GlobalBestMetricCallback(keras.callbacks.Callback):
    """Keep the best validation weights across factor-guidance stages."""

    def __init__(self, monitor: str, mode: str = "max") -> None:
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode 必须为 min 或 max")
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.best_weights = None

    def _is_better(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def on_epoch_end(self, epoch, logs=None):
        del epoch
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None or not np.isfinite(float(value)):
            return
        value = float(value)
        if self._is_better(value):
            self.best = value
            self.best_weights = [np.array(weight, copy=True) for weight in self.model.get_weights()]

    def on_train_end(self, logs=None):
        del logs
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class MultiTaskLSTMAttentionModel(BaseTimeSeriesModel):
    """LSTM-Transformer backbone with regression and classification heads."""

    def __init__(
        self,
        look_back: int,
        feature_dim: int,
        config: Optional[ModelConfig] = None,
        feature_names: Optional[list] = None,
        universe_symbols: Optional[list] = None,
    ) -> None:
        self.universe_symbols = list(universe_symbols or [])
        super().__init__(look_back, feature_dim, config, feature_names)
        configured_indicators = self.config.indicator_target_columns or []
        self.indicator_names = list(configured_indicators)
        if self.config.indicator_auxiliary_weight > 0.0 and not self.indicator_names:
            raise ValueError("启用指标辅助任务时 indicator_target_columns 不能为空")
        self._max_relative_distance = min(
            self.config.max_relative_distance, self.look_back
        )
        self.model = self.build_model()

    def _build_transformer(self, x: keras.KerasTensor) -> keras.KerasTensor:
        cfg = self.config
        return custom_layers.TransformerEncoder(
            num_layers=cfg.transformer_layers,
            num_heads=cfg.num_heads,
            d_model=cfg.lstm_units,
            ffn_multiplier=cfg.ffn_multiplier,
            dropout=cfg.transformer_dropout,
            max_relative_distance=self._max_relative_distance,
            causal_mask=cfg.causal_mask,
            name="transformer",
        )(x)

    def build_model(self) -> keras.Model:
        cfg = self.config
        inputs = keras.Input(
            shape=(self.look_back, self.feature_dim), name="sequence_input"
        )
        x = layers.LSTM(cfg.lstm_units, return_sequences=True, name="lstm")(inputs)
        x = layers.Dropout(cfg.lstm_dropout, name="lstm_dropout")(x)
        x = self._build_transformer(x)
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
        x = layers.Dropout(cfg.dense_dropout, name="dense_dropout")(x)
        pred_return = layers.Dense(1, activation="linear", name="pred_return")(x)
        event_probability = layers.Dense(
            1, activation="sigmoid", name="event_probability"
        )(x)
        outputs = {
            "pred_return": pred_return,
            "event_probability": event_probability,
        }
        if self.config.indicator_auxiliary_weight > 0.0:
            outputs["indicator_values"] = layers.Dense(
                len(self.indicator_names), activation="linear", name="indicator_values"
            )(x)
        model = keras.Model(
            inputs,
            outputs,
            name="multitask_lstm_transformer",
        )
        self._compile_model(model, guidance_weight=0.0)
        return model

    def _compile_model(self, model: keras.Model, guidance_weight: float) -> None:
        cfg = self.config
        losses = {
            "pred_return": (
                CompositeReturnLoss(
                    guidance_weight=guidance_weight,
                    ranking_weight=cfg.ranking_loss_weight,
                    dispersion_weight=cfg.dispersion_loss_weight,
                    target_std=cfg.dispersion_target_std,
                )
                if (
                    cfg.ranking_loss_weight > 0.0
                    or guidance_weight > 0.0
                    or cfg.dispersion_loss_weight > 0.0
                )
                else keras.losses.Huber()
            ),
            "event_probability": keras.losses.BinaryCrossentropy(),
        }
        loss_weights = {
            "pred_return": cfg.regression_loss_weight,
            "event_probability": cfg.classification_loss_weight,
        }
        metrics = {
            "event_probability": [keras.metrics.BinaryAccuracy(name="accuracy")]
        }
        if cfg.indicator_auxiliary_weight > 0.0:
            losses["indicator_values"] = keras.losses.Huber()
            loss_weights["indicator_values"] = cfg.indicator_auxiliary_weight
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm
            ),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
        )

    def _targets(
        self,
        y: Mapping[str, np.ndarray] | tuple,
        factor_teacher: Optional[np.ndarray] = None,
        ranking_group_ids: Optional[np.ndarray] = None,
        indicator_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        if isinstance(y, Mapping):
            ret = y.get("pred_return", y.get("future_return"))
            event = y.get("event_probability", y.get("event_label"))
        elif isinstance(y, (tuple, list)) and len(y) == 2:
            ret, event = y
        else:
            raise ValueError(
                "多任务标签必须是 mapping 或 (future_return, event_label) 二元组"
            )
        if ret is None or event is None:
            raise ValueError("多任务标签必须同时包含 future_return 和 event_label")
        ret = np.asarray(ret, dtype="float32").reshape(-1, 1)
        event = np.asarray(event, dtype="float32").reshape(-1, 1)
        if self.config.return_clip and self.config.return_clip > 0 and not self.config.return_target_scaling:
            ret = np.clip(ret, -self.config.return_clip, self.config.return_clip)
        if len(ret) != len(event):
            raise ValueError("future_return 与 event_label 长度不一致")
        if factor_teacher is not None:
            teacher = np.asarray(factor_teacher, dtype="float32").reshape(-1, 1)
            if len(teacher) != len(ret):
                raise ValueError("factor_teacher 与标签长度不一致")
            ret = np.concatenate([ret, teacher], axis=1)
        if self.config.ranking_loss_weight > 0.0:
            if ranking_group_ids is None:
                raise ValueError("启用 ranking_loss_weight 时必须提供 ranking_group_ids")
            groups = np.asarray(ranking_group_ids, dtype="float32").reshape(-1, 1)
            if len(groups) != len(ret):
                raise ValueError("ranking_group_ids 与标签长度不一致")
            ret = np.concatenate([ret, groups], axis=1)
        targets = {"pred_return": ret, "event_probability": event}
        if self.config.indicator_auxiliary_weight > 0.0:
            if indicator_targets is None:
                raise ValueError("启用指标辅助任务时必须提供 indicator_targets")
            indicators = np.asarray(indicator_targets, dtype="float32")
            if indicators.ndim != 2 or indicators.shape != (len(ret), len(self.indicator_names)):
                raise ValueError(
                    "indicator_targets shape 必须为 "
                    f"({len(ret)}, {len(self.indicator_names)})"
                )
            targets["indicator_values"] = np.nan_to_num(indicators, nan=0.0)
        return targets

    def build_sample_weights(
        self,
        event_labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Build per-head weights while compensating for rare positive events."""
        event = np.asarray(event_labels, dtype="float32").reshape(-1)
        base = (
            np.ones(len(event), dtype="float32")
            if sample_weight is None
            else np.asarray(sample_weight, dtype="float32").reshape(-1)
        )
        if len(base) != len(event):
            raise ValueError("sample_weight 长度与 event_labels 不一致")
        positive = int(np.sum(event > 0.5))
        negative = int(len(event) - positive)
        positive_weight = 1.0
        if positive and negative:
            positive_weight = min(
                float(self.config.max_class_weight), negative / positive
            )
        event_weight = base * np.where(event > 0.5, positive_weight, 1.0)
        weights = {
            "pred_return": base,
            "event_probability": event_weight.astype("float32"),
        }
        if self.config.indicator_auxiliary_weight > 0.0:
            weights["indicator_values"] = base
        return weights

    def train(
        self,
        X_train: np.ndarray,
        y_train: Mapping[str, np.ndarray] | tuple,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Mapping[str, np.ndarray] | tuple] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("train 前必须先 build_model")
        X_train = np.asarray(X_train, dtype="float32")
        factor_teacher = kwargs.get("factor_teacher")
        factor_teacher_val = kwargs.get("factor_teacher_val")
        indicator_targets = kwargs.get("indicator_targets")
        indicator_targets_val = kwargs.get("indicator_targets_val")
        ranking_group_ids = kwargs.get("ranking_group_ids")
        ranking_group_ids_val = kwargs.get("ranking_group_ids_val")
        targets = self._targets(
            y_train,
            factor_teacher=factor_teacher,
            ranking_group_ids=ranking_group_ids,
            indicator_targets=indicator_targets,
        )
        sample_weights = self.build_sample_weights(
            targets["event_probability"], sample_weight=sample_weight
        )
        validation_data = None
        val_meta = kwargs.get("val_meta")
        if X_val is not None and y_val is not None:
            validation_data = (
                np.asarray(X_val, dtype="float32"),
                self._targets(
                    y_val,
                    factor_teacher=factor_teacher_val,
                    ranking_group_ids=ranking_group_ids_val,
                    indicator_targets=indicator_targets_val,
                ),
            )
        train_dataset = None
        validation_input = validation_data
        if self.config.ranking_loss_weight > 0.0 or getattr(self.config, "batch_mode", "auto") == "date_grouped":
            train_dataset = build_date_grouped_dataset(
                X_train, targets, sample_weights, ranking_group_ids
            )
            if validation_data is not None:
                validation_input = build_date_grouped_dataset(
                    X_val,
                    validation_data[1],
                    self.build_sample_weights(validation_data[1]["event_probability"]),
                    ranking_group_ids_val,
                )
        cfg = self.config
        guidance_enabled = factor_teacher is not None
        if guidance_enabled:
            configured = cfg.factor_guidance_schedule
            schedule = (
                validate_factor_guidance_schedule(configured)
                if configured is not None
                else build_factor_guidance_schedule(
                    initial_weight=cfg.factor_guidance_weight,
                    final_weight=0.0,
                    stages=4,
                )
            )
        else:
            schedule = [0.0]

        histories = []
        global_monitor = (
            "val_rank_ic"
            if validation_data is not None and val_meta is not None and cfg.validation_monitor == "rank_ic"
            else ("val_loss" if validation_data is not None else "loss")
        )
        global_best_callback = GlobalBestMetricCallback(
            monitor=global_monitor,
            mode="max" if global_monitor == "val_rank_ic" else "min",
        )
        stage_epochs = allocate_guidance_epochs(cfg.epochs, len(schedule))
        for guidance_weight, epochs in zip(schedule, stage_epochs):
            self._compile_model(self.model, guidance_weight=guidance_weight)
            monitor = "val_loss" if validation_data else "loss"
            use_rank_ic = (
                validation_data is not None
                and val_meta is not None
                and cfg.validation_monitor == "rank_ic"
            )
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=cfg.reduce_lr_factor,
                    patience=cfg.reduce_lr_patience,
                    min_lr=cfg.min_learning_rate,
                ),
            ]
            if validation_data is not None and val_meta is not None:
                callbacks.append(ValidationRankICCallback(X_val, val_meta))
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=("val_rank_ic" if use_rank_ic else monitor),
                patience=cfg.early_stop_patience,
                mode=("max" if use_rank_ic else "auto"),
                restore_best_weights=True,
            ))
            callbacks.append(global_best_callback)
            if train_dataset is not None:
                history = self.model.fit(
                    train_dataset,
                    validation_data=validation_input,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=kwargs.get("verbose", 0),
                )
            else:
                history = self.model.fit(
                    X_train,
                    targets,
                    sample_weight=sample_weights,
                    validation_data=validation_data,
                    epochs=epochs,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=kwargs.get("verbose", 0),
                )
            histories.append(history.history)

        merged_history: Dict[str, list] = {}
        for history in histories:
            for key, values in history.items():
                merged_history.setdefault(key, []).extend(values)
        save_path = kwargs.get("save_path")
        if save_path:
            self.save(save_path)
        return {
            "history": merged_history,
            "epochs_run": len(merged_history.get("loss", [])),
            "status": "trained",
            "factor_guidance_enabled": guidance_enabled,
            "factor_guidance_weights": schedule,
            "indicator_auxiliary_enabled": self.config.indicator_auxiliary_weight > 0.0,
        }

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("predict 前必须先 build_model")
        arr = np.asarray(X, dtype="float32")
        if arr.ndim != 3:
            raise ValueError("predict 输入必须为 [n, look_back, feature_dim]")
        if arr.shape[1:] != (self.look_back, self.feature_dim):
            raise ValueError(
                f"predict 输入 shape {arr.shape} 与模型要求 "
                f"[n, {self.look_back}, {self.feature_dim}] 不匹配"
            )
        raw = self.model.predict(arr, verbose=0)
        if isinstance(raw, dict):
            result = {}
            for key, value in raw.items():
                array = np.asarray(value)
                result[key] = array.reshape(-1) if key != "indicator_values" else array
            return result
        return {
            "pred_return": np.asarray(raw[0]).reshape(-1),
            "event_probability": np.asarray(raw[1]).reshape(-1),
        }
