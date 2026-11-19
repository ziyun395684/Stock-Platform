"""Task 2.2 LSTMAttentionModel：LSTM → Dropout → Transformer → GAP → Dense → Sigmoid.

结构（需求§4.2.1）：
    输入 [B, look_back, F]
      -> LSTM(1 层, lstm_units, return_sequences=True)
      -> Dropout(lstm_dropout 0.2-0.4)
      -> TransformerEncoder(2 层, 4 头, d_model=lstm_units, FFN=d_model*2,
                            dropout 0.1-0.2, RelativePositionBias, causal_mask)
      -> GlobalAveragePooling1D
      -> Dense(dense_units 16-32, ReLU)
      -> Dropout(dense_dropout 0.3-0.5)
      -> Dense(1, Sigmoid) -> 上涨概率

Transformer 实现与降级（Task2.2 要求）：
    * 优先尝试 `keras_nlp.layers.TransformerEncoder / RelativePositionBias`；
    * keras-nlp 导入失败（当前环境因 keras-hub 依赖链冲突必然失败）
      自动降级为 `model_and_train.layers` 自定义实现，不阻断程序；
    * RelativePositionBias 最大相对距离 = min(look_back, 120)（内存优化，
      匹配最长窗口 120，V1.2 口径）；causal_mask=True。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import keras
from keras import layers

from .base import BaseTimeSeriesModel
from .config import ModelConfig
from . import layers as custom_layers


def resolve_transformer_backend() -> str:
    """探测可用的 Transformer 实现后端。

    Returns:
        "keras_nlp"：keras_nlp 可导入且可用；
        "custom"：使用本项目自定义实现（默认路径）。
    """
    try:
        import keras_nlp  # noqa: F401
        from keras_nlp.layers import (  # noqa: F401
            RelativePositionBias,
            TransformerEncoder,
        )
        return "keras_nlp"
    except Exception:
        return "custom"


#: 模块级后端缓存（构建时探测一次；测试可 monkeypatch）。
TRANSFORMER_BACKEND: str = resolve_transformer_backend()


class LSTMAttentionModel(BaseTimeSeriesModel):
    """LSTM-Transformer 混合子模型（Task 2.2）。"""

    def __init__(
        self,
        look_back: int,
        feature_dim: int,
        config: Optional[ModelConfig] = None,
        feature_names: Optional[list] = None,
    ) -> None:
        super().__init__(look_back, feature_dim, config, feature_names)
        # 相对位置偏置最大距离：min(look_back, 120) 内存优化（V1.2 口径）。
        max_rel = min(self.config.max_relative_distance, self.look_back)
        self._max_relative_distance = int(max_rel)
        self._transformer_backend = TRANSFORMER_BACKEND
        self.model = self.build_model()

    # ------------------------------------------------------------------
    def _build_transformer(self, x: keras.KerasTensor) -> keras.KerasTensor:
        cfg = self.config
        kwargs = dict(
            intermediate_dim=cfg.lstm_units * cfg.ffn_multiplier,
            num_heads=cfg.num_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
            use_causal_mask=cfg.causal_mask,
            name="transformer",
        )
        if self._transformer_backend == "keras_nlp":
            try:
                from keras_nlp.layers import TransformerEncoder as KNEncoder

                return KNEncoder(**kwargs)(x)
            except Exception:
                # 运行期降级：keras-nlp 可用性探测通过但构建失败。
                self._transformer_backend = "custom"
        # 自定义实现（默认路径）。
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
        x = layers.LSTM(
            cfg.lstm_units,
            return_sequences=True,
            name="lstm",
        )(inputs)
        x = layers.Dropout(cfg.lstm_dropout, name="lstm_dropout")(x)
        x = self._build_transformer(x)
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
        x = layers.Dropout(cfg.dense_dropout, name="dense_dropout")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        self.model = keras.Model(inputs, outputs, name="lstm_transformer")
        return self.model

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .training import train_model

        return train_model(
            model=self,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            **kwargs,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("predict 前必须先 build_model")
        arr = np.asarray(X, dtype="float32")
        if arr.ndim != 3:
            raise ValueError(
                f"predict 输入必须为 [n, look_back, feature_dim]，收到 shape {arr.shape}"
            )
        if arr.shape[1] != self.look_back or arr.shape[2] != self.feature_dim:
            raise ValueError(
                f"predict 输入 shape {arr.shape} 与模型 "
                f"[n, {self.look_back}, {self.feature_dim}] 不匹配"
            )
        return np.asarray(self.model.predict(arr, verbose=0)).reshape(-1)
