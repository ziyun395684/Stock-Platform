"""Task 2.2 自定义 Transformer 层（keras-nlp 降级实现）.

技术背景：需求§4.2.3 位置编码方案为 RelativePositionBias，首选
`keras_nlp.layers`；但 keras-nlp 0.31.x 依赖 keras-hub，而 keras-hub
会强制升级 keras3 并引入 tensorflow-text，与 tf2.15/keras2.15 冲突。
因此这里**自定义实现**（对应需求§10.2「备选：自定义实现」）：

    * `RelativePositionBias`：相对位置偏置表 [2*max_dist+1, H]，最大相对
      距离默认 120（匹配最长 look_back=120，V1.2 口径）；构造时按
      `min(look_back, 120)` 缩减表长做内存优化；`causal_mask=True` 时
      未来位置（key > query）偏置置 -inf，配合 softmax 即因果掩码。
    * `TransformerEncoder`：2 层编码器块，4 头注意力，d_model 与 LSTM
      一致，FFN=d_model*2，Pre-LN + 残差 + Dropout，内置因果掩码。

两个层均实现 get_config/from_config 并注册 keras 序列化，保证
.keras 保存/加载 roundtrip 正常。
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
import keras
from keras import layers

from .base import register_custom_object


@keras.utils.register_keras_serializable(package="StockRNN")
class RelativePositionBias(layers.Layer):
    """相对位置偏置层（自定义实现，需求§4.2.3）。

    call 输入为注意力 logits，shape [B, T, T, H]（B 批、T 序列长、
    H 头数），返回 logits + bias。bias 由可学习表
    `relative_embeddings[2*max_relative_distance+1, H]` 经相对距离索引
    得到；`causal_mask=True` 时 key 索引 > query 索引的位置置 -inf。
    """

    def __init__(
        self,
        max_relative_distance: int = 120,
        num_heads: int = 4,
        causal_mask: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if int(max_relative_distance) < 0:
            raise ValueError("max_relative_distance 不能为负")
        self.max_relative_distance = int(max_relative_distance)
        self.num_heads = int(num_heads)
        self.causal_mask = bool(causal_mask)

    def build(self, input_shape) -> None:
        table_len = 2 * self.max_relative_distance + 1
        self.relative_embeddings = self.add_weight(
            name="relative_embeddings",
            shape=(table_len, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, logits: tf.Tensor, training=None) -> tf.Tensor:
        # logits: [B, T, T, H]
        T = tf.shape(logits)[1]
        # 相对距离矩阵：dist[i, j] = j - i（key 相对 query）
        positions = tf.range(T)
        dist = positions[None, :] - positions[:, None]  # [T, T]
        dist = tf.clip_by_value(
            dist, -self.max_relative_distance, self.max_relative_distance
        )
        idx = dist + self.max_relative_distance  # [T, T] -> [0, 2*max]
        bias = tf.gather(self.relative_embeddings, idx)  # [T, T, H]

        if self.causal_mask:
            # 因果掩码：query 位置 i 只能关注 key 位置 j <= i。
            # 用 band_part 保留下三角（含对角线），上三角（未来）置 -inf。
            upper = 1.0 - tf.linalg.band_part(
                tf.ones([T, T], dtype=tf.float32), -1, 0
            )  # [T, T]：上三角=1（含对角线外），下三角=0
            bias = tf.where(
                tf.cast(upper[None, :, :, None] > 0, tf.bool),
                tf.fill(tf.shape(bias), float("-inf")),
                bias,
            )
        return logits + bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            max_relative_distance=self.max_relative_distance,
            num_heads=self.num_heads,
            causal_mask=self.causal_mask,
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "RelativePositionBias":
        return cls(**config)


@keras.utils.register_keras_serializable(package="StockRNN")
class TransformerEncoder(layers.Layer):
    """Transformer 编码器（自定义实现，需求§4.2.3）。

    结构（每块）：
        x -> LayerNorm -> 多头自注意力(含相对位置偏置+因果掩码) -> Dropout
          -> + 残差 -> LayerNorm -> FFN(d_model -> d_model*2 -> d_model)
          -> Dropout -> + 残差

    Args:
        num_layers: 编码器块层数（默认 2）。
        num_heads: 注意力头数（默认 4）。
        d_model: 模型宽度，须与 LSTM 单元数一致且能被 num_heads 整除。
        ffn_multiplier: FFN 维度 = d_model * ffn_multiplier（默认 2）。
        dropout: 各子层 dropout（0.1-0.2）。
        max_relative_distance: 相对位置偏置最大距离（默认 120；
            调用方可传 min(look_back, 120) 做内存优化）。
        causal_mask: 因果掩码开关（默认 True，禁止未来泄露）。
    """

    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 4,
        d_model: int = 64,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
        max_relative_distance: int = 120,
        causal_mask: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if int(num_layers) <= 0:
            raise ValueError("num_layers 必须为正整数")
        if int(num_heads) <= 0:
            raise ValueError("num_heads 必须为正整数")
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        self.head_dim = self.d_model // self.num_heads
        self.ffn_multiplier = int(ffn_multiplier)
        self.ffn_dim = self.d_model * self.ffn_multiplier
        self.dropout = float(dropout)
        self.max_relative_distance = int(max_relative_distance)
        self.causal_mask = bool(causal_mask)

        # 每块子层（按层索引组织列表，序列化时由 get_config 重建）。
        self.query_dense = [
            layers.Dense(self.d_model, name=f"query_{i}") for i in range(self.num_layers)
        ]
        self.key_dense = [
            layers.Dense(self.d_model, name=f"key_{i}") for i in range(self.num_layers)
        ]
        self.value_dense = [
            layers.Dense(self.d_model, name=f"value_{i}") for i in range(self.num_layers)
        ]
        self.output_dense = [
            layers.Dense(self.d_model, name=f"out_{i}") for i in range(self.num_layers)
        ]
        self.ln1 = [
            layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")
            for i in range(self.num_layers)
        ]
        self.ln2 = [
            layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")
            for i in range(self.num_layers)
        ]
        self.attn_dropout = [
            layers.Dropout(self.dropout, name=f"attn_drop_{i}")
            for i in range(self.num_layers)
        ]
        self.ffn_dropout = [
            layers.Dropout(self.dropout, name=f"ffn_drop_{i}")
            for i in range(self.num_layers)
        ]
        self.ffn1 = [
            layers.Dense(self.ffn_dim, activation="relu", name=f"ffn1_{i}")
            for i in range(self.num_layers)
        ]
        self.ffn2 = [
            layers.Dense(self.d_model, name=f"ffn2_{i}") for i in range(self.num_layers)
        ]
        self.relative_bias = RelativePositionBias(
            max_relative_distance=self.max_relative_distance,
            num_heads=self.num_heads,
            causal_mask=self.causal_mask,
            name="relative_position_bias",
        )

    def _attention(
        self, x: tf.Tensor, i: int, training: Optional[bool]
    ) -> tf.Tensor:
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        q = self.query_dense[i](x)  # [B, T, d_model]
        k = self.key_dense[i](x)
        v = self.value_dense[i](x)

        def _split(t: tf.Tensor) -> tf.Tensor:
            return tf.transpose(
                tf.reshape(t, (B, T, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )  # [B, H, T, head_dim]

        q, k, v = _split(q), _split(k), _split(v)
        logits = tf.matmul(q, k, transpose_b=True)  # [B, H, T, T]
        logits = logits / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # 相对位置偏置：logits [B, H, T, T] -> [B, T, T, H] -> 加 bias -> 转回
        logits = tf.transpose(logits, (0, 2, 3, 1))  # [B, T, T, H]
        logits = self.relative_bias(logits)
        logits = tf.transpose(logits, (0, 3, 1, 2))  # [B, H, T, T]

        attn_weights = tf.nn.softmax(logits)  # 上三角已被 -inf 屏蔽
        attn_weights = self.attn_dropout[i](attn_weights, training=training)
        out = tf.matmul(attn_weights, v)  # [B, H, T, head_dim]
        out = tf.transpose(out, (0, 2, 1, 3))  # [B, T, H, head_dim]
        out = tf.reshape(out, (B, T, self.d_model))
        return self.output_dense[i](out)

    def _block(self, x: tf.Tensor, i: int, training: Optional[bool]) -> tf.Tensor:
        # Pre-LN 注意力块 + 残差
        attn_out = self._attention(self.ln1[i](x), i, training)
        x = x + self.attn_dropout[i](attn_out, training=training)
        # FFN 块 + 残差
        ffn_out = self.ffn2[i](self.ffn1[i](self.ln2[i](x)))
        x = x + self.ffn_dropout[i](ffn_out, training=training)
        return x

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        for i in range(self.num_layers):
            x = self._block(x, i, training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.d_model,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            max_relative_distance=self.max_relative_distance,
            causal_mask=self.causal_mask,
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TransformerEncoder":
        return cls(**config)


# 登记自定义层供 BaseTimeSeriesModel.load 反序列化兜底。
register_custom_object("RelativePositionBias", RelativePositionBias)
register_custom_object("TransformerEncoder", TransformerEncoder)
