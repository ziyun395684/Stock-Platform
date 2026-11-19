"""Task 4.5 性能调优：tf.data 流水线（时序不 shuffle）。

对应 Todo Task4.5：
    * tf.data 流水线，**禁止 .shuffle()** —— 金融时序数据打乱会破坏
      时间序语义（且与阶段一"时序切分禁 shuffle"红线一致）；
    * 单元测试校验时序顺序保持。

性能目标（供训练脚本参考）：单子模型 GPU 训练 <=5min、总训练 <=15min、
单样本推理 <0.5s；本模块提供高效数据管道，不改变语义。
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def build_tf_dataset(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    batch_size: int = 32,
    *,
    shuffle: bool = False,
    seed: Optional[int] = None,
    drop_remainder: bool = False,
):
    """构建 tf.data.Dataset（默认**不 shuffle**，保持时序顺序）。

    Args:
        X: 特征 [n, look_back, F]。
        y: 标签 [n]（可 None，仅特征）。
        sample_weight: 样本权重 [n]（可 None）。
        batch_size: 批大小。
        shuffle: **默认 False**；设为 True 必须显式传 seed 且仅供调试。
        seed: 随机种子（shuffle=True 时必需）。
        drop_remainder: 是否丢弃不足一批的尾部。

    Returns:
        tf.data.Dataset：仅特征 -> (X,)；有标签 -> (X, y)；有权重 ->
        (X, y, w)。
    """
    import tensorflow as tf

    X = np.asarray(X, dtype="float32")
    if X.ndim != 3:
        raise ValueError(f"X 必须为 [n, look_back, F]，收到 {X.shape}")
    n = X.shape[0]

    tensors = [X]
    if y is not None:
        y = np.asarray(y, dtype="float32")
        if y.shape[0] != n:
            raise ValueError("y 长度与 X 不一致")
        tensors.append(y)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype="float32")
        if w.shape[0] != n:
            raise ValueError("sample_weight 长度与 X 不一致")
        tensors.append(w)

    dataset = tf.data.Dataset.from_tensor_slices(tuple(tensors))
    if shuffle:
        if seed is None:
            raise ValueError("shuffle=True 时必须提供 seed（调试用）")
        dataset = dataset.shuffle(buffer_size=n, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def verify_order_preserved(X: np.ndarray, batch_size: int = 8) -> bool:
    """校验 tf.data 流水线（无 shuffle）是否保持原始行顺序。

    逐样本比对重建顺序与原始 X 行号一致（Task4.5 单元测试用）。
    """
    ds = build_tf_dataset(X, batch_size=batch_size)
    recon = []
    for batch in ds:
        b = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon.append(np.asarray(b))
    if not recon:
        return True
    flat = np.concatenate(recon, axis=0)
    return flat.shape[0] == X.shape[0] and np.allclose(flat, X, atol=1e-7)
