"""模型层超参数配置 (Task 2.2 / 2.3 / 2.4).

超参统一以需求文档 §4.2（模型结构）与 §9（超参表）为准写入本模块，
供 `LSTMAttentionModel` / `train_model` / `train_submodels` 共用；
阶段四（Task4.3）再将其接入 config.yaml + ConfigValidator。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Union


@dataclass
class ModelConfig:
    # ---------- 模型结构（需求§4.2 / §9 超参表） ----------
    # LSTM 层：1 层 32-64 单元（10 年数据可 128），return_sequences=True
    lstm_units: int = 64
    # LSTM 后 Dropout：0.2-0.4
    lstm_dropout: float = 0.3
    # Transformer 编码器层数：2（固定）
    transformer_layers: int = 2
    # 注意力头数：4（可试 2/8）
    num_heads: int = 4
    # Transformer dropout：0.1-0.2
    transformer_dropout: float = 0.1
    # FFN 维度 = d_model * 2（d_model 与 LSTM 单元数一致）
    ffn_multiplier: int = 2
    # GAP 后全连接：16-32 单元 ReLU
    dense_units: int = 32
    # 全连接后 Dropout：0.3-0.5
    dense_dropout: float = 0.4
    # 相对位置偏置最大相对距离：120（匹配最长 look_back=120，需求§10.1 V1.2）
    max_relative_distance: int = 120
    # 因果掩码：固定开启，禁止未来信息泄露
    causal_mask: bool = True
    # 多任务学习：收益率回归与上涨事件分类损失权重。
    regression_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    # 收益率标签裁剪范围，None/0 表示不裁剪。
    return_clip: float = 0.2
    return_target_scaling: bool = False
    ranking_target: str = "raw"
    ranking_loss_weight: float = 0.0
    dispersion_loss_weight: float = 0.0
    dispersion_target_std: float = 0.0
    batch_mode: str = "auto"
    validation_monitor: str = "rank_ic"
    # 原策略因子教师信号：训练阶段按 schedule 逐步衰减。
    factor_guidance_weight: float = 0.5
    factor_guidance_schedule: Optional[list[float]] = None
    factor_guidance_epochs_per_stage: int = 1
    # 指标辅助任务：默认关闭，开启后新增 indicator_values 输出头。
    indicator_auxiliary_weight: float = 0.0
    indicator_target_columns: list[str] = None

    # ---------- 训练（需求§4.3） ----------
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stop_patience: int = 15          # 早停 patience 15-20
    reduce_lr_patience: int = 5            # 验证 loss 连续 5 轮不降则衰减
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    clipnorm: float = 1.0                  # 梯度裁剪，防梯度爆炸
    max_class_weight: float = 5.0          # 类别权重上限截断（Task2.3）
    random_seed: Optional[int] = None

    # ---------- Task2.4 三周期批量训练门禁 ----------
    # 子模型正例门禁：正例数 < min_positive_count 或正例占比 < min_positive_ratio 则跳过
    min_positive_count: int = 50
    min_positive_ratio: float = 0.05
    min_sample_count: int = 50
    # 默认串行；True 且多 GPU 时并行
    parallel_training: bool = False

    # ---------- 推理/集成 ----------
    infer_threshold: float = 0.5

    # ---------- 保存 ----------
    # 存储优先 .keras，兼容 .h5
    save_format: str = ".keras"

    # ---------- 便捷方法 ----------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ModelConfig":
        if not d:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# 默认配置单例（不可变语义：使用时 via merge_config）。
DEFAULT_CONFIG = ModelConfig()


def merge_config(
    base: Optional[ModelConfig] = None,
    override: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
) -> ModelConfig:
    """合并配置：override 覆盖 base；None 视为不覆盖。

    Args:
        base: 基础配置，None 时用 DEFAULT_CONFIG。
        override: 覆盖项，可以是 ModelConfig 或 dict。

    Returns:
        新的 ModelConfig 实例（不修改入参）。
    """
    base = base or DEFAULT_CONFIG
    merged = ModelConfig.from_dict(base.to_dict())
    if override is None:
        return merged
    if isinstance(override, ModelConfig):
        override = override.to_dict()
    if not isinstance(override, dict):
        raise TypeError(f"override 必须为 dict 或 ModelConfig，收到 {type(override)!r}")
    for k, v in override.items():
        if hasattr(merged, k):
            setattr(merged, k, v)
        else:
            raise ValueError(f"merge_config: 未知配置项 {k!r}")
    return merged
