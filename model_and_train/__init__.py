"""Model & training layer package (M2/M3).

阶段二：模型层开发（Task 2.1~2.4）。
阶段三：集成与评估层开发（Task 3.1~3.4，含 3.2b 阈值灵敏度）。
技术基线：tensorflow 2.15.1 + keras 2.15.0。
注意：**不使用 keras-nlp** —— keras-nlp 0.31.x 依赖 keras-hub，
而 keras-hub 会强制升级 keras3 并引入 tensorflow-text，与
tf2.15/keras2.15 冲突（详见 requirements.txt 注释）。因此
TransformerEncoder / RelativePositionBias 采用自定义实现
（`model_and_train.layers`），并保留"keras-nlp 可用则优先、不可用
自动降级自定义"的兼容逻辑（Task2.2 要求）。
"""
from __future__ import annotations

from importlib import import_module

_MODULES = (
    "acceptance", "backtest", "base", "baseline", "config", "config_validator",
    "data_pipeline", "ensemble", "evaluation", "inference", "layers",
    "lstm_transformer", "pipeline", "report", "threshold_sensitivity",
    "train_pipeline", "training", "visualization", "multitask_model", "realtime_signals",
)

# Keep package import lightweight. TensorFlow/Keras-backed modules are loaded
# only when the requested attribute is accessed, so data/ranking utilities can
# run in environments that intentionally omit the ML runtime.
def __getattr__(name: str):
    if name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_MODULES))

__all__ = [
    "base",
    "config",
    "layers",
    "lstm_transformer",
    "training",
    "train_pipeline",
    "ensemble",
    "evaluation",
    "threshold_sensitivity",
    "baseline",
    "backtest",
    "visualization",
    "report",
    "config_validator",
    "pipeline",
    "inference",
    "data_pipeline",
    "acceptance",
    "realtime_signals",
]
