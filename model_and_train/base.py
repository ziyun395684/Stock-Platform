"""Task 2.1 模型基类与抽象接口.

`BaseTimeSeriesModel` 统一所有子模型的生命周期接口：
    build_model / train / predict / save / load

存储约定：
    * **优先 `.keras`**（keras 2.15 原生格式），兼容 `.h5`（HDF5）。
    * 保存时在模型文件旁写一个同名 `.meta.json` 侧车文件，记录
      look_back / feature_dim / config / feature_names，供 `load` 重建实例
      （.keras 本身不含这些业务元数据）。
    * `load` 通过 `custom_objects` 兼容自定义层（自定义层均已注册
      `register_keras_serializable`，此参数为双保险）。

所有子模型（如 LSTMAttentionModel）必须继承本类并实现抽象方法。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

import keras

from .config import DEFAULT_CONFIG, ModelConfig, merge_config

# 自定义层注册表（由 layers/training 模块填充），供 load_model 反序列化。
CUSTOM_OBJECTS: Dict[str, Any] = {}


def register_custom_object(name: str, cls: type) -> None:
    """登记自定义层/metric 类，供 `load` 时反序列化。"""
    CUSTOM_OBJECTS[name] = cls


def _normalize_save_path(path: str) -> str:
    """存储优先 .keras；路径无扩展名时补 .keras。"""
    lower = path.lower()
    if lower.endswith(".keras") or lower.endswith(".h5"):
        return path
    return path + ".keras"


class BaseTimeSeriesModel(ABC):
    """时序预测子模型统一抽象基类（Task 2.1）。"""

    #: 序列长度（look_back 窗口）
    look_back: int
    #: 特征维度
    feature_dim: int
    #: 合并后的模型配置
    config: ModelConfig
    #: keras 模型实例（build_model 后非 None）
    model: Optional[keras.Model]
    #: 特征名列表（与 data.sequences 的 feature_names 对齐，推理时校验）
    feature_names: Optional[list]

    def __init__(
        self,
        look_back: int,
        feature_dim: int,
        config: Optional[ModelConfig] = None,
        feature_names: Optional[list] = None,
    ) -> None:
        if int(look_back) <= 0:
            raise ValueError(f"look_back 必须为正整数，收到 {look_back!r}")
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim 必须为正整数，收到 {feature_dim!r}")
        self.look_back = int(look_back)
        self.feature_dim = int(feature_dim)
        self.config = merge_config(DEFAULT_CONFIG, config)
        self.model = None
        self.feature_names = list(feature_names) if feature_names else None

    # ------------------------------------------------------------------
    # 抽象接口（子类必须实现）
    # ------------------------------------------------------------------
    @abstractmethod
    def build_model(self) -> keras.Model:
        """构建并返回 keras 模型（可编译前向传播）。"""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """训练模型，返回含 history / best_val_f1 等信息的 dict。"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """输入 [n, look_back, feature_dim]，输出上涨概率 [n]。"""

    # ------------------------------------------------------------------
    # 具体实现：保存 / 加载
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """保存模型（优先 .keras，兼容 .h5），并写 .meta.json 侧车。"""
        if self.model is None:
            raise RuntimeError("save 前必须先 build_model")
        path = _normalize_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if path.lower().endswith(".h5"):
            self.model.save(path, save_format="h5")
        else:
            self.model.save(path)  # keras 2.15: .keras 原生格式

        meta = {
            "class": type(self).__name__,
            "look_back": self.look_back,
            "feature_dim": self.feature_dim,
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
        }
        if hasattr(self, "universe_symbols"):
            meta["universe_symbols"] = list(getattr(self, "universe_symbols") or [])
        for attr in ("scaler_metadata", "label_config", "model_version"):
            if hasattr(self, attr):
                meta[attr] = getattr(self, attr)
        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(
        cls,
        path: str,
        custom_objects: Optional[Dict[str, Any]] = None,
    ) -> "BaseTimeSeriesModel":
        """从 .keras / .h5 加载模型并重建实例（依赖 .meta.json 侧车）。"""
        path = _normalize_save_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        meta_path = path + ".meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"缺少模型元数据侧车 {meta_path}（需与模型同目录同名）"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        config = ModelConfig.from_dict(meta.get("config") or {})
        look_back = int(meta["look_back"])
        feature_dim = int(meta["feature_dim"])
        feature_names = meta.get("feature_names")

        # 反序列化模型；custom_objects 兜底（自定义层已全局注册）。
        objects = dict(CUSTOM_OBJECTS)
        if custom_objects:
            objects.update(custom_objects)
        model = keras.models.load_model(path, custom_objects=objects)

        kwargs = {
            "look_back": look_back,
            "feature_dim": feature_dim,
            "config": config,
            "feature_names": feature_names,
        }
        if "universe_symbols" in meta:
            kwargs["universe_symbols"] = meta["universe_symbols"]
        try:
            instance = cls(**kwargs)
        except TypeError:
            # Legacy single-symbol models do not accept universe metadata.
            kwargs.pop("universe_symbols", None)
            instance = cls(**kwargs)
        instance.model = model
        for attr in ("scaler_metadata", "label_config", "model_version"):
            if attr in meta:
                setattr(instance, attr, meta[attr])
        return instance
