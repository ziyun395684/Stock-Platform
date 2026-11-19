"""Completeness and reproducibility checks for training metadata."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_training_metadata(metadata: dict[str, Any]) -> None:
    required = ("task", "model_type", "training_windows", "seeds", "config", "data_version", "config_digest", "artifact_manifest")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"训练元数据缺少字段: {missing}")
    if not metadata["training_windows"] or not metadata["seeds"]:
        raise ValueError("训练元数据必须包含 training_windows 和 seeds")
    if not isinstance(metadata["artifact_manifest"], list):
        raise ValueError("训练元数据 artifact_manifest 必须是列表")
    if metadata["config_digest"] != config_digest(metadata["config"]):
        raise ValueError("训练元数据 config_digest 与 config 不一致")
