import pytest

from research_platform.backend.app.independent_retest import build_retest_config
from research_platform.backend.app.training_metadata import config_digest, validate_training_metadata


def test_training_metadata_requires_complete_reproducibility_fields() -> None:
    config = {"data_dir": "quotes2", "start": "2025-01-01", "end": "2025-12-31"}
    metadata = {"task": "train", "model_type": "factor-common-v1", "training_windows": [500], "seeds": [42], "config": config, "data_version": "quotes2:2025-01-01:2025-12-31", "config_digest": config_digest(config), "artifact_manifest": []}
    validate_training_metadata(metadata)
    metadata["config_digest"] = "0" * 64
    with pytest.raises(ValueError, match="config_digest"):
        validate_training_metadata(metadata)


def test_frozen_retest_config_digest_is_stable() -> None:
    strategy = {"id": "s", "strategy_id": "frozen-v1", "status": "frozen"}
    version = {"id": "v", "config": {"end": "2025-12-31", "top_k": 2}}
    request = {"start": "2026-01-01", "end": "2026-03-31"}
    first = build_retest_config(strategy, version, request)
    second = build_retest_config(strategy, version, request)
    assert first["retest_config_digest"] == second["retest_config_digest"]
