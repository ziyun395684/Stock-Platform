import hashlib
from pathlib import Path

from research_platform.backend.app.model_integrity import verify_artifact_file


def test_model_artifact_integrity_accepts_matching_digest_and_size(tmp_path: Path) -> None:
    artifact = tmp_path / "model.keras"
    content = b"model-bytes"
    artifact.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    result = verify_artifact_file(tmp_path, "model.keras", digest, len(content))
    assert result["valid"] is True
    assert result["checks"] == {"sha256": True, "size_bytes": True}


def test_model_artifact_integrity_detects_tampering_and_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "model.keras"
    artifact.write_bytes(b"original")
    result = verify_artifact_file(tmp_path, "model.keras", "0" * 64, 999)
    assert result["valid"] is False
    assert verify_artifact_file(tmp_path, "../model.keras", None, None)["valid"] is False
