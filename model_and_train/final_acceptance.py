"""One-time final-test freeze and result recording."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_final_test_manifest(
    config: Mapping[str, Any],
    universe: Sequence[str],
    test_start: Any,
    test_end: Any,
    output_path: str | Path,
) -> dict:
    """Freeze all final-test inputs before the test is claimed."""
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"最终测试 manifest 已存在：{path}")
    manifest = {
        "status": "FROZEN",
        "frozen_at": _utc_now(),
        "config_sha256": _config_hash(config),
        "config": dict(config),
        "universe": list(universe),
        "test_start": test_start,
        "test_end": test_end,
        "claimed_at": None,
        "completed_at": None,
        "result": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def claim_final_test(path: str | Path) -> dict:
    """Claim the frozen test exactly once."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") == "CLAIMED":
        raise RuntimeError("最终测试只能领取一次")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError(f"最终测试 manifest 状态不可领取：{manifest.get('status')}")
    manifest["status"] = "CLAIMED"
    manifest["claimed_at"] = _utc_now()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def record_final_test_result(path: str | Path, result: Mapping[str, Any]) -> dict:
    """Record the immutable result after the one-time claim."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CLAIMED":
        raise RuntimeError("最终测试 manifest 尚未领取")
    manifest["status"] = "COMPLETED"
    manifest["completed_at"] = _utc_now()
    manifest["result"] = dict(result)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest
