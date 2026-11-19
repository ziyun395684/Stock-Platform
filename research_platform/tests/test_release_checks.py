from pathlib import Path

import pytest
import yaml

from research_platform.backend.app.release_checks import validate_compose_document, validate_migration_inventory


ROOT = Path(__file__).resolve().parents[1]


def test_platform_compose_has_api_worker_frontend_contract() -> None:
    document = yaml.safe_load((ROOT / "docker" / "docker-compose.platform.yml").read_text(encoding="utf-8"))
    validate_compose_document(document)


def test_platform_migration_inventory_is_complete() -> None:
    validate_migration_inventory(ROOT / "db" / "migrations")


def test_compose_check_rejects_missing_worker() -> None:
    with pytest.raises(ValueError, match="缺少平台服务"):
        validate_compose_document({"services": {"research-platform-api": {}}})
