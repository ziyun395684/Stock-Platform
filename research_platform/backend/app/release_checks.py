"""Static release checks that do not require Docker or external services."""
from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_SERVICES = {"research-platform-api", "research-platform-worker", "research-platform-frontend"}
REQUIRED_MIGRATIONS = {"001_initial.sql", "002_governance_and_results.sql", "003_strategy_lineage.sql"}


def validate_compose_document(document: dict[str, Any]) -> None:
    services = document.get("services")
    if not isinstance(services, dict) or not REQUIRED_SERVICES.issubset(services):
        raise ValueError(f"Compose 缺少平台服务: {sorted(REQUIRED_SERVICES - set(services or {}))}")
    for name in REQUIRED_SERVICES:
        service = services[name]
        if not isinstance(service, dict) or not service.get("build") or not service.get("restart"):
            raise ValueError(f"Compose 服务缺少 build/restart: {name}")
    for name in ("research-platform-api", "research-platform-worker"):
        environment = services[name].get("environment", {})
        if environment.get("STOCKRNN_STORAGE_BACKEND") != "mysql":
            raise ValueError(f"Compose 服务必须使用 MySQL backend: {name}")
        if "DATABASE_URL" not in environment or "REDIS_URL" not in environment:
            raise ValueError(f"Compose 服务缺少数据库或 Redis 配置: {name}")


def validate_migration_inventory(directory: Path) -> None:
    names = {path.name for path in directory.glob("*.sql")}
    missing = REQUIRED_MIGRATIONS - names
    if missing:
        raise ValueError(f"缺少平台迁移文件: {sorted(missing)}")
    ordered = sorted(names)
    if ordered.index("003_strategy_lineage.sql") < ordered.index("001_initial.sql"):
        raise ValueError("迁移版本顺序无效")
