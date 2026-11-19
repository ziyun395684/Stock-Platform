"""External dependency diagnostics for the platform API."""
from __future__ import annotations

from .config import settings
from .database import check_database


def check_redis() -> dict[str, str | bool]:
    if not settings.redis_url:
        return {"configured": False, "reachable": False, "detail": "REDIS_URL missing"}
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        return {"configured": True, "reachable": True, "detail": "ok"}
    except Exception as exc:  # dependency diagnostics must not crash the API
        return {"configured": True, "reachable": False, "detail": type(exc).__name__}


def dependency_status() -> dict[str, object]:
    return {
        "database": check_database(),
        "redis": check_redis(),
    }
