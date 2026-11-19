"""Environment-backed platform configuration.

This module deliberately avoids reading or printing secrets. The real values
belong in a local .env file or container secret store; only .env.example is
committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during syntax-only checks
    load_dotenv = None

if load_dotenv is not None:
    # Prefer the ignored platform-local file, then fall back to the repository
    # root .env. Existing process environment variables always win.
    load_dotenv("research_platform/.env.local", override=False)
    load_dotenv(override=False)


@dataclass(frozen=True)
class PlatformSettings:
    app_name: str = "StockRNN Research Platform"
    app_env: str = "development"
    api_prefix: str = "/api"
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    artifact_root: str = "output/platform"
    storage_backend: str = "memory"

    @classmethod
    def from_env(cls) -> "PlatformSettings":
        return cls(
            app_name=os.getenv("STOCKRNN_PLATFORM_APP_NAME", cls.app_name),
            app_env=os.getenv("STOCKRNN_PLATFORM_ENV", cls.app_env),
            api_prefix=os.getenv("STOCKRNN_PLATFORM_API_PREFIX", cls.api_prefix),
            database_url=os.getenv("DATABASE_URL", ""),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            artifact_root=os.getenv("STOCKRNN_ARTIFACT_ROOT", cls.artifact_root),
            storage_backend=os.getenv("STOCKRNN_STORAGE_BACKEND", cls.storage_backend),
        )

    def public_dict(self) -> dict[str, str | bool]:
        """Return safe diagnostics; never expose database credentials."""
        return {
            "app_name": self.app_name,
            "environment": self.app_env,
            "api_prefix": self.api_prefix,
            "database_configured": bool(self.database_url),
            "redis_configured": bool(self.redis_url),
            "artifact_root": self.artifact_root,
        }


settings = PlatformSettings.from_env()
