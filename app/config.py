"""Runtime settings for the API and frontend, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.common import load_config, resolve

DISCLAIMER = (
    "This is a model estimate for educational and research purposes, not a medical diagnosis. "
    "The model can be wrong. Please consult a qualified healthcare professional for testing and interpretation."
)
LABELS = {1: "Diabetes Risk Indicated", 0: "Diabetes Risk Not Indicated"}


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "Diabetes Risk Prediction API"
    api_version: str = "1.0.0"
    model_dir: Path = field(default_factory=lambda: resolve(os.getenv("MODEL_DIR", "models/production")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    # Operational prediction metadata (time bucket, model version, class, probability, latency):
    # no identifiers and no raw health inputs. On by default.
    log_predictions: bool = field(default_factory=lambda: _bool("MONITORING_LOG_PREDICTIONS", True))
    # Raw feature values are health data: only store them for drift analysis when an operator
    # has documented approval/consent. Off by default.
    log_features: bool = field(default_factory=lambda: _bool("MONITORING_LOG_FEATURES", False))
    # Enables POST /admin/reload when set. Never commit a real value.
    admin_token: str | None = field(default_factory=lambda: os.getenv("ADMIN_TOKEN") or None)
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))
    config: dict[str, Any] = field(default_factory=load_config)

    # --- Authentication (users and admins) ---
    auth_db_path: Path = field(default_factory=lambda: resolve(os.getenv("AUTH_DB_PATH", "data/auth/users.db")))
    # HMAC key for access tokens (>= 32 chars). If unset, a random key is generated at start-up,
    # which signs everyone out whenever the API restarts.
    auth_secret_key: str | None = field(default_factory=lambda: os.getenv("AUTH_SECRET_KEY") or None)
    token_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("TOKEN_TTL_MINUTES", "60")))
    allow_self_registration: bool = field(default_factory=lambda: _bool("ALLOW_SELF_REGISTRATION", True))
    # Bootstrap admin, created on start-up only if no active admin exists yet.
    bootstrap_admin_username: str | None = field(default_factory=lambda: os.getenv("ADMIN_USERNAME") or None)
    bootstrap_admin_password: str | None = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD") or None)
    # Admin drift-check uploads (CSV batches) may be larger than prediction requests.
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))))

    @property
    def input_limits(self) -> dict[str, dict[str, Any]]:
        return self.config["input_limits"]

    @property
    def monitoring(self) -> dict[str, Any]:
        return self.config["monitoring"]

    @property
    def max_request_bytes(self) -> int:
        return int(os.getenv("MAX_REQUEST_BYTES", self.config["api"]["max_request_bytes"]))


@lru_cache
def get_settings() -> Settings:
    return Settings()
