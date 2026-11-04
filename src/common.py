"""Shared constants and helpers for the training / monitoring pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical snake_case feature names, in the order the model expects them.
FEATURES: list[str] = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]
TARGET = "outcome"

# Column names used by the original Kaggle/UCI CSV header (when a header is present).
ORIGINAL_COLUMN_MAP = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
    "Outcome": "outcome",
}


def resolve(path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = path or os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    with open(resolve(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(resolve(path), encoding="utf-8") as fh:
        return json.load(fh)


def git_revision() -> str | None:
    """Return the current git commit hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("matplotlib", "PIL", "urllib3", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)
