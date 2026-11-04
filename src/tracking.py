"""Thin MLflow helpers. Tracking is optional: set MLFLOW_ENABLED=false to disable it."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-risk-prediction")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "diabetes-risk-classifier")


def mlflow_enabled() -> bool:
    return os.getenv("MLFLOW_ENABLED", "true").strip().lower() in {"1", "true", "yes"}


def tracking_uri() -> str:
    # Local default: SQLite backend (supports the model registry) + ./mlruns artifacts.
    # In Docker Compose / remote setups use e.g. http://localhost:5000.
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup():
    import mlflow

    # MLflow prints emoji (e.g. "🏃 View run ...") when talking to a tracking server. On Windows,
    # redirected output uses the legacy code page and would raise UnicodeEncodeError, so replace
    # characters the console cannot encode instead of failing the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


@contextmanager
def mlflow_client_run(run_id: str):
    """Re-open an existing run (e.g. to append test metrics from the evaluate stage)."""
    mlflow = setup()
    with mlflow.start_run(run_id=run_id):
        yield mlflow
