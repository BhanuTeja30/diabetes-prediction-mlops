"""Shared fixtures. Tests never call external services (MLflow is disabled)."""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["MLFLOW_ENABLED"] = "false"
os.environ.setdefault("MONITORING_LOG_PREDICTIONS", "false")

from src.common import FEATURES, PROJECT_ROOT, TARGET, load_config, load_params  # noqa: E402

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "diabetes_raw.csv"


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Plausible synthetic records (with zero-coded missing values) for unit tests."""
    rng = np.random.default_rng(0)
    n = 400
    glucose = rng.normal(120, 30, n).clip(50, 250)
    bmi = rng.normal(32, 7, n).clip(15, 60)
    age = rng.integers(21, 80, n)
    logit = 0.04 * (glucose - 120) + 0.08 * (bmi - 32) + 0.02 * (age - 35) - 0.6
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "pregnancies": rng.integers(0, 10, n),
        "glucose": glucose.round(),
        "blood_pressure": rng.normal(72, 12, n).clip(40, 120).round(),
        "skin_thickness": rng.normal(28, 9, n).clip(7, 60).round(),
        "insulin": rng.normal(120, 60, n).clip(15, 600).round(),
        "bmi": bmi.round(1),
        "diabetes_pedigree_function": rng.uniform(0.08, 2.0, n).round(3),
        "age": age,
        TARGET: outcome,
    })
    df.loc[rng.choice(n, 120, replace=False), "insulin"] = 0
    df.loc[rng.choice(n, 80, replace=False), "skin_thickness"] = 0
    df.loc[rng.choice(n, 60, replace=False), "pregnancies"] = 0
    return df[FEATURES + [TARGET]]


@pytest.fixture
def params() -> dict:
    p = load_params()
    # Smaller forests keep the unit tests fast.
    p["models"]["random_forest"]["n_estimators"] = 50
    p["models"]["xgboost"]["n_estimators"] = 50
    return p


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory) -> Path:
    """Run the REAL pipeline (prepare -> train -> evaluate) on the real dataset in a temp dir."""
    if not RAW_DATA.exists():
        pytest.skip("Real dataset missing: run `python -m src.data_processing --download` first.")
    from src.data_processing import prepare
    from src.evaluate import evaluate_model_dir
    from src.train import train

    root = tmp_path_factory.mktemp("pipeline")
    params = load_params()
    prepare(params, raw_path=RAW_DATA, out_dir=root / "processed", figures_dir=root / "figures")
    train(params, processed_dir=root / "processed", output_dir=root / "model", use_mlflow=False)
    evaluate_model_dir(root / "model", processed_dir=root / "processed", out_dir=root / "evaluation")
    return root / "model"


@pytest.fixture
def test_config(tmp_path) -> dict:
    """Project config with all writable monitoring / model paths redirected into tmp_path."""
    cfg = copy.deepcopy(load_config())
    mon = cfg["monitoring"]
    mon["predictions_log"] = str(tmp_path / "monitoring" / "predictions.csv")
    mon["features_log"] = str(tmp_path / "monitoring" / "inputs.csv")
    mon["reports_dir"] = str(tmp_path / "drift")
    mon["status_file"] = str(tmp_path / "drift" / "latest_status.json")
    return cfg


@pytest.fixture
def model_copy(trained_model_dir, tmp_path) -> Path:
    dest = tmp_path / "production"
    shutil.copytree(trained_model_dir, dest)
    return dest
