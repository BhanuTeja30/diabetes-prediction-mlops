"""Training stage: fit and compare candidate models, select one on validation data.

Every model is a single scikit-learn Pipeline (zero-code handling -> imputation ->
scaling -> classifier) so the exact preprocessing used in training is what the API
applies at prediction time. The held-out test split is NOT touched here.

Usage:
    python -m src.train
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common import (
    FEATURES,
    TARGET,
    get_logger,
    git_revision,
    load_config,
    load_params,
    read_json,
    resolve,
    sha256_file,
    write_json,
)
from src.evaluate import compute_metrics, tune_threshold
from src.features import ZeroAsMissing

log = get_logger("train")

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def build_pipelines(params: dict[str, Any], y_train: pd.Series) -> dict[str, Pipeline]:
    seed = params["split"]["random_seed"]
    zero_cols = params["zero_as_missing"]
    m = params["models"]

    lr = m["logistic_regression"]
    rf = m["random_forest"]
    xg = dict(m["xgboost"])
    if xg.get("scale_pos_weight") is None:
        xg["scale_pos_weight"] = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    return {
        "logistic_regression": Pipeline([
            ("zero_as_missing", ZeroAsMissing(zero_cols)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], class_weight=lr["class_weight"],
                                       random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("zero_as_missing", ZeroAsMissing(zero_cols)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
                                           min_samples_leaf=rf["min_samples_leaf"],
                                           class_weight=rf["class_weight"], random_state=seed, n_jobs=-1)),
        ]),
        # XGBoost handles NaN natively, so no imputer is needed.
        "xgboost": Pipeline([
            ("zero_as_missing", ZeroAsMissing(zero_cols)),
            ("clf", XGBClassifier(**xg, random_state=seed, eval_metric="logloss", n_jobs=1)),
        ]),
    }


def select_model(results: dict[str, dict[str, Any]], target_recall: float) -> tuple[str, str]:
    """Pick the model with the best validation specificity among those meeting target recall."""
    eligible = {k: v for k, v in results.items() if v["validation"]["recall"] >= target_recall}
    if not eligible:
        raise RuntimeError(f"No model reached the target validation recall of {target_recall:.2f}.")
    ranked = sorted(
        eligible.items(),
        key=lambda kv: (-round(kv[1]["validation"]["specificity"], 4), kv[1]["validation"]["brier_score"]),
    )
    best = ranked[0][0]
    v = results[best]["validation"]
    reason = (
        f"{MODEL_DISPLAY_NAMES[best]} had the highest validation specificity ({v['specificity']:.3f}) while "
        f"keeping recall >= {target_recall:.2f} (recall={v['recall']:.3f}); Brier score {v['brier_score']:.3f} "
        "was used as tie-breaker. Accuracy was not used for selection."
    )
    return best, reason


def next_version(production_dir: Path) -> str:
    meta = production_dir / "metadata.json"
    if not meta.exists():
        return "1.0.0"
    major, minor, _patch = (int(x) for x in read_json(meta)["model_version"].split("."))
    return f"{major}.{minor + 1}.0"


def train(params: dict[str, Any] | None = None, processed_dir: str | Path | None = None,
          output_dir: str | Path | None = None, use_mlflow: bool | None = None) -> dict[str, Any]:
    params = params or load_params()
    config = load_config()
    processed_dir = resolve(processed_dir or params["data"]["processed_dir"])
    output_dir = resolve(output_dir or config["model"]["candidate_dir"])
    target_recall = params["selection"]["target_recall"]

    from src.tracking import mlflow_enabled

    use_mlflow = mlflow_enabled() if use_mlflow is None else use_mlflow

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    dataset_version = sha256_file(processed_dir / "train.csv")[:12]

    pipelines = build_pipelines(params, y_train)
    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        duration = time.perf_counter() - start
        prob_val = pipe.predict_proba(X_val)[:, 1]
        threshold = tune_threshold(y_val, prob_val, target_recall)
        results[name] = {
            "training_seconds": round(duration, 3),
            "validation": compute_metrics(y_val, prob_val, threshold),
            "validation_at_0_5": compute_metrics(y_val, prob_val, 0.5),
        }
        fitted[name] = pipe
        v = results[name]["validation"]
        log.info("%-20s t=%.3f recall=%.3f spec=%.3f roc_auc=%.3f brier=%.3f (%.2fs)", name, threshold,
                 v["recall"], v["specificity"], v["roc_auc"], v["brier_score"], duration)

    best, reason = select_model(results, target_recall)
    log.info("Selected: %s. %s", best, reason)

    version = next_version(resolve(config["model"]["production_dir"]))
    metadata = {
        "model_name": MODEL_DISPLAY_NAMES[best],
        "model_key": best,
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_threshold": results[best]["validation"]["threshold"],
        "features": FEATURES,
        "dataset": {
            "source": params["data"]["source_name"],
            "source_url": params["data"]["source_url"],
            "raw_sha256": sha256_file(resolve(params["data"]["raw_path"]))
            if resolve(params["data"]["raw_path"]).exists() else None,
            "train_version": dataset_version,
            "processed_dir": processed_dir.relative_to(resolve(".")).as_posix()
            if processed_dir.is_relative_to(resolve(".")) else str(processed_dir),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
        },
        "selection": {"criteria": params["selection"], "reason": reason},
        "validation_metrics": results[best]["validation"],
        "all_models_validation": results,
        "hyperparameters": params["models"][best],
        "random_seed": params["split"]["random_seed"],
        "git_revision": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "known_limitations": [
            "Trained on 768 records of Pima Indian women aged 21+; may not generalise to other "
            "populations, men, children or different clinical settings.",
            "Many insulin and skin-thickness values are missing in the source data and were imputed.",
            "The decision threshold favours sensitivity, so false positives are expected.",
            "Probabilities are model estimates and may be imperfectly calibrated.",
            "Not a diagnostic device; it has not been clinically validated.",
        ],
    }

    if use_mlflow:
        metadata.update(_log_to_mlflow(params, results, best, fitted[best], X_train, metadata))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], output_dir / "model.joblib")
    write_json(output_dir / "metadata.json", metadata)
    comparison_path = (resolve("reports/evaluation/validation_comparison.json")
                       if output_dir == resolve(config["model"]["candidate_dir"])
                       else output_dir / "validation_comparison.json")
    write_json(comparison_path, {"selected": best, "reason": reason, "models": results})
    log.info("Saved candidate model v%s to %s", version, output_dir)
    return metadata


def _log_to_mlflow(params, results, best, best_pipe, X_train, metadata) -> dict[str, Any]:
    """Log every model as a nested run; register the selected pipeline as a candidate version."""
    try:
        from mlflow.models import infer_signature

        from src.tracking import REGISTERED_MODEL_NAME, setup

        mlflow = setup()
        with mlflow.start_run(run_name=f"training-v{metadata['model_version']}") as parent:
            mlflow.set_tags({
                "model_version": metadata["model_version"],
                "selected_model": best,
                "dataset_version": metadata["dataset"]["train_version"],
                "git_revision": metadata["git_revision"] or "unavailable",
            })
            mlflow.log_params({"random_seed": params["split"]["random_seed"],
                               "target_recall": params["selection"]["target_recall"],
                               "dataset_raw_sha256": metadata["dataset"]["raw_sha256"],
                               "dataset_train_version": metadata["dataset"]["train_version"]})
            for name, res in results.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("model_type", name)
                    mlflow.log_params({f"hp_{k}": v for k, v in params["models"][name].items()})
                    mlflow.log_metric("training_duration_seconds", res["training_seconds"])
                    mlflow.log_metrics({f"val_{k}": v for k, v in res["validation"].items()
                                        if isinstance(v, (int, float))})
            mlflow.log_params({f"selected_hp_{k}": v for k, v in params["models"][best].items()})
            mlflow.log_param("selected_model_type", best)
            mlflow.log_metrics({f"val_{k}": v for k, v in results[best]["validation"].items()
                                if isinstance(v, (int, float))})
            mlflow.log_metric("training_duration_seconds", results[best]["training_seconds"])
            sample = X_train.head(5).astype(float)
            info = mlflow.sklearn.log_model(
                sk_model=best_pipe,
                name="model",
                signature=infer_signature(sample, best_pipe.predict_proba(sample)),
                input_example=sample,
                registered_model_name=REGISTERED_MODEL_NAME,
                code_paths=[str(resolve("src"))],
                skops_trusted_types=["numpy.dtype", "src.features.ZeroAsMissing"],
            )
            registered_version = getattr(info, "registered_model_version", None)
            if registered_version is not None:
                client = mlflow.MlflowClient()
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, "candidate", registered_version)
                client.set_model_version_tag(REGISTERED_MODEL_NAME, registered_version, "app_version",
                                             metadata["model_version"])
            return {"mlflow_run_id": parent.info.run_id,
                    "mlflow_registered_version": str(registered_version) if registered_version else None}
    except Exception as exc:  # tracking must never block producing a model
        log.warning("MLflow logging failed (%s). Continuing without tracking.", exc)
        return {"mlflow_run_id": None, "mlflow_registered_version": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run.")
    args = parser.parse_args()
    train(use_mlflow=False if args.no_mlflow else None)


if __name__ == "__main__":
    main()
