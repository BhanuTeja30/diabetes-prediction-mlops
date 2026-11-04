"""Evaluation stage: score the candidate model on the held-out test set.

The test split is used ONLY here (never for training, threshold tuning or model
selection), so the numbers reported are an honest estimate of generalisation.

Usage:
    python -m src.evaluate                       # evaluate models/candidate
    python -m src.evaluate --model-dir models/production
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common import FEATURES, TARGET, get_logger, load_config, read_json, resolve, write_json  # noqa: E402

log = get_logger("evaluate")


def compute_metrics(y_true, y_prob, threshold: float) -> dict[str, Any]:
    """Screening-oriented metrics at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_threshold(y_true, y_prob, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall.

    Picking the *highest* such threshold keeps as many true negatives as possible
    (best specificity) while guaranteeing the required sensitivity on validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = y_true.sum()
    if n_pos == 0:
        raise ValueError("Cannot tune a threshold without positive examples.")
    for t in np.unique(y_prob)[::-1]:
        if ((y_prob >= t) & (y_true == 1)).sum() / n_pos >= target_recall:
            return float(t)
    return float(np.min(y_prob))


def check_acceptance(metrics: dict[str, Any], criteria: dict[str, float]) -> dict[str, Any]:
    """Compare test metrics with the absolute acceptance criteria from model_config.yaml."""
    checks = {
        "min_test_recall": metrics["recall"] >= criteria["min_test_recall"],
        "min_test_roc_auc": metrics["roc_auc"] >= criteria["min_test_roc_auc"],
        "max_test_brier_score": metrics["brier_score"] <= criteria["max_test_brier_score"],
        "min_test_specificity": metrics["specificity"] >= criteria["min_test_specificity"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def save_plots(y_true, y_prob, threshold: float, out_dir: Path, title: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No risk", "Risk"], ax=ax,
                                            colorbar=False)
    ax.set_title(f"{title}: confusion matrix (t={threshold:.2f})")
    paths.append(_save(fig, out_dir / "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{title}: ROC curve")
    paths.append(_save(fig, out_dir / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{title}: precision-recall curve")
    paths.append(_save(fig, out_dir / "precision_recall_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"{title}: calibration")
    ax.legend()
    paths.append(_save(fig, out_dir / "calibration_curve.png"))
    return paths


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.relative_to(resolve(".")).as_posix() if path.is_relative_to(resolve(".")) else str(path)


def load_split(name: str, processed_dir: str | Path = "data/processed") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(resolve(processed_dir) / f"{name}.csv")
    return df[FEATURES], df[TARGET]


def evaluate_model_dir(model_dir: str | Path, processed_dir: str | Path | None = None,
                       out_dir: str | Path = "reports/evaluation") -> dict[str, Any]:
    model_dir = resolve(model_dir)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model artifact at {model_path}. Run `python -m src.train` first.")
    meta_path = model_dir / "metadata.json"
    metadata = read_json(meta_path)
    model = joblib.load(model_path)
    threshold = metadata["decision_threshold"]
    # Evaluate on the test split that belongs to the data the model was trained on.
    processed_dir = processed_dir or metadata["dataset"].get("processed_dir", "data/processed")

    X_test, y_test = load_split("test", processed_dir)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, prob, threshold)
    acceptance = check_acceptance(metrics, load_config()["acceptance_criteria"])
    plots = save_plots(y_test, prob, threshold, resolve(out_dir), metadata["model_name"])

    # Written next to (not into) metadata.json so the DVC train-stage output is never mutated.
    write_json(model_dir / "evaluation.json", {"test_metrics": metrics, "acceptance": acceptance,
                                               "processed_dir": str(processed_dir)})
    report = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "test_metrics": metrics,
        "acceptance": acceptance,
        "plots": plots,
    }
    write_json(resolve(out_dir) / "test_metrics.json", report)
    _log_to_mlflow(metadata, metrics, acceptance, plots)

    log.info("Test metrics (%s v%s): recall=%.3f specificity=%.3f roc_auc=%.3f brier=%.3f",
             metadata["model_name"], metadata["model_version"], metrics["recall"], metrics["specificity"],
             metrics["roc_auc"], metrics["brier_score"])
    log.info("Acceptance criteria %s: %s", "PASSED" if acceptance["passed"] else "FAILED", acceptance["checks"])
    return report


def _log_to_mlflow(metadata, metrics, acceptance, plots) -> None:
    run_id = metadata.get("mlflow_run_id")
    if not run_id:
        return
    try:
        from src.tracking import mlflow_client_run

        with mlflow_client_run(run_id) as mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metric("test_acceptance_passed", float(acceptance["passed"]))
            for p in plots:
                mlflow.log_artifact(str(resolve(p)), artifact_path="test_evaluation")
    except Exception as exc:  # MLflow is optional: never break evaluation on tracking errors
        log.warning("Could not log test metrics to MLflow: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=load_config()["model"]["candidate_dir"])
    args = parser.parse_args()
    evaluate_model_dir(args.model_dir)


if __name__ == "__main__":
    main()
