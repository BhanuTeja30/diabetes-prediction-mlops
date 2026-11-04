"""Controlled retraining workflow (manual or approved schedule). It NEVER promotes by itself.

Retraining is only allowed on an *approved* dataset with a manifest that documents its
provenance. User-submitted health data and model predictions are never used as labels.

Expected layout of an approved dataset:
    data/approved/<release>/dataset.csv     # 8 features + outcome (verified labels)
    data/approved/<release>/manifest.yaml

manifest.yaml fields:
    dataset_name: pima-diabetes
    version: "2026-10-01"
    source: "<where the data comes from / licence>"
    sha256: "<sha256 of dataset.csv>"
    approved_by: "<data steward>"
    approved_at: "2026-10-01"
    labels_verified: true            # outcomes are confirmed diagnoses, not model predictions
    contains_user_submissions: false # true only with documented consent (consent_reference)
    consent_reference: null

Steps: verify manifest + checksum -> validate quality -> prepare/split -> train (MLflow)
-> evaluate on held-out test -> compare with production -> write report.
Promotion afterwards requires: python -m src.promote_model --approve --approved-by NAME

Usage:
    python -m src.retrain --dataset-dir data/approved/2026-10-01
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.common import get_logger, load_config, load_params, resolve, sha256_file, write_json
from src.data_processing import DataValidationError, load_raw, prepare, validate_dataset
from src.evaluate import evaluate_model_dir
from src.promote_model import assess_candidate
from src.train import train

log = get_logger("retrain")

REQUIRED_MANIFEST_KEYS = ("dataset_name", "version", "source", "sha256", "approved_by", "approved_at",
                          "labels_verified", "contains_user_submissions")


class RetrainingRejected(RuntimeError):
    pass


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Provenance, approval and integrity checks. Raises RetrainingRejected on any failure."""
    manifest_path, data_path = dataset_dir / "manifest.yaml", dataset_dir / "dataset.csv"
    if not manifest_path.exists() or not data_path.exists():
        raise RetrainingRejected(f"{dataset_dir} must contain dataset.csv and manifest.yaml.")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    missing = [k for k in REQUIRED_MANIFEST_KEYS if manifest.get(k) in (None, "")]
    if missing:
        raise RetrainingRejected(f"Manifest missing required fields: {missing}")
    if manifest["labels_verified"] is not True:
        raise RetrainingRejected("Outcome labels are not verified. Predictions must never be used as labels.")
    if manifest["contains_user_submissions"] and not manifest.get("consent_reference"):
        raise RetrainingRejected("Dataset contains user submissions without a documented consent reference.")
    actual = sha256_file(data_path)
    if actual != manifest["sha256"]:
        raise RetrainingRejected(f"Checksum mismatch: manifest {manifest['sha256']} vs file {actual}.")
    return manifest


def retrain(dataset_dir: str | Path, use_mlflow: bool | None = None) -> dict[str, Any]:
    dataset_dir = resolve(dataset_dir)
    started = datetime.now(UTC)
    report: dict[str, Any] = {"started_at": started.isoformat(timespec="seconds"), "dataset_dir": str(dataset_dir)}
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    report_path = f"reports/retraining/retraining_{stamp}.json"

    try:
        manifest = verify_dataset(dataset_dir)
        report["manifest"] = manifest
        report["quality"] = validate_dataset(load_raw(dataset_dir / "dataset.csv"))
    except (RetrainingRejected, DataValidationError) as exc:
        report.update({"status": "rejected", "reason": str(exc)})
        write_json(report_path, report)
        log.error("Retraining rejected: %s (report: %s)", exc, report_path)
        raise RetrainingRejected(str(exc)) from exc

    params = load_params()
    processed_dir = resolve(f"data/retraining/{manifest['version']}/processed")
    report["preparation"] = prepare(params, raw_path=dataset_dir / "dataset.csv", out_dir=processed_dir,
                                    figures_dir=f"reports/retraining/figures_{manifest['version']}")["splits"]

    config = load_config()
    metadata = train(params, processed_dir=processed_dir, use_mlflow=use_mlflow)
    evaluation = evaluate_model_dir(config["model"]["candidate_dir"])
    assessment = assess_candidate(config)
    report.update({
        "status": "candidate_ready" if assessment["eligible"] else "candidate_failed",
        "candidate_version": metadata["model_version"],
        "candidate_model": metadata["model_name"],
        "test_metrics": evaluation["test_metrics"],
        "assessment": assessment,
        "next_step": (
            "Review this report, then run: python -m src.promote_model --approve --approved-by \"<name>\""
            if assessment["eligible"] else
            "Candidate failed acceptance/comparison checks. Production model unchanged."
        ),
    })
    write_json(report_path, report)
    log.info("Retraining finished: %s. Report: %s", report["status"], report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Folder with dataset.csv and manifest.yaml.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    try:
        retrain(args.dataset_dir, use_mlflow=False if args.no_mlflow else None)
    except RetrainingRejected as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
