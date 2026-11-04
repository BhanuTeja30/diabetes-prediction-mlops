"""Controlled promotion of a candidate model to production (and rollback).

A candidate becomes the production model only when ALL of these hold:
  1. it has been evaluated on the held-out test set (src.evaluate),
  2. it passes the absolute acceptance criteria in configs/model_config.yaml,
  3. it is not meaningfully worse than the current production model on the same test set,
  4. a named person explicitly approves it with --approve --approved-by NAME.

The previous production model is archived under models/archive/<version>/ for rollback.

Usage:
    python -m src.promote_model --check                          # dry run: report only
    python -m src.promote_model --approve --approved-by "Dr. A"  # promote if checks pass
    python -m src.promote_model --rollback --approved-by "Dr. A" # restore last archived model
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.common import get_logger, load_config, read_json, resolve, write_json
from src.evaluate import check_acceptance, compute_metrics, load_split

log = get_logger("promote_model")
ARTIFACT_FILES = ("model.joblib", "metadata.json", "evaluation.json")


class PromotionError(RuntimeError):
    pass


def compare_with_production(candidate_dir: Path, production_dir: Path, criteria: dict[str, float],
                            processed_dir: str | Path | None = None) -> dict[str, Any]:
    """Score candidate and production on the SAME test split and check for regressions."""
    if not (production_dir / "model.joblib").exists():
        return {"production_exists": False, "passed": True, "note": "No production model yet."}

    cand_meta, prod_meta = read_json(candidate_dir / "metadata.json"), read_json(production_dir / "metadata.json")
    # Both models are scored on the candidate's held-out test split.
    X_test, y_test = load_split("test", processed_dir or cand_meta["dataset"].get("processed_dir", "data/processed"))
    cand = compute_metrics(y_test, joblib.load(candidate_dir / "model.joblib").predict_proba(X_test)[:, 1],
                           cand_meta["decision_threshold"])
    prod = compute_metrics(y_test, joblib.load(production_dir / "model.joblib").predict_proba(X_test)[:, 1],
                           prod_meta["decision_threshold"])
    checks = {
        "roc_auc_not_regressed": cand["roc_auc"] >= prod["roc_auc"] - criteria["max_roc_auc_regression"],
        "recall_not_regressed": cand["recall"] >= prod["recall"] - criteria["max_recall_regression"],
    }
    return {
        "production_exists": True,
        "production_version": prod_meta["model_version"],
        "candidate": {k: cand[k] for k in ("recall", "specificity", "roc_auc", "brier_score")},
        "production": {k: prod[k] for k in ("recall", "specificity", "roc_auc", "brier_score")},
        "checks": checks,
        "passed": all(checks.values()),
    }


def assess_candidate(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    cand_dir = resolve(config["model"]["candidate_dir"])
    prod_dir = resolve(config["model"]["production_dir"])
    if not (cand_dir / "model.joblib").exists():
        raise PromotionError(f"No candidate model in {cand_dir}. Run `dvc repro` or `python -m src.train`.")
    meta = read_json(cand_dir / "metadata.json")
    if not (cand_dir / "evaluation.json").exists():
        raise PromotionError("Candidate has not been evaluated on the test set. Run `python -m src.evaluate`.")
    criteria = config["acceptance_criteria"]
    # Re-check against the current criteria rather than trusting the stored verdict.
    acceptance = check_acceptance(read_json(cand_dir / "evaluation.json")["test_metrics"], criteria)
    comparison = compare_with_production(cand_dir, prod_dir, criteria)
    return {
        "candidate_version": meta["model_version"],
        "candidate_model": meta["model_name"],
        "acceptance": acceptance,
        "comparison": comparison,
        "eligible": acceptance["passed"] and comparison["passed"],
    }


def promote(approved_by: str | None, approve: bool, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    assessment = assess_candidate(config)
    _write_report(assessment, config)
    if not assessment["eligible"]:
        raise PromotionError("Candidate failed evaluation checks; production model is unchanged. "
                             f"Details: {assessment}")
    if not approve:
        log.info("Candidate v%s is eligible. Re-run with --approve --approved-by NAME to promote.",
                 assessment["candidate_version"])
        return {"promoted": False, **assessment}
    if not approved_by or not approved_by.strip():
        raise PromotionError("Promotion requires --approved-by NAME (explicit human approval).")

    cand_dir = resolve(config["model"]["candidate_dir"])
    prod_dir = resolve(config["model"]["production_dir"])
    archive_root = resolve(config["model"]["archive_dir"])
    previous = None
    if (prod_dir / "model.joblib").exists():
        previous = read_json(prod_dir / "metadata.json")["model_version"]
        dest = archive_root / previous
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(prod_dir, dest)
        log.info("Archived production v%s to %s", previous, dest)

    prod_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        shutil.copy2(cand_dir / name, prod_dir / name)
    meta = read_json(prod_dir / "metadata.json")
    meta["promotion"] = {"approved_by": approved_by.strip(),
                         "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                         "previous_version": previous}
    write_json(prod_dir / "metadata.json", meta)
    _set_mlflow_production_alias(meta)
    _append_log(config, {"action": "promote", "version": meta["model_version"], "previous_version": previous,
                 **meta["promotion"]})
    log.info("Promoted v%s to production (approved by %s). Restart or POST /admin/reload on the API.",
             meta["model_version"], approved_by)
    return {"promoted": True, **assessment}


def rollback(approved_by: str, version: str | None = None, config: dict[str, Any] | None = None) -> str:
    config = config or load_config()
    if not approved_by or not approved_by.strip():
        raise PromotionError("Rollback requires --approved-by NAME.")
    archive_root = resolve(config["model"]["archive_dir"])
    prod_dir = resolve(config["model"]["production_dir"])
    versions = sorted((p.name for p in archive_root.glob("*") if (p / "model.joblib").exists()),
                      key=lambda v: tuple(int(x) for x in v.split(".")))
    if not versions:
        raise PromotionError("No archived model available for rollback.")
    target = version or versions[-1]
    if target not in versions:
        raise PromotionError(f"Version {target} not in archive {versions}.")
    current = read_json(prod_dir / "metadata.json")["model_version"] if (prod_dir / "metadata.json").exists() else None
    for name in ARTIFACT_FILES:
        if (archive_root / target / name).exists():
            shutil.copy2(archive_root / target / name, prod_dir / name)
    _append_log(config, {"action": "rollback", "version": target, "previous_version": current,
                 "approved_by": approved_by.strip(), "at": datetime.now(UTC).isoformat(timespec="seconds")})
    log.info("Rolled production back from v%s to v%s.", current, target)
    return target


def _write_report(assessment: dict[str, Any], config: dict[str, Any]) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = resolve(config["model"].get("reports_dir", "reports/evaluation"))
    write_json(reports_dir / f"promotion_check_{stamp}.json", assessment)


def _append_log(config: dict[str, Any], entry: dict[str, Any]) -> None:
    # Audit trail of promotions / rollbacks, next to the archive (models/promotion_log.json by default).
    path = resolve(config["model"]["archive_dir"]).parent / "promotion_log.json"
    history = read_json(path) if path.exists() else []
    history.append(entry)
    write_json(path, history)


def _set_mlflow_production_alias(meta: dict[str, Any]) -> None:
    version = meta.get("mlflow_registered_version")
    if not version:
        return
    try:
        from src.tracking import REGISTERED_MODEL_NAME, mlflow_enabled, setup

        if not mlflow_enabled():
            return
        client = setup().MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "approved_by", meta["promotion"]["approved_by"])
    except Exception as exc:
        log.warning("Could not update MLflow 'production' alias: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Only assess the candidate (default).")
    group.add_argument("--approve", action="store_true", help="Promote the candidate if all checks pass.")
    group.add_argument("--rollback", action="store_true", help="Restore an archived production model.")
    parser.add_argument("--approved-by", default=None, help="Name of the person approving the change.")
    parser.add_argument("--version", default=None, help="Archived version to roll back to (default: latest).")
    args = parser.parse_args()
    try:
        if args.rollback:
            rollback(args.approved_by, args.version)
        else:
            result = promote(args.approved_by, approve=args.approve)
            log.info("Assessment: %s", {k: result[k] for k in ("candidate_version", "eligible", "promoted")})
    except PromotionError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
