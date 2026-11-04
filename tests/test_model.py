import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.common import FEATURES, TARGET, load_config, sha256_file
from src.data_processing import clean, split
from src.evaluate import check_acceptance, compute_metrics, tune_threshold
from src.promote_model import PromotionError, assess_candidate, promote, rollback
from src.retrain import RetrainingRejected, verify_dataset
from src.train import build_pipelines, select_model

ZERO_COLS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


# ----------------------------------------------------------------------------- metrics
def test_compute_metrics_known_values():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.2]
    m = compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)
    assert 0 <= m["brier_score"] <= 1 and 0 <= m["roc_auc"] <= 1


def test_tune_threshold_reaches_target_recall():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    t = tune_threshold(y, p, target_recall=0.8)
    assert compute_metrics(y, p, t)["recall"] >= 0.8
    # The highest such threshold: any higher threshold loses the target recall.
    higher = np.unique(p)[np.unique(p) > t]
    if len(higher):
        assert compute_metrics(y, p, higher.min())["recall"] < 0.8


def test_select_model_uses_specificity_at_target_recall_not_accuracy():
    def res(recall, spec, brier, acc):
        return {"validation": {"recall": recall, "specificity": spec, "brier_score": brier, "accuracy": acc}}
    results = {
        "logistic_regression": res(0.85, 0.60, 0.15, 0.95),  # best accuracy, worse specificity
        "random_forest": res(0.82, 0.75, 0.16, 0.70),
        "xgboost": res(0.60, 0.95, 0.10, 0.90),             # misses too many positives
    }
    best, reason = select_model(results, target_recall=0.8)
    assert best == "random_forest"
    assert "Accuracy was not used" in reason


def test_select_model_fails_when_no_model_reaches_recall():
    with pytest.raises(RuntimeError):
        select_model({"a": {"validation": {"recall": 0.1, "specificity": 1, "brier_score": 0.1}}}, 0.8)


def test_acceptance_criteria():
    criteria = load_config()["acceptance_criteria"]
    good = {"recall": 0.8, "roc_auc": 0.85, "brier_score": 0.15, "specificity": 0.7}
    assert check_acceptance(good, criteria)["passed"]
    bad = dict(good, recall=0.4)
    result = check_acceptance(bad, criteria)
    assert not result["passed"] and result["checks"]["min_test_recall"] is False


# ----------------------------------------------------------------------------- training
@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_each_pipeline_trains_and_outputs_probabilities(synthetic_df, params, name):
    df, _ = clean(synthetic_df, ZERO_COLS)
    train, val, _ = split(df, 0.2, 0.2, seed=0)
    pipe = build_pipelines(params, train[TARGET])[name].fit(train[FEATURES], train[TARGET])
    prob = pipe.predict_proba(val[FEATURES])[:, 1]
    assert prob.shape == (len(val),)
    assert np.all((prob >= 0) & (prob <= 1))
    assert compute_metrics(val[TARGET], prob, 0.5)["roc_auc"] > 0.6  # learns the synthetic signal


# ----------------------------------------------------------------------------- real artifact
@pytest.mark.integration
def test_real_artifact_loads_and_predicts(trained_model_dir):
    model = joblib.load(trained_model_dir / "model.joblib")
    meta = json.loads((trained_model_dir / "metadata.json").read_text())
    evaluation = json.loads((trained_model_dir / "evaluation.json").read_text())
    assert meta["features"] == FEATURES
    assert 0 < meta["decision_threshold"] < 1
    assert evaluation["test_metrics"]["n"] > 100
    row = pd.DataFrame([[2, 135, 80, 25, 90, 29.5, 0.45, 35]], columns=FEATURES, dtype=float)
    assert 0 <= model.predict_proba(row)[0, 1] <= 1


@pytest.mark.integration
def test_real_model_meets_acceptance_criteria(trained_model_dir):
    evaluation = json.loads((trained_model_dir / "evaluation.json").read_text())
    assert check_acceptance(evaluation["test_metrics"], load_config()["acceptance_criteria"])["passed"]


# ----------------------------------------------------------------------------- promotion
@pytest.fixture
def governance_config(tmp_path, trained_model_dir):
    import shutil

    cfg = copy.deepcopy(load_config())
    cfg["model"].update(candidate_dir=str(tmp_path / "candidate"), production_dir=str(tmp_path / "production"),
                        archive_dir=str(tmp_path / "archive"), reports_dir=str(tmp_path / "reports"))
    shutil.copytree(trained_model_dir, tmp_path / "candidate")
    return cfg


@pytest.mark.integration
def test_promotion_requires_explicit_approval(governance_config, tmp_path):
    result = promote(approved_by=None, approve=False, config=governance_config)
    assert result["eligible"] and not result["promoted"]
    assert not (tmp_path / "production" / "model.joblib").exists()
    with pytest.raises(PromotionError, match="approved-by"):
        promote(approved_by="  ", approve=True, config=governance_config)
    assert promote(approved_by="Tester", approve=True, config=governance_config)["promoted"]
    meta = json.loads((tmp_path / "production" / "metadata.json").read_text())
    assert meta["promotion"]["approved_by"] == "Tester"


@pytest.mark.integration
def test_failing_candidate_keeps_production_and_rollback_works(governance_config, tmp_path):
    promote(approved_by="Tester", approve=True, config=governance_config)
    prod_bytes = (tmp_path / "production" / "model.joblib").read_bytes()

    strict = copy.deepcopy(governance_config)
    strict["acceptance_criteria"]["min_test_roc_auc"] = 0.999
    assert not assess_candidate(strict)["eligible"]
    with pytest.raises(PromotionError):
        promote(approved_by="Tester", approve=True, config=strict)
    assert (tmp_path / "production" / "model.joblib").read_bytes() == prod_bytes

    # Promote again (archives the current production), then roll back to it.
    promote(approved_by="Tester", approve=True, config=governance_config)
    version = json.loads((tmp_path / "production" / "metadata.json").read_text())["model_version"]
    assert (tmp_path / "archive" / version / "model.joblib").exists()
    assert rollback("Tester", config=governance_config) == version
    log = json.loads((tmp_path / "promotion_log.json").read_text())
    assert [e["action"] for e in log] == ["promote", "promote", "rollback"]


def test_rollback_without_archive_fails(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["model"].update(archive_dir=str(tmp_path / "archive"), production_dir=str(tmp_path / "prod"))
    with pytest.raises(PromotionError, match="No archived model"):
        rollback("Tester", config=cfg)


# ----------------------------------------------------------------------------- retraining approval
def _approved_dataset(tmp_path, synthetic_df, **overrides):
    folder = tmp_path / "approved"
    folder.mkdir()
    synthetic_df.to_csv(folder / "dataset.csv", index=False)
    manifest = {"dataset_name": "synthetic", "version": "t1", "source": "unit test", "approved_by": "Steward",
                "approved_at": "2026-10-29", "labels_verified": True, "contains_user_submissions": False,
                "sha256": sha256_file(folder / "dataset.csv")}
    manifest.update(overrides)
    (folder / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return folder


def test_retraining_accepts_verified_dataset(tmp_path, synthetic_df):
    assert verify_dataset(_approved_dataset(tmp_path, synthetic_df))["version"] == "t1"


@pytest.mark.parametrize("override, message", [
    ({"labels_verified": False}, "not verified"),
    ({"sha256": "0" * 64}, "Checksum mismatch"),
    ({"contains_user_submissions": True}, "consent"),
    ({"approved_by": ""}, "approved_by"),
])
def test_retraining_rejects_unapproved_datasets(tmp_path, synthetic_df, override, message):
    with pytest.raises(RetrainingRejected, match=message):
        verify_dataset(_approved_dataset(tmp_path, synthetic_df, **override))


def test_retraining_rejects_missing_manifest(tmp_path):
    with pytest.raises(RetrainingRejected):
        verify_dataset(tmp_path)
