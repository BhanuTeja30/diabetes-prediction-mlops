import json

import pandas as pd
import pytest
from prometheus_client import REGISTRY

from src.common import PROJECT_ROOT
from src.drift_detection import InsufficientDataError, run_drift_check
from tests.test_api import VALID, make_client, sign_in


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


class _client:
    """Signed-in (admin) client context manager with optional approved feature logging."""

    def __init__(self, model_dir, config, log_features=False):
        self._cm = make_client(model_dir, config, log_features=log_features)

    def __enter__(self):
        return sign_in(self._cm.__enter__(), "admin")

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


@pytest.mark.integration
def test_metrics_count_requests_success_failure_and_latency(model_copy, test_config):
    before = {n: sample(n) for n in ("prediction_requests_total", "prediction_success_total",
                                     "prediction_latency_seconds_count")}
    before_fail = sample("prediction_failures_total", {"reason": "validation_error"})
    with _client(model_copy, test_config) as c:
        c.post("/predict", json=VALID)
        c.post("/predict", json=dict(VALID, glucose=-1))
        text = c.get("/metrics").text
    assert sample("prediction_requests_total") - before["prediction_requests_total"] == 2
    assert sample("prediction_success_total") - before["prediction_success_total"] == 1
    assert sample("prediction_latency_seconds_count") - before["prediction_latency_seconds_count"] == 1
    assert sample("prediction_failures_total", {"reason": "validation_error"}) - before_fail == 1
    assert sample("model_loaded") == 1
    for name in ("prediction_requests_total", "prediction_failures_total", "prediction_latency_seconds",
                 "model_loaded", "model_info", "model_load_failures_total"):
        assert name in text


@pytest.mark.integration
def test_metrics_never_contain_health_inputs(model_copy, test_config):
    with _client(model_copy, test_config) as c:
        c.post("/predict", json=dict(VALID, glucose=173, bmi=37.3))
        text = c.get("/metrics").text
    for feature in VALID:
        assert feature not in text
    assert "37.3" not in text


def test_model_load_failure_is_counted(tmp_path, test_config):
    before = sample("model_load_failures_total")
    with _client(tmp_path / "missing", test_config) as c:
        c.get("/health")
    assert sample("model_load_failures_total") - before == 1
    assert sample("model_loaded") == 0


@pytest.mark.integration
def test_prediction_log_contains_no_raw_inputs_by_default(model_copy, test_config):
    with _client(model_copy, test_config) as c:
        c.post("/predict", json=VALID)
        summary = c.get("/monitoring/summary").json()
    log = pd.read_csv(test_config["monitoring"]["predictions_log"])
    assert list(log.columns) == ["time_bucket", "model_version", "prediction", "probability", "latency_ms"]
    assert log["time_bucket"].str.endswith(":00Z").all()  # coarsened to the hour
    assert not (PROJECT_ROOT / test_config["monitoring"]["features_log"]).exists()
    assert summary["successful_predictions"] >= 1
    assert summary["feature_logging_enabled"] is False


@pytest.mark.integration
def test_feature_logging_only_when_explicitly_enabled(model_copy, test_config):
    with _client(model_copy, test_config, log_features=True) as c:
        c.post("/predict", json=VALID)
    log = pd.read_csv(test_config["monitoring"]["features_log"])
    assert "glucose" in log.columns and len(log) == 1
    assert not {"name", "id", "request_id", "ip"} & set(log.columns)


# ----------------------------------------------------------------------------- drift detection
@pytest.fixture
def drift_config(test_config, trained_model_dir):
    processed = json.loads((trained_model_dir / "metadata.json").read_text())["dataset"]["processed_dir"]
    test_config["monitoring"]["reference_path"] = f"{processed}/train.csv"
    return test_config, processed


@pytest.mark.integration
def test_no_drift_alert_on_same_distribution(drift_config, trained_model_dir):
    cfg, processed = drift_config
    status = run_drift_check(f"{processed}/test.csv", config=cfg, model_dir=trained_model_dir)
    assert status["n_drifted"] <= 1
    assert status["feature_drift_alert"] is False
    assert (PROJECT_ROOT / cfg["monitoring"]["status_file"]).exists()


@pytest.mark.integration
def test_drift_alert_on_shifted_inputs(drift_config, trained_model_dir, tmp_path):
    cfg, processed = drift_config
    shifted = pd.read_csv(f"{processed}/test.csv")
    shifted["glucose"] += 40
    shifted["bmi"] += 8
    shifted["age"] += 20
    path = tmp_path / "shifted.csv"
    shifted.to_csv(path, index=False)
    status = run_drift_check(str(path), config=cfg, model_dir=trained_model_dir)
    assert {"glucose", "bmi", "age"} <= set(status["drifted_features"])
    assert status["alert"] is True
    assert list((tmp_path / "drift").glob("drift_report_*.html"))
    # The API exports the alert to Prometheus.
    with _client(trained_model_dir, cfg) as c:
        assert "data_drift_alert 1.0" in c.get("/metrics").text


def test_drift_check_refuses_tiny_batches(test_config, tmp_path, synthetic_df):
    path = tmp_path / "tiny.csv"
    synthetic_df.head(5).to_csv(path, index=False)
    with pytest.raises(InsufficientDataError):
        run_drift_check(str(path), config=test_config)


def test_drift_check_without_any_approved_data(test_config):
    with pytest.raises(InsufficientDataError, match="MONITORING_LOG_FEATURES"):
        run_drift_check(None, config=test_config)


def test_zero_coded_missing_values_are_not_quality_issues(test_config, tmp_path, synthetic_df):
    """0 means "not measured" in these fields; it must not be reported as an out-of-range value."""
    batch = synthetic_df.drop(columns=["outcome"]).copy()
    batch.loc[:9, ["glucose", "blood_pressure", "bmi"]] = 0
    path = tmp_path / "batch.csv"
    batch.to_csv(path, index=False)
    test_config["monitoring"]["reference_path"] = str(path)  # compare with itself: no drift expected
    status = run_drift_check(str(path), config=test_config)
    assert status["quality_details"]["blood_pressure"]["out_of_range"] == 0
    assert not any("out-of-range" in issue for issue in status["quality_issues"])
