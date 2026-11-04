import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.predictor import Predictor
from src.common import FEATURES

VALID = {
    "pregnancies": 2,
    "glucose": 135,
    "blood_pressure": 80,
    "skin_thickness": 25,
    "insulin": 90,
    "bmi": 29.5,
    "diabetes_pedigree_function": 0.45,
    "age": 35,
}


ADMIN = ("admin", "Adm1n-Pass!")
USER = ("alice", "alice-Pass1")


def make_client(model_dir, config, **kwargs) -> TestClient:
    """App with an isolated user database and a bootstrap admin account."""
    db = Path(config["monitoring"]["predictions_log"]).parent.parent / "users.db"
    defaults = {"auth_db_path": db, "auth_secret_key": "test-secret-key-0123456789-abcdefghijkl",
                "bootstrap_admin_username": ADMIN[0], "bootstrap_admin_password": ADMIN[1]}
    defaults.update(log_predictions=True, log_features=False)
    settings = Settings(model_dir=model_dir, config=config, **{**defaults, **kwargs})
    return TestClient(create_app(settings=settings, predictor=Predictor(model_dir)))


def sign_in(c: TestClient, role: str = "user") -> TestClient:
    """Authenticate the client as the admin or as a (self-registered) regular user."""
    username, password = ADMIN if role == "admin" else USER
    if role != "admin":
        c.post("/auth/register", json={"username": username, "password": password})
    r = c.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return c


@pytest.fixture
def client(model_copy, test_config):
    with make_client(model_copy, test_config) as c:
        yield sign_in(c, "user")


@pytest.fixture
def admin_client(model_copy, test_config):
    with make_client(model_copy, test_config) as c:
        yield sign_in(c, "admin")


@pytest.fixture
def no_model_client(tmp_path, test_config):
    with make_client(tmp_path / "empty", test_config) as c:
        yield sign_in(c, "user")


# ----------------------------------------------------------------------------- service endpoints
def test_root(client):
    body = client.get("/").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert "not a medical diagnosis" in body["disclaimer"]


def test_health_and_ready_with_model(client, model_copy):
    version = json.loads((model_copy / "metadata.json").read_text())["model_version"]
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": version, "detail": None}
    assert client.get("/ready").json() == {"ready": True, "model_version": version}


def test_health_reports_503_without_model(no_model_client):
    r = no_model_client.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False
    assert "No trained model found" in r.json()["detail"]
    assert no_model_client.get("/ready").status_code == 503


def test_predict_without_model_never_returns_a_fake_prediction(no_model_client):
    r = no_model_client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "prediction" not in r.json()


# ----------------------------------------------------------------------------- prediction
@pytest.mark.integration
def test_predict_matches_the_real_model(client, model_copy):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    model = joblib.load(model_copy / "model.joblib")
    meta = json.loads((model_copy / "metadata.json").read_text())
    expected = model.predict_proba(pd.DataFrame([VALID], columns=FEATURES, dtype=float))[0, 1]
    assert body["probability"] == pytest.approx(expected, abs=1e-4)
    assert body["prediction"] == int(expected >= meta["decision_threshold"])
    assert body["prediction_label"] == ("Diabetes Risk Indicated" if body["prediction"]
                                        else "Diabetes Risk Not Indicated")
    assert body["model_version"] == meta["model_version"]
    assert "not a medical diagnosis" in body["disclaimer"]
    assert r.headers.get("X-Request-ID")


@pytest.mark.integration
def test_predictions_respond_to_inputs(client):
    low = dict(VALID, glucose=85, bmi=21.0, age=22, pregnancies=0, diabetes_pedigree_function=0.15)
    high = dict(VALID, glucose=195, bmi=42.0, age=55, pregnancies=6, diabetes_pedigree_function=1.2)
    p_low = client.post("/predict", json=low).json()
    p_high = client.post("/predict", json=high).json()
    assert p_high["probability"] > p_low["probability"]
    assert p_low["prediction_label"] == "Diabetes Risk Not Indicated"
    assert "does NOT prove" in p_low["interpretation"]
    assert p_high["prediction_label"] == "Diabetes Risk Indicated"


def test_zero_for_unmeasured_insulin_is_accepted(client):
    assert client.post("/predict", json=dict(VALID, insulin=0, skin_thickness=0)).status_code == 200


# ----------------------------------------------------------------------------- validation
@pytest.mark.parametrize("payload", [
    {k: v for k, v in VALID.items() if k != "glucose"},  # missing field
    dict(VALID, glucose=0),                              # implausible (missing-code) glucose
    dict(VALID, glucose=1000),                           # above limit
    dict(VALID, age=-5),                                 # negative
    dict(VALID, bmi="heavy"),                            # wrong type
    dict(VALID, pregnancies=2.5),                        # non-integer count
    dict(VALID, name="Jane Doe"),                        # unexpected (identifying) field
    {},
])
def test_invalid_requests_are_rejected(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list) and r.json()["detail"][0]["field"]


def test_nan_is_rejected(client):
    body = json.dumps(VALID).replace("29.5", "NaN")
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_oversized_request_is_rejected(client):
    r = client.post("/predict", content="x" * 10_000, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_validation_errors_do_not_echo_values(client):
    r = client.post("/predict", json=dict(VALID, glucose=987654))
    assert "987654" not in r.text


# ----------------------------------------------------------------------------- model info / admin
def test_model_info_reports_version_and_metrics(admin_client, model_copy):
    info = admin_client.get("/model/info").json()
    meta = json.loads((model_copy / "metadata.json").read_text())
    assert info["model_version"] == meta["model_version"]
    assert info["model_name"] == meta["model_name"]
    assert info["test_metrics"]["recall"] > 0
    assert info["dataset"]["source"]
    assert info["known_limitations"]


def test_admin_reload_requires_admin_or_automation_token(model_copy, test_config):
    with make_client(model_copy, test_config) as c:
        assert c.post("/admin/reload").status_code == 401                                   # anonymous
        assert c.post("/admin/reload", headers={"X-Admin-Token": "x"}).status_code == 401  # token not configured
        assert sign_in(c, "user").post("/admin/reload").status_code == 403                 # plain user
    with make_client(model_copy, test_config, admin_token="s3cret-test-token") as c:
        assert c.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        r = c.post("/admin/reload", headers={"X-Admin-Token": "s3cret-test-token"})
        assert r.status_code == 200 and r.json()["reloaded"] is True


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]
