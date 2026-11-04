"""Users vs admins: accounts, sign-in, token handling and role enforcement."""

import sqlite3
import time

import pytest

from app.auth import (
    AccountLocked,
    AuthError,
    Conflict,
    InvalidCredentials,
    TokenSigner,
    UserStore,
    hash_password,
    verify_password,
)
from tests.test_api import ADMIN, USER, VALID, make_client, sign_in

ADMIN_ONLY = [
    ("get", "/model/info"),
    ("get", "/monitoring/summary"),
    ("get", "/admin/users"),
    ("post", "/admin/reload"),
    ("get", "/admin/drift-reports/drift_report_20260101T000000Z.html"),
]


# ----------------------------------------------------------------------------- unit: store & tokens
@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users.db")


def test_passwords_are_salted_hashes(store, tmp_path):
    store.create_user("bob", "bob-Pass123")
    row = sqlite3.connect(tmp_path / "users.db").execute("SELECT password_hash FROM users").fetchone()[0]
    assert "bob-Pass123" not in row and row.startswith("pbkdf2_sha256$")
    assert hash_password("same-Pass1") != hash_password("same-Pass1")  # random salt
    assert verify_password("bob-Pass123", row) and not verify_password("wrong-Pass1", row)


@pytest.mark.parametrize("username, password", [
    ("ab", "good-Pass1"),          # username too short
    ("bad name", "good-Pass1"),    # space
    ("carol", "short1"),           # password too short
    ("carol", "onlyletters"),      # no digits / symbols
])
def test_weak_or_invalid_credentials_rejected(store, username, password):
    with pytest.raises(AuthError):
        store.create_user(username, password)


def test_duplicate_usernames_rejected_case_insensitively(store):
    store.create_user("Dave", "dave-Pass1")
    with pytest.raises(Conflict):
        store.create_user("dave", "other-Pass1")


def test_lockout_after_repeated_failures(store):
    store.create_user("erin", "erin-Pass1")
    for _ in range(5):
        with pytest.raises(InvalidCredentials):
            store.authenticate("erin", "wrong-Pass1")
    with pytest.raises(AccountLocked):
        store.authenticate("erin", "erin-Pass1")  # even the right password is refused while locked


def test_last_admin_is_protected(store):
    store.create_user("root", "root-Pass1", role="admin")
    with pytest.raises(Conflict):
        store.update_user("root", role="user")
    with pytest.raises(Conflict):
        store.update_user("root", is_active=False)
    with pytest.raises(Conflict):
        store.delete_user("root")
    store.create_user("root2", "root-Pass2", role="admin")
    store.update_user("root", role="user")  # fine once another admin exists


def test_bootstrap_admin_only_when_no_admin_exists(store):
    assert store.ensure_admin("boss", "boss-Pass1") == "boss"
    assert store.ensure_admin("boss2", "boss-Pass2") is None
    assert [u["username"] for u in store.list_users()] == ["boss"]


def test_tokens_reject_tampering_and_expiry():
    signer = TokenSigner("k" * 40, ttl_seconds=60)
    token, _ = signer.issue("frank", "user")
    assert signer.verify(token)["sub"] == "frank"
    payload, sig = token.split(".")
    forged_payload = TokenSigner("k" * 40, 60).issue("frank", "admin")[0].split(".")[0]
    with pytest.raises(InvalidCredentials):
        signer.verify(f"{forged_payload}.{sig}")
    with pytest.raises(InvalidCredentials):
        TokenSigner("x" * 40, 60).verify(token)  # different secret
    with pytest.raises(InvalidCredentials, match="expired"):
        signer.verify(token, now=time.time() + 120)


# ----------------------------------------------------------------------------- API: roles
@pytest.fixture
def app_client(model_copy, test_config):
    with make_client(model_copy, test_config) as c:
        yield c


def test_prediction_requires_sign_in(app_client):
    r = app_client.post("/predict", json=VALID)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_public_endpoints_need_no_sign_in(app_client):
    for path in ("/", "/health", "/ready", "/metrics"):
        assert app_client.get(path).status_code == 200


def test_user_can_predict_but_not_use_admin_endpoints(app_client):
    c = sign_in(app_client, "user")
    assert c.get("/auth/me").json()["role"] == "user"
    assert c.post("/predict", json=VALID).status_code == 200
    for method, path in ADMIN_ONLY:
        assert getattr(c, method)(path).status_code == 403, path
    assert c.post("/admin/users", json={"username": "x1y", "password": "abc-Pass1"}).status_code == 403


@pytest.mark.parametrize("method, path", ADMIN_ONLY)
def test_admin_endpoints_reject_anonymous(app_client, method, path):
    assert getattr(app_client, method)(path).status_code == 401


def test_admin_can_access_admin_endpoints(app_client):
    c = sign_in(app_client, "admin")
    assert c.get("/auth/me").json()["role"] == "admin"
    assert c.post("/predict", json=VALID).status_code == 200
    assert c.get("/model/info").status_code == 200
    assert c.get("/monitoring/summary").status_code == 200
    assert c.post("/admin/reload").json()["reloaded"] is True


def test_self_registration_always_creates_plain_users(app_client):
    r = app_client.post("/auth/register", json={"username": "mallory", "password": "mallory-Pass1", "role": "admin"})
    assert r.status_code == 422  # 'role' is not accepted on self-registration
    r = app_client.post("/auth/register", json={"username": "mallory", "password": "mallory-Pass1"})
    assert r.status_code == 201 and r.json()["role"] == "user"


def test_self_registration_can_be_disabled(model_copy, test_config):
    with make_client(model_copy, test_config, allow_self_registration=False) as c:
        assert c.post("/auth/register", json={"username": "gina", "password": "gina-Pass1"}).status_code == 403
        assert c.get("/").json()["self_registration"] is False


def test_wrong_password_and_unknown_user_get_same_message(app_client):
    sign_in(app_client, "user")
    app_client.headers.pop("Authorization")
    a = app_client.post("/auth/login", json={"username": USER[0], "password": "wrong-Pass9"})
    b = app_client.post("/auth/login", json={"username": "nobody", "password": "wrong-Pass9"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_password_never_echoed_in_validation_errors(app_client):
    r = app_client.post("/auth/login", json={"username": "ab", "password": "Secr3t-Value", "extra": 1})
    assert r.status_code == 422 and "Secr3t-Value" not in r.text


# ----------------------------------------------------------------------------- API: admin user management
def test_admin_manages_accounts_and_changes_apply_immediately(app_client):
    user_token = sign_in(app_client, "user").headers.pop("Authorization")
    admin = sign_in(app_client, "admin")

    r = admin.post("/admin/users", json={"username": "henry", "password": "henry-Pass1", "role": "admin"})
    assert r.status_code == 201 and r.json()["role"] == "admin"
    names = {u["username"]: u["role"] for u in admin.get("/admin/users").json()}
    assert names == {"admin": "admin", "alice": "user", "henry": "admin"}
    assert all("password" not in k for u in admin.get("/admin/users").json() for k in u)

    # Disable alice: her existing token stops working at once.
    assert admin.patch("/admin/users/alice", json={"is_active": False}).json()["is_active"] is False
    assert app_client.post("/predict", json=VALID, headers={"Authorization": user_token}).status_code == 401

    # Promote then re-enable alice: her old token now has admin rights (role read from the database).
    admin.patch("/admin/users/alice", json={"is_active": True, "role": "admin"})
    assert app_client.get("/admin/users", headers={"Authorization": user_token}).status_code == 200

    # Password reset.
    admin.patch("/admin/users/henry", json={"password": "henry-New-Pass2"})
    admin_header = admin.headers.pop("Authorization")
    assert app_client.post("/auth/login", json={"username": "henry", "password": "henry-New-Pass2"}).status_code == 200
    admin.headers["Authorization"] = admin_header

    assert admin.delete("/admin/users/henry").status_code == 204
    assert admin.delete("/admin/users/admin").status_code == 409  # cannot delete yourself
    assert admin.delete("/admin/users/nobody").status_code == 404


def test_admin_cannot_remove_last_admin(app_client):
    admin = sign_in(app_client, "admin")
    r = admin.patch(f"/admin/users/{ADMIN[0]}", json={"role": "user"})
    assert r.status_code == 409 and "last active admin" in r.json()["detail"]


@pytest.mark.integration
def test_admin_drift_check_upload_and_report(model_copy, test_config, trained_model_dir):
    import json

    processed = json.loads((trained_model_dir / "metadata.json").read_text())["dataset"]["processed_dir"]
    test_config["monitoring"]["reference_path"] = f"{processed}/train.csv"
    with make_client(model_copy, test_config) as c:
        sign_in(c, "user")
        with open(f"{processed}/test.csv", "rb") as fh:
            assert c.post("/admin/drift-check", files={"file": ("b.csv", fh, "text/csv")}).status_code == 403
        sign_in(c, "admin")
        assert c.post("/admin/drift-check", files={"file": ("b.txt", b"x", "text/plain")}).status_code == 400
        with open(f"{processed}/test.csv", "rb") as fh:
            r = c.post("/admin/drift-check", files={"file": ("b.csv", fh, "text/csv")})
        assert r.status_code == 200
        report = r.json()["report_html"]
        assert c.get(f"/admin/drift-reports/{report}").text.lstrip().lower().startswith("<")
        assert c.get("/admin/drift-reports/..%2F..%2Fsecret.html").status_code in (400, 404)
