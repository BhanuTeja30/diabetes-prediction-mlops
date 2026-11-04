"""Streamlit frontend for the diabetes-risk prediction service.

Everyone signs in. Users see Home and Prediction; admins additionally see Model information,
Monitoring, Drift check and User management. The API enforces these roles; the UI only
hides pages a role cannot use.

Run locally (with the API running on API_URL, default http://localhost:8000):
    streamlit run app/frontend.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `streamlit run app/frontend.py` from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.config import DISCLAIMER, get_settings  # noqa: E402
from app.schemas import PredictionRequest, PredictionResponse  # noqa: E402

settings = get_settings()
API_URL = settings.api_url.rstrip("/")
LIMITS = settings.input_limits
TIMEOUT = 15

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- session & API helpers
def session() -> dict | None:
    s = st.session_state.get("auth")
    if s and s["expires_at"] <= time.time():
        st.session_state.pop("auth", None)
        st.session_state["flash"] = "Your session expired. Please sign in again."
        return None
    return s


def sign_out(message: str | None = None) -> None:
    st.session_state.pop("auth", None)
    if message:
        st.session_state["flash"] = message


def api(method: str, path: str, *, timeout: float = TIMEOUT, **kwargs) -> requests.Response | None:
    """Call the API with the signed-in user's token. Returns None (after showing an error) if unreachable."""
    headers = kwargs.pop("headers", {})
    if s := session():
        headers["Authorization"] = f"Bearer {s['token']}"
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=timeout, **kwargs)
    except requests.RequestException:
        st.error(f"Cannot reach the prediction service at {API_URL}. Is the API running?")
        return None
    if r.status_code == 401 and session():
        sign_out("Your session is no longer valid. Please sign in again.")
        st.rerun()
    return r


def error_text(r: requests.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        return f"HTTP {r.status_code}"
    if isinstance(detail, list):
        return "; ".join(f"{e.get('field')}: {e.get('message')}" for e in detail)
    return str(detail)


def show_disclaimer() -> None:
    st.warning(f"**Medical disclaimer:** {DISCLAIMER}", icon="⚠️")


# --------------------------------------------------------------------------- sign-in page
def page_sign_in() -> None:
    st.title("🩺 AI-Powered Diabetes Risk Prediction")
    st.markdown("Please sign in to use the risk estimator. New users can create an account.")
    if msg := st.session_state.pop("flash", None):
        st.info(msg)

    info = api("GET", "/")
    allow_signup = bool(info is not None and info.ok and info.json().get("self_registration"))
    tabs = st.tabs(["Sign in", "Create account"] if allow_signup else ["Sign in"])

    with tabs[0], st.form("login"):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        if st.form_submit_button("Sign in", type="primary"):
            r = api("POST", "/auth/login", json={"username": username, "password": password})
            if r is not None and r.ok:
                body = r.json()
                st.session_state["auth"] = {"token": body["access_token"], "username": body["username"],
                                            "role": body["role"], "expires_at": body["expires_at"]}
                st.rerun()
            elif r is not None:
                st.error(error_text(r))

    if allow_signup:
        with tabs[1], st.form("register"):
            st.caption("Accounts are for regular users. Administrator accounts are created by an admin.")
            new_user = st.text_input("Choose a username", help="3–32 characters: letters, digits, . _ -")
            new_pw = st.text_input("Choose a password", type="password",
                                   help="At least 8 characters, mixing letters with numbers or symbols.")
            new_pw2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create account"):
                if new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    r = api("POST", "/auth/register", json={"username": new_user, "password": new_pw})
                    if r is not None and r.status_code == 201:
                        st.success("Account created. You can now sign in on the 'Sign in' tab.")
                    elif r is not None:
                        st.error(error_text(r))
    show_disclaimer()


# --------------------------------------------------------------------------- user pages
def page_home() -> None:
    s = session()
    st.title("🩺 AI-Powered Diabetes Risk Prediction")
    st.markdown(f"Welcome, **{s['username']}**.")
    st.markdown(
        """
This application estimates the **risk of diabetes** from eight routine health measurements using a
machine-learning model trained on the *Pima Indians Diabetes Database*.

**How to use it**
1. Open **Prediction** in the sidebar.
2. Enter the eight health values (hover the ❔ icons for help). If skin-fold thickness or insulin
   were not measured, enter **0** – the model handles them as missing.
3. Click **Predict**. The result and an estimated probability appear immediately.

**What the result means** – the output is a *model-estimated risk*, not a diagnosis. A low-risk
result does not rule out diabetes, and a high-risk result does not confirm it.

**Privacy** – your predictions are not stored against your account.
"""
    )
    show_disclaimer()
    r = api("GET", "/health")
    if r is not None and r.ok:
        st.success(f"Service status: ready (model version {r.json()['model_version']})", icon="✅")
    elif r is not None:
        st.error(f"Service status: unavailable – {error_text(r)}")


def _number(name: str, default: float, help_text: str):
    lim = LIMITS[name]
    cast = int if float(lim["step"]).is_integer() else float
    return st.number_input(
        f"{lim['label']} ({lim['unit']})",
        min_value=cast(lim["min"]),
        max_value=cast(lim["max"]),
        value=cast(default),
        step=cast(lim["step"]),
        help=f"{help_text} Allowed range: {lim['min']}–{lim['max']}.",
        key=name,
    )


def page_predict() -> None:
    st.title("Diabetes risk prediction")
    st.caption("All fields are required. Values outside the allowed ranges are rejected.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = _number("pregnancies", 1, "Number of times pregnant (0 if none or not applicable).")
            glucose = _number("glucose", 120, "Plasma glucose 2 hours after an oral glucose tolerance test.")
            blood_pressure = _number("blood_pressure", 72, "Diastolic blood pressure.")
            skin_thickness = _number("skin_thickness", 20, "Triceps skin-fold thickness. Enter 0 if not measured.")
        with c2:
            insulin = _number("insulin", 80, "2-hour serum insulin. Enter 0 if not measured.")
            bmi = _number("bmi", 28.0, "Weight in kg divided by height in metres squared.")
            dpf = _number("diabetes_pedigree_function", 0.4,
                          "Score summarising family history of diabetes (typical values 0.1–1.0).")
            age = _number("age", 35, "Age in years.")
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if not submitted:
        show_disclaimer()
        return

    try:
        payload = PredictionRequest(
            pregnancies=pregnancies, glucose=glucose, blood_pressure=blood_pressure,
            skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,
            diabetes_pedigree_function=dpf, age=age,
        )
    except ValidationError as exc:
        for e in exc.errors():
            st.error(f"{'.'.join(map(str, e['loc']))}: {e['msg']}")
        return

    with st.spinner("Calculating risk estimate..."):
        r = api("POST", "/predict", json=payload.model_dump())
    if r is None:
        return
    if r.status_code != 200:
        st.error(f"Prediction unavailable: {error_text(r)}")
        return
    render_result(PredictionResponse(**r.json()))


def render_result(result: PredictionResponse) -> None:
    risk = result.prediction == 1
    st.subheader("Result")
    # Icon + words + colour: the meaning never depends on colour alone.
    if risk:
        st.error(f"### ⚠️ Model Prediction: {result.prediction_label}")
    else:
        st.success(f"### ✓ Model Prediction: {result.prediction_label}")

    if result.probability is not None:
        pct = result.probability * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated probability", f"{pct:.1f}%")
        c2.metric("Risk threshold used", f"{result.decision_threshold * 100:.1f}%")
        c3.metric("Model version", result.model_version)
        st.progress(min(max(result.probability, 0.0), 1.0),
                    text=f"Estimated probability {pct:.1f}% "
                         f"({'at or above' if risk else 'below'} the {result.decision_threshold * 100:.1f}% threshold)")
    st.info(result.interpretation)
    if risk:
        st.markdown("**Recommendation:** please book an appointment with a doctor or other qualified "
                    "healthcare professional to discuss diagnostic testing.")
    show_disclaimer()


# --------------------------------------------------------------------------- admin pages
def page_model() -> None:
    st.title("Model information")
    r = api("GET", "/model/info")
    if r is None:
        return
    if not r.ok:
        st.error(error_text(r))
        return
    info = r.json()
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", info["model_name"])
    c2.metric("Model version", info["model_version"])
    c3.metric("Training date (UTC)", (info.get("trained_at") or "")[:10])
    st.markdown(f"**Dataset source:** {info['dataset']['source']}  \n"
                f"**Training-data version:** `{info['dataset']['train_version']}` "
                f"({info['dataset']['train_rows']} training rows)  \n"
                f"**Decision threshold:** {info['decision_threshold']:.3f}")
    if info.get("promotion"):
        p = info["promotion"]
        st.markdown(f"**Promoted:** {p.get('promoted_at')} by {p.get('approved_by')}")
    if info.get("selection"):
        st.markdown(f"**Why this model:** {info['selection']['reason']}")

    metric_keys = ["recall", "specificity", "precision", "f1", "accuracy", "roc_auc", "pr_auc", "brier_score"]
    rows = {}
    if info.get("validation_metrics"):
        rows["Validation"] = {k: info["validation_metrics"][k] for k in metric_keys}
    if info.get("test_metrics"):
        rows["Held-out test"] = {k: info["test_metrics"][k] for k in metric_keys}
    if rows:
        st.subheader("Evaluation metrics")
        st.dataframe(pd.DataFrame(rows).T.round(3), width="stretch")
        if info.get("test_metrics"):
            cm = info["test_metrics"]["confusion_matrix"]
            st.caption(f"Test confusion matrix – true negatives {cm['tn']}, false positives {cm['fp']}, "
                       f"false negatives {cm['fn']}, true positives {cm['tp']}.")
    if info.get("all_models_validation"):
        st.subheader("Model comparison (validation set)")
        comp = {k: {m: v["validation"][m] for m in metric_keys} for k, v in info["all_models_validation"].items()}
        st.dataframe(pd.DataFrame(comp).T.round(3), width="stretch")
    st.subheader("Known limitations")
    for item in info.get("known_limitations") or []:
        st.markdown(f"- {item}")
    figures = Path(__file__).resolve().parents[1] / "reports" / "evaluation"
    imgs = [figures / n for n in ("roc_curve.png", "precision_recall_curve.png", "confusion_matrix.png",
                                  "calibration_curve.png") if (figures / n).exists()]
    if imgs:
        st.subheader("Test-set evaluation plots")
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            cols[i % 2].image(str(img))


def page_monitoring() -> None:
    st.title("Service monitoring")
    st.caption("Aggregated operational statistics only. No individual health data is shown.")
    r = api("GET", "/monitoring/summary")
    if r is None:
        return
    if not r.ok:
        st.error(error_text(r))
        return
    summary = r.json()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total prediction requests", summary["total_requests"])
    c2.metric("Successful predictions", summary["successful_predictions"])
    c3.metric("Failed requests", summary["failed_requests"])
    avg = summary["average_latency_ms"]
    c4.metric("Average latency", f"{avg:.1f} ms" if avg is not None else "n/a")
    c5.metric("Serving model version", summary["model_version"] or "none")
    st.caption(f"Counters since API start: {summary['since']} (UTC).")

    st.subheader("Model operations")
    st.markdown("After a model is promoted or rolled back with `python -m src.promote_model`, reload it here.")
    if st.button("Reload production model"):
        rr = api("POST", "/admin/reload")
        if rr is not None and rr.ok:
            st.success(f"Reloaded. Now serving model version {rr.json()['model_version']}.")
        elif rr is not None:
            st.error(error_text(rr))

    st.subheader("Data drift")
    drift = summary.get("drift_status")
    if not drift:
        st.info("No drift report has been generated yet. Use the **Drift check** page.")
    else:
        flag = "🚨 ALERT – thresholds exceeded" if drift.get("alert") else "✓ Within configured thresholds"
        st.markdown(f"**Latest report:** {drift.get('generated_at')} – **{flag}**")
        render_drift_summary(drift)
    reports = summary.get("drift_reports") or []
    if reports:
        st.markdown("**Available drift reports**")
        st.dataframe(pd.DataFrame(reports), width="stretch", hide_index=True)
        chosen = st.selectbox("Open a report", [rep["name"] for rep in reports])
        if st.button("Show report"):
            show_report(chosen)
    st.caption(f"Raw-feature logging for drift analysis: "
               f"{'enabled' if summary.get('feature_logging_enabled') else 'disabled (default)'}.")


def render_drift_summary(drift: dict) -> None:
    st.markdown(f"- Drifted features: {drift.get('n_drifted')} of {drift.get('n_features')} "
                f"(share {drift.get('drift_share', 0):.2f}, threshold {drift.get('drift_share_threshold')})\n"
                f"- Drifted feature names: {', '.join(drift.get('drifted_features') or []) or 'none'}\n"
                f"- Predicted-positive rate: current {_pct(drift.get('current_positive_rate'))} vs "
                f"reference {_pct(drift.get('reference_positive_rate'))}\n"
                f"- Data-quality issues: {', '.join(drift.get('quality_issues') or []) or 'none'}")
    st.caption("Drift means the inputs look different from the training data. It does not by itself "
               "prove the model has become less accurate – that needs labelled outcome data. "
               "Retraining is never triggered automatically.")


def show_report(name: str) -> None:
    r = api("GET", f"/admin/drift-reports/{name}", timeout=60)
    if r is not None and r.ok:
        components.html(r.text, height=900, scrolling=True)
    elif r is not None:
        st.error(error_text(r))


def page_drift() -> None:
    st.title("Drift check")
    st.markdown(
        "Upload an **approved** CSV batch containing the eight feature columns "
        f"(`{', '.join(LIMITS)}`). It is compared with the training data using Evidently. "
        "The uploaded file is deleted after the report is generated."
    )
    upload = st.file_uploader("Approved monitoring batch (.csv)", type=["csv"])
    if upload is not None and st.button("Run drift check", type="primary"):
        with st.spinner("Running Evidently drift analysis..."):
            r = api("POST", "/admin/drift-check", timeout=180,
                    files={"file": (upload.name, upload.getvalue(), "text/csv")})
        if r is None:
            return
        if not r.ok:
            st.error(error_text(r))
            return
        result = r.json()
        if result["alert"]:
            st.error("🚨 Drift / data-quality alert: configured thresholds exceeded.")
        else:
            st.success("✓ No alert: within configured thresholds.")
        render_drift_summary(result)
        show_report(result["report_html"])


def page_users() -> None:
    st.title("User management")
    me = session()["username"]
    r = api("GET", "/admin/users")
    if r is None:
        return
    if not r.ok:
        st.error(error_text(r))
        return
    users = r.json()
    st.dataframe(pd.DataFrame(users), width="stretch", hide_index=True)

    st.subheader("Create account")
    with st.form("create_user", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        username = c1.text_input("Username")
        password = c2.text_input("Initial password", type="password")
        role = c3.selectbox("Role", ["user", "admin"])
        if st.form_submit_button("Create"):
            rr = api("POST", "/admin/users", json={"username": username, "password": password, "role": role})
            if rr is not None and rr.status_code == 201:
                st.success(f"Created {role} '{username}'.")
                st.rerun()
            elif rr is not None:
                st.error(error_text(rr))

    st.subheader("Edit account")
    names = [u["username"] for u in users]
    target = st.selectbox("Account", names)
    user = next(u for u in users if u["username"] == target)
    c1, c2 = st.columns(2)
    with c1, st.form("edit_user"):
        new_role = st.selectbox("Role", ["user", "admin"], index=["user", "admin"].index(user["role"]))
        active = st.checkbox("Account active", value=user["is_active"])
        new_pw = st.text_input("Reset password (optional)", type="password")
        if st.form_submit_button("Save changes"):
            body = {"role": new_role, "is_active": active}
            if new_pw:
                body["password"] = new_pw
            rr = api("PATCH", f"/admin/users/{target}", json=body)
            if rr is not None and rr.ok:
                st.success("Saved.")
                st.rerun()
            elif rr is not None:
                st.error(error_text(rr))
    with c2:
        st.markdown("**Delete account**")
        if target == me:
            st.caption("You cannot delete your own account.")
        else:
            confirm = st.checkbox(f"I understand '{target}' will be permanently deleted.")
            if st.button("Delete account", disabled=not confirm):
                rr = api("DELETE", f"/admin/users/{target}")
                if rr is not None and rr.status_code == 204:
                    st.success(f"Deleted '{target}'.")
                    st.rerun()
                elif rr is not None:
                    st.error(error_text(rr))


def _pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


# --------------------------------------------------------------------------- navigation
USER_PAGES = {"Home": page_home, "Prediction": page_predict}
ADMIN_PAGES = {"Model information": page_model, "Monitoring": page_monitoring,
               "Drift check": page_drift, "User management": page_users}

s = session()
if s is None:
    page_sign_in()
else:
    pages = dict(USER_PAGES)
    if s["role"] == "admin":
        pages.update(ADMIN_PAGES)
    st.sidebar.markdown(f"Signed in as **{s['username']}**  \nRole: **{s['role'].capitalize()}**")
    if st.sidebar.button("Sign out"):
        sign_out("You have been signed out.")
        st.rerun()
    choice = st.sidebar.radio("Navigation", list(pages))
    st.sidebar.caption(f"API: {API_URL}")
    pages[choice]()
