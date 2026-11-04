"""FastAPI backend serving diabetes-risk predictions from the trained model artifact.

Access control (enforced here, not just in the UI):
  public     GET /, /health, /ready, /metrics (operational data only), POST /auth/login, /auth/register
  user+admin POST /predict, GET /auth/me
  admin      GET /model/info, /monitoring/summary, /admin/users, /admin/drift-reports/{name}
             POST /admin/reload, /admin/users, /admin/drift-check; PATCH/DELETE /admin/users/{name}

Run locally:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import monitoring
from app.auth import AccountLocked, AuthError, InvalidCredentials, TokenSigner, UserStore
from app.config import DISCLAIMER, Settings, get_settings
from app.predictor import ModelNotAvailableError, Predictor
from app.schemas import (
    Credentials,
    ErrorResponse,
    HealthResponse,
    NewUser,
    PredictionRequest,
    PredictionResponse,
    TokenResponse,
    UserOut,
    UserUpdate,
)
from src.common import resolve

DRIFT_REPORT_NAME = re.compile(r"^drift_report_\d{8}T\d{6}Z\.html$")


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Only operational fields are ever logged, never request bodies."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("diabetes_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def create_app(settings: Settings | None = None, predictor: Predictor | None = None,
               users: UserStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    log = _configure_logging(settings.log_level)
    predictor = predictor or Predictor(settings.model_dir)
    users = users or UserStore(settings.auth_db_path)
    if not settings.auth_secret_key:
        log.warning("AUTH_SECRET_KEY not set: using a random key; all sessions end when the API restarts")
    signer = TokenSigner(settings.auth_secret_key or secrets.token_urlsafe(48), settings.token_ttl_minutes * 60)
    bearer = HTTPBearer(auto_error=False)

    def audit(action: str, **fields) -> None:
        log.info("audit", extra={"extra_fields": {"action": action, **fields}})

    def try_load_model() -> None:
        try:
            predictor.load()
            monitoring.set_model_status(True, predictor.version, predictor.metadata.get("model_name"))
            log.info("model loaded", extra={"extra_fields": {"model_version": predictor.version}})
        except ModelNotAvailableError as exc:
            monitoring.MODEL_LOAD_FAILURES.inc()
            monitoring.set_model_status(predictor.is_loaded, predictor.version,
                                        predictor.metadata.get("model_name"))
            log.error("model not available", extra={"extra_fields": {"error": str(exc)}})

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        created = users.ensure_admin(settings.bootstrap_admin_username, settings.bootstrap_admin_password)
        if created:
            audit("bootstrap_admin_created", username=created)
        elif users.count_active_admins() == 0:
            log.warning("no admin account exists: set ADMIN_USERNAME and ADMIN_PASSWORD, or run "
                        "`python -m app.manage_users create <name> --role admin`")
        try_load_model()
        monitoring.read_drift_status(settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description="Educational diabetes-risk estimation API with user and admin roles. " + DISCLAIMER,
        lifespan=lifespan,
    )
    app.state.predictor = predictor
    app.state.settings = settings
    app.state.users = users

    # ------------------------------------------------------------------------- auth dependencies
    def current_user(request: Request,
                     creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        try:
            if creds is None or creds.scheme.lower() != "bearer":
                raise InvalidCredentials("Sign in required.")
            claims = signer.verify(creds.credentials)
            user = users.get_user(claims["sub"])  # role/active status read fresh: changes apply immediately
            if not user["is_active"]:
                raise InvalidCredentials("Account is disabled.")
            return user
        except AuthError as exc:
            if request.url.path == "/predict":
                monitoring.record_failure("unauthorized")
            raise HTTPException(status_code=401, detail=str(exc) if isinstance(exc, InvalidCredentials)
                                else "Sign in required.", headers={"WWW-Authenticate": "Bearer"}) from exc

    def require_admin(user: dict = Depends(current_user)) -> dict:
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Administrator access required.")
        return user

    # ------------------------------------------------------------------------- middleware / handlers
    @app.middleware("http")
    async def limit_and_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        limit = settings.max_upload_bytes if request.url.path == "/admin/drift-check" else settings.max_request_bytes
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > limit):
            if request.url.path == "/predict":
                monitoring.PREDICTION_REQUESTS.inc()
                monitoring.record_failure("payload_too_large")
            response = JSONResponse({"detail": f"Request body exceeds {limit} bytes."},
                                    status_code=status.HTTP_413_CONTENT_TOO_LARGE)
        else:
            if request.url.path == "/predict" and request.method == "POST":
                monitoring.PREDICTION_REQUESTS.inc()
            response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        if request.url.path != "/metrics":
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": response.status_code, "duration_ms": duration_ms}})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        if request.url.path == "/predict":
            monitoring.record_failure("validation_error")
        # Return location + message only; never echo submitted values (health data or passwords).
        errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
                  for e in exc.errors()]
        return JSONResponse({"detail": jsonable_encoder(errors)}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    @app.exception_handler(AuthError)
    async def auth_error_handler(_request: Request, exc: AuthError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    # ------------------------------------------------------------------------- public
    @app.get("/", tags=["service"])
    def root() -> dict:
        return {
            "service": settings.app_name,
            "api_version": settings.api_version,
            "status": "ok",
            "model_loaded": predictor.is_loaded,
            "model_version": predictor.version,
            "self_registration": settings.allow_self_registration,
            "docs": "/docs",
            "disclaimer": DISCLAIMER,
        }

    @app.get("/health", response_model=HealthResponse, tags=["service"],
             responses={503: {"model": HealthResponse}})
    def health(response: Response) -> HealthResponse:
        if not predictor.is_loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(status="degraded", model_loaded=False, detail=predictor.load_error)
        return HealthResponse(status="ok", model_loaded=True, model_version=predictor.version)

    @app.get("/ready", tags=["service"])
    def ready(response: Response) -> dict:
        is_ready = predictor.is_loaded
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": is_ready, "model_version": predictor.version}

    @app.get("/metrics", tags=["monitoring"])
    def metrics() -> Response:
        """Prometheus scrape target. Aggregated operational counters only, no health data or usernames."""
        monitoring.read_drift_status(settings)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------------- authentication
    @app.post("/auth/register", response_model=UserOut, status_code=201, tags=["auth"])
    def register(body: Credentials) -> dict:
        if not settings.allow_self_registration:
            raise HTTPException(status_code=403, detail="Self-registration is disabled. Ask an administrator.")
        user = users.create_user(body.username, body.password, role="user")  # never admin via self-signup
        monitoring.AUTH_EVENTS.labels(event="register").inc()
        audit("register", username=user["username"])
        return user

    @app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
    def login(body: Credentials) -> TokenResponse:
        try:
            user = users.authenticate(body.username, body.password)
        except AccountLocked:
            monitoring.AUTH_EVENTS.labels(event="login_locked").inc()
            raise
        except InvalidCredentials:
            monitoring.AUTH_EVENTS.labels(event="login_failure").inc()
            raise
        token, exp = signer.issue(user["username"], user["role"])
        monitoring.AUTH_EVENTS.labels(event="login_success").inc()
        return TokenResponse(access_token=token, expires_at=exp, username=user["username"], role=user["role"])

    @app.get("/auth/me", response_model=UserOut, tags=["auth"])
    def me(user: dict = Depends(current_user)) -> dict:
        return user

    # ------------------------------------------------------------------------- prediction (user + admin)
    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
              responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                         503: {"model": ErrorResponse}})
    def predict(payload: PredictionRequest, _user: dict = Depends(current_user)) -> PredictionResponse:
        features = payload.model_dump()
        start = time.perf_counter()
        try:
            result = predictor.predict(features)
        except ModelNotAvailableError as exc:
            monitoring.record_failure("model_unavailable")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            monitoring.record_failure("internal_error")
            log.exception("prediction failed", extra={"extra_fields": {"error_type": type(exc).__name__}})
            raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from exc
        latency = time.perf_counter() - start
        monitoring.record_success(latency, result["prediction"])
        try:
            # Deliberately NOT linked to the signed-in user.
            monitoring.log_prediction(settings, result, latency, features)
        except OSError as exc:  # monitoring storage problems must not break predictions
            log.warning("could not write monitoring log", extra={"extra_fields": {"error": str(exc)}})
        return PredictionResponse(**result)

    # ------------------------------------------------------------------------- admin: model & monitoring
    @app.get("/model/info", tags=["admin"])
    def model_info(_admin: dict = Depends(require_admin)) -> dict:
        if not predictor.is_loaded:
            raise HTTPException(status_code=503, detail=predictor.load_error or "Model is not loaded.")
        meta = predictor.metadata
        keys = ("model_name", "model_version", "trained_at", "decision_threshold", "dataset", "selection",
                "validation_metrics", "all_models_validation", "test_metrics", "acceptance", "hyperparameters",
                "known_limitations", "promotion", "environment", "mlflow_run_id", "mlflow_registered_version")
        return {k: meta.get(k) for k in keys}

    @app.get("/monitoring/summary", tags=["admin"])
    def monitoring_summary(_admin: dict = Depends(require_admin)) -> dict:
        return {
            **monitoring.STATS.snapshot(),
            "model_loaded": predictor.is_loaded,
            "model_version": predictor.version,
            "drift_status": monitoring.read_drift_status(settings),
            "drift_reports": monitoring.list_drift_reports(settings),
            "feature_logging_enabled": settings.log_features,
        }

    @app.post("/admin/reload", tags=["admin"])
    def reload_model(request: Request, x_admin_token: str | None = Header(default=None),
                     creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        """Reload the production artifact after an approved promotion or rollback.

        Accepts an admin's bearer token, or the X-Admin-Token header (ADMIN_TOKEN) for automation scripts.
        """
        if x_admin_token is not None:
            if not settings.admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
                raise HTTPException(status_code=401, detail="Invalid admin token.")
            actor = "automation-token"
        else:
            actor = require_admin(current_user(request, creds))["username"]
        try_load_model()
        if not predictor.is_loaded:
            raise HTTPException(status_code=503, detail=predictor.load_error)
        audit("model_reload", actor=actor, model_version=predictor.version)
        return {"reloaded": True, "model_version": predictor.version}

    @app.post("/admin/drift-check", tags=["admin"])
    def drift_check(file: UploadFile = File(..., description="Approved CSV batch with the 8 feature columns"),
                    admin: dict = Depends(require_admin)) -> dict:
        """Run an Evidently drift report on an uploaded, approved batch. The upload is deleted afterwards."""
        from src.drift_detection import InsufficientDataError, run_drift_check

        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a .csv file.")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.csv"
            path.write_bytes(data)
            try:
                result = run_drift_check(str(path), config=settings.config, model_dir=settings.model_dir)
            except (InsufficientDataError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                log.exception("drift check failed")
                raise HTTPException(status_code=500, detail="Drift check failed. See API logs.") from exc
        monitoring.read_drift_status(settings)
        audit("drift_check", actor=admin["username"], rows=result["n_current"], alert=result["alert"])
        return result

    @app.get("/admin/drift-reports/{name}", response_class=HTMLResponse, tags=["admin"])
    def drift_report(name: str, _admin: dict = Depends(require_admin)) -> HTMLResponse:
        if not DRIFT_REPORT_NAME.fullmatch(name):
            raise HTTPException(status_code=400, detail="Invalid report name.")
        path = resolve(settings.monitoring["reports_dir"]) / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report not found.")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------------- admin: user management
    @app.get("/admin/users", response_model=list[UserOut], tags=["admin"])
    def list_users(_admin: dict = Depends(require_admin)) -> list[dict]:
        return users.list_users()

    @app.post("/admin/users", response_model=UserOut, status_code=201, tags=["admin"])
    def create_user(body: NewUser, admin: dict = Depends(require_admin)) -> dict:
        user = users.create_user(body.username, body.password, role=body.role)
        audit("user_created", actor=admin["username"], username=user["username"], role=user["role"])
        return user

    @app.patch("/admin/users/{username}", response_model=UserOut, tags=["admin"])
    def update_user(username: str, body: UserUpdate, admin: dict = Depends(require_admin)) -> dict:
        user = users.update_user(username, role=body.role, is_active=body.is_active, password=body.password)
        audit("user_updated", actor=admin["username"], username=user["username"], role=user["role"],
              is_active=user["is_active"], password_reset=body.password is not None)
        return user

    @app.delete("/admin/users/{username}", status_code=204, tags=["admin"])
    def delete_user(username: str, admin: dict = Depends(require_admin)) -> Response:
        if username.lower() == admin["username"].lower():
            raise HTTPException(status_code=409, detail="You cannot delete your own account.")
        users.delete_user(username)
        audit("user_deleted", actor=admin["username"], username=username)
        return Response(status_code=204)

    return app


app = create_app()
