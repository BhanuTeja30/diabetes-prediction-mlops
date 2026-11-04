"""Prometheus metrics and privacy-conscious prediction logging.

No metric label ever contains health inputs. Labels are limited to small, fixed
vocabularies (failure reason, predicted class, model version).
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from src.common import FEATURES, resolve

PREDICTION_REQUESTS = Counter("prediction_requests_total", "Prediction requests received")
PREDICTION_SUCCESS = Counter("prediction_success_total", "Predictions returned successfully")
PREDICTION_FAILURES = Counter("prediction_failures_total", "Failed prediction requests", ["reason"])
PREDICTION_CLASS = Counter("prediction_class_total", "Predictions by predicted class", ["predicted_class"])
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
MODEL_LOADED = Gauge("model_loaded", "1 if a trained model is loaded and serving, else 0")
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Failed attempts to load the model artifact")
MODEL_INFO = Gauge("model_info", "Currently served model (value is always 1)", ["model_version", "model_name"])
DRIFT_ALERT = Gauge("data_drift_alert", "1 if the latest drift report exceeded configured thresholds")
DRIFT_SHARE = Gauge("data_drift_share", "Share of monitored features flagged as drifted in the latest report")
AUTH_EVENTS = Counter("auth_events_total", "Authentication events (no usernames recorded)", ["event"])
DRIFT_REPORT_AGE = Gauge("data_drift_report_timestamp_seconds", "Unix time of the latest drift report")

FAILURE_REASONS = ("validation_error", "model_unavailable", "payload_too_large", "unauthorized", "internal_error")
for _reason in FAILURE_REASONS:
    PREDICTION_FAILURES.labels(reason=_reason)  # pre-create so series exist at 0
for _event in ("login_success", "login_failure", "login_locked", "register"):
    AUTH_EVENTS.labels(event=_event)
for _cls in ("risk_indicated", "risk_not_indicated"):
    PREDICTION_CLASS.labels(predicted_class=_cls)


class Stats:
    """Small in-process counters used by the Streamlit monitoring page."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.total = self.success = self.failed = 0
        self.latency_sum = 0.0
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")

    def record(self, ok: bool, latency: float | None = None) -> None:
        with self._lock:
            self.total += 1
            if ok:
                self.success += 1
                self.latency_sum += latency or 0.0
            else:
                self.failed += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg = (self.latency_sum / self.success * 1000) if self.success else None
            return {
                "total_requests": self.total,
                "successful_predictions": self.success,
                "failed_requests": self.failed,
                "average_latency_ms": round(avg, 2) if avg is not None else None,
                "since": self.started_at,
            }


STATS = Stats()


def set_model_status(loaded: bool, version: str | None = None, name: str | None = None) -> None:
    MODEL_LOADED.set(1 if loaded else 0)
    MODEL_INFO.clear()
    if loaded and version:
        MODEL_INFO.labels(model_version=version, model_name=name or "unknown").set(1)


def record_success(latency: float, prediction: int) -> None:
    PREDICTION_SUCCESS.inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CLASS.labels(predicted_class="risk_indicated" if prediction else "risk_not_indicated").inc()
    STATS.record(True, latency)


def record_failure(reason: str) -> None:
    PREDICTION_FAILURES.labels(reason=reason if reason in FAILURE_REASONS else "internal_error").inc()
    STATS.record(False)


_file_lock = threading.Lock()


def _append_csv(path: Path, header: list[str], row: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(header)
            writer.writerow(row)


def log_prediction(settings, result: dict[str, Any], latency: float, features: dict[str, float]) -> None:
    """Persist monitoring data. Timestamps are coarsened to the hour; no identifiers are stored."""
    mon = settings.monitoring
    hour = datetime.now(UTC).strftime("%Y-%m-%dT%H:00Z")
    if settings.log_predictions:
        _append_csv(resolve(mon["predictions_log"]),
                    ["time_bucket", "model_version", "prediction", "probability", "latency_ms"],
                    [hour, result["model_version"], result["prediction"], result["probability"],
                     round(latency * 1000, 2)])
    if settings.log_features:
        # Only enabled with documented approval (MONITORING_LOG_FEATURES=true). Rows are not linked
        # to any request id, user or exact timestamp.
        _append_csv(resolve(mon["features_log"]), ["date", *FEATURES, "prediction"],
                    [hour[:10], *[features[f] for f in FEATURES], result["prediction"]])


def read_drift_status(settings) -> dict[str, Any] | None:
    path = resolve(settings.monitoring["status_file"])
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    DRIFT_ALERT.set(1 if status.get("alert") else 0)
    DRIFT_SHARE.set(float(status.get("drift_share") or 0.0))
    try:
        DRIFT_REPORT_AGE.set(datetime.fromisoformat(status["generated_at"]).timestamp())
    except (KeyError, ValueError):
        pass
    return status


def list_drift_reports(settings, limit: int = 20) -> list[dict[str, Any]]:
    folder = resolve(settings.monitoring["reports_dir"])
    reports = sorted(folder.glob("drift_report_*.html"), reverse=True)[:limit]
    return [{"name": p.name, "modified": datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat(timespec="seconds")}
            for p in reports]
