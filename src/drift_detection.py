"""Data-drift, prediction-drift and data-quality monitoring with Evidently AI.

Compares a *current* batch of approved monitoring data with the training reference data
and writes an HTML report, a JSON summary and reports/drift/latest_status.json (read by
the API's /metrics endpoint -> Prometheus -> Grafana alert).

What this does and does NOT tell you:
  * feature drift      - input distributions differ from training data
  * prediction drift   - the share of "risk indicated" outputs has shifted
  * data quality       - missing / out-of-range values in incoming data
  * model performance  - NOT measurable here: that needs verified outcome labels, which
                         this system does not collect. Drift is a prompt to investigate,
                         not proof that the model is wrong, and never triggers retraining.

Usage:
    python -m src.drift_detection                          # use the approved feature log
    python -m src.drift_detection --current batch.csv      # an operator-supplied approved batch
Schedule it periodically (cron / Windows Task Scheduler), e.g. daily.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, get_logger, load_config, read_json, resolve, write_json

log = get_logger("drift_detection")


class InsufficientDataError(RuntimeError):
    pass


def _load_current(config: dict[str, Any], current_path: str | None) -> pd.DataFrame:
    mon = config["monitoring"]
    path = resolve(current_path or mon["features_log"])
    if not path.exists():
        raise InsufficientDataError(
            f"No current monitoring data at {path}. Either enable approved feature logging "
            "(MONITORING_LOG_FEATURES=true) or pass --current <approved_batch.csv>."
        )
    df = pd.read_csv(path)
    missing = [c for c in mon["drift_features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Current data is missing columns: {missing}")
    return df


ZERO_AS_MISSING = ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi")


def _quality_issues(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    limits = config["input_limits"]
    issues, details = [], {}
    for col in FEATURES:
        if col not in df:
            continue
        lim = limits[col]
        out = int(((df[col] < lim["min"]) | (df[col] > lim["max"])).sum())
        miss = float(df[col].isna().mean())
        details[col] = {"out_of_range": out, "missing_share": round(miss, 4)}
        if out:
            issues.append(f"{col}: {out} out-of-range values")
        if miss > 0.5 and col not in ("insulin", "skin_thickness"):
            issues.append(f"{col}: {miss:.0%} missing")
    return issues, details


def _parse_drift(snapshot_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract per-column drift results from an Evidently snapshot."""
    out = {}
    for m in snapshot_dict["metrics"]:
        cfg = m.get("config", {})
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        method, threshold, value = cfg.get("method", ""), float(cfg.get("threshold")), float(m["value"])
        # p-value based tests flag drift when p < threshold; distance metrics when value >= threshold.
        drifted = value < threshold if "p_value" in method else value >= threshold
        out[cfg["column"]] = {"method": method, "score": round(value, 5), "threshold": threshold,
                              "drifted": bool(drifted)}
    return out


def run_drift_check(current_path: str | None = None, config: dict[str, Any] | None = None,
                    model_dir: str | Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    config = config or load_config()
    mon = config["monitoring"]
    features = mon["drift_features"]
    now = now or datetime.now(UTC)

    reference = pd.read_csv(resolve(mon["reference_path"]))
    current = _load_current(config, current_path)
    if len(current) < mon["min_current_rows"]:
        raise InsufficientDataError(
            f"Only {len(current)} current rows; at least {mon['min_current_rows']} are needed for a meaningful test."
        )

    # Same zero-as-missing convention as training so distributions are comparable. A 0 here means
    # "not measured", so it must count as missing, not as an out-of-range value, in the quality check.
    coded = current.copy()
    for col in ZERO_AS_MISSING:
        if col in coded:
            coded[col] = coded[col].astype(float).replace(0.0, np.nan)
    quality_issues, quality_details = _quality_issues(coded, config)
    cur = coded[features].astype(float).copy()
    ref = reference[features].astype(float)

    # Prediction drift: compare the predicted-positive rate of the current batch with the rate the
    # serving model produces on the (unseen during training) validation split.
    model_dir = resolve(model_dir or config["model"]["production_dir"])
    ref_rate = cur_rate = None
    if (model_dir / "model.joblib").exists():
        model = joblib.load(model_dir / "model.joblib")
        threshold = read_json(model_dir / "metadata.json")["decision_threshold"]
        val = pd.read_csv(resolve("data/processed/validation.csv"))
        ref_rate = float((model.predict_proba(val[FEATURES])[:, 1] >= threshold).mean())
        if "prediction" in current:
            cur_rate = float(current["prediction"].mean())
        elif all(f in current for f in FEATURES):
            cur_rate = float((model.predict_proba(current[FEATURES].astype(float))[:, 1] >= threshold).mean())

    report = Report([DataDriftPreset(columns=features), DataSummaryPreset(columns=features)])
    snapshot = report.run(current_data=cur, reference_data=ref)
    per_column = _parse_drift(snapshot.dict())
    drifted = sorted(c for c, r in per_column.items() if r["drifted"])
    drift_share = len(drifted) / len(features) if features else 0.0

    feature_alert = drift_share >= mon["drift_share_threshold"]
    prediction_alert = (ref_rate is not None and cur_rate is not None
                        and abs(cur_rate - ref_rate) > mon["prediction_rate_shift_threshold"])
    alert = bool(feature_alert or prediction_alert or quality_issues)

    reports_dir = resolve(mon["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    html_path = reports_dir / f"drift_report_{stamp}.html"
    snapshot.save_html(str(html_path))

    status = {
        "generated_at": now.isoformat(timespec="seconds"),
        "report_html": html_path.name,
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "n_features": len(features),
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": mon["drift_share_threshold"],
        "feature_drift_alert": bool(feature_alert),
        "reference_positive_rate": ref_rate,
        "current_positive_rate": cur_rate,
        "prediction_drift_alert": bool(prediction_alert),
        "quality_issues": quality_issues,
        "quality_details": quality_details,
        "per_feature": per_column,
        "alert": alert,
        "note": "Drift indicates changed inputs, not confirmed loss of accuracy. Performance cannot be "
                "assessed without verified outcome labels. Retraining is never triggered automatically.",
    }
    write_json(reports_dir / f"drift_summary_{stamp}.json", status)
    write_json(mon["status_file"], status)

    if alert:
        # Operator alert: logged at WARNING here and exported as data_drift_alert=1 for Prometheus/Grafana.
        log.warning("DRIFT ALERT: drifted=%s share=%.2f prediction_rate %s->%s quality=%s",
                    drifted, drift_share, ref_rate, cur_rate, quality_issues)
    else:
        log.info("No drift alert (drift share %.2f, %d rows). Report: %s", drift_share, len(cur), html_path)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", default=None, help="CSV of approved current data (default: feature log).")
    args = parser.parse_args()
    try:
        run_drift_check(args.current)
    except InsufficientDataError as exc:
        log.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
