"""Loads the trained pipeline artifact once and serves predictions from it."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import DISCLAIMER, LABELS
from src.common import FEATURES


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model artifact can be used. Predictions are never faked."""


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._model = None
        self._metadata: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def version(self) -> str | None:
        return self._metadata.get("model_version")

    def load(self) -> None:
        """(Re)load the artifact. On failure the previously loaded model (if any) stays active."""
        model_path = self.model_dir / "model.joblib"
        meta_path = self.model_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            self.load_error = (
                f"No trained model found in '{self.model_dir}'. Train and promote one first: "
                "`dvc repro` (or `python -m src.data_processing && python -m src.train && python -m src.evaluate`) "
                "then `python -m src.promote_model --approve --approved-by \"<your name>\"`."
            )
            raise ModelNotAvailableError(self.load_error)
        try:
            model = joblib.load(model_path)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            eval_path = self.model_dir / "evaluation.json"
            if eval_path.exists():  # held-out test results written by src.evaluate
                metadata.update(json.loads(eval_path.read_text(encoding="utf-8")))
        except Exception as exc:
            self.load_error = f"Failed to load model artifact: {exc}"
            raise ModelNotAvailableError(self.load_error) from exc
        if metadata.get("features") != FEATURES:
            self.load_error = "Model artifact feature list does not match the API schema."
            raise ModelNotAvailableError(self.load_error)
        if not hasattr(model, "predict_proba"):
            self.load_error = "Model artifact does not support probability estimates."
            raise ModelNotAvailableError(self.load_error)
        # Single-row inference is fastest single-threaded (avoids thread-pool start-up per request).
        if "clf" in getattr(model, "named_steps", {}) and "n_jobs" in model.named_steps["clf"].get_params():
            model.set_params(clf__n_jobs=1)
        with self._lock:
            self._model, self._metadata, self.load_error = model, metadata, None

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        with self._lock:
            model, meta = self._model, self._metadata
        if model is None:
            raise ModelNotAvailableError(self.load_error or "Model is not loaded.")

        row = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES, dtype=float)
        probability = float(model.predict_proba(row)[0, 1])
        threshold = float(meta["decision_threshold"])
        prediction = int(probability >= threshold)
        if prediction:
            interpretation = (
                "The model estimates an elevated diabetes risk for these values. This is not a diagnosis: "
                "please discuss appropriate testing (e.g. HbA1c or fasting glucose) with a healthcare professional."
            )
        else:
            interpretation = (
                "The model did not indicate elevated risk for these values. This does NOT prove the absence "
                "of diabetes; the model can miss cases. Seek medical advice if you have symptoms or concerns."
            )
        return {
            "prediction": prediction,
            "prediction_label": LABELS[prediction],
            "probability": round(probability, 4),
            "decision_threshold": round(threshold, 4),
            "model_version": meta["model_version"],
            "model_name": meta["model_name"],
            "interpretation": interpretation,
            "disclaimer": DISCLAIMER,
        }
