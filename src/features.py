"""Preprocessing components that are pickled inside the model pipeline.

Keep this module importable by both the training code and the API: the saved
joblib artifact references ``src.features.ZeroAsMissing`` by import path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in the given columns (and only those columns).

    In the Pima dataset a 0 in glucose, blood pressure, skin thickness, insulin or
    BMI is a missing-value code, not a measurement. Pregnancies = 0 is a real value
    and must be left untouched, so the columns are passed explicitly.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroAsMissing expects a pandas DataFrame with named columns.")
        missing = set(self.columns or []) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in input: {sorted(missing)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns or []:
            X[col] = X[col].astype(float).replace(0.0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
