import numpy as np
import pandas as pd
import pytest

from src.common import FEATURES, TARGET
from src.data_processing import DataValidationError, clean, load_raw, profile, split, validate_dataset
from src.features import ZeroAsMissing
from src.train import build_pipelines

ZERO_COLS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def test_load_raw_without_header(tmp_path, synthetic_df):
    path = tmp_path / "raw.csv"
    synthetic_df.to_csv(path, index=False, header=False)
    df = load_raw(path)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == len(synthetic_df)


def test_load_raw_with_original_kaggle_header(tmp_path, synthetic_df):
    header = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
              "DiabetesPedigreeFunction", "Age", "Outcome"]
    path = tmp_path / "raw.csv"
    synthetic_df.set_axis(header, axis=1).to_csv(path, index=False)
    assert list(load_raw(path).columns) == FEATURES + [TARGET]


def test_load_raw_missing_file_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        load_raw(tmp_path / "nope.csv")


def test_zero_as_missing_only_touches_listed_columns(synthetic_df):
    out = ZeroAsMissing(ZERO_COLS).fit(synthetic_df[FEATURES]).transform(synthetic_df[FEATURES])
    assert out["insulin"].isna().sum() == (synthetic_df["insulin"] == 0).sum()
    # Pregnancies = 0 is a genuine value and must never become missing.
    assert out["pregnancies"].isna().sum() == 0
    assert (out["pregnancies"] == 0).sum() == (synthetic_df["pregnancies"] == 0).sum()


def test_zero_as_missing_rejects_unknown_columns(synthetic_df):
    with pytest.raises(ValueError):
        ZeroAsMissing(["not_a_column"]).fit(synthetic_df[FEATURES])


def test_profile_reports_zeros_duplicates_and_classes(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(3)])
    prof = profile(df, ZERO_COLS)
    assert prof["duplicate_rows"] == 3
    assert prof["zero_counts"]["insulin"] >= 120
    assert sum(prof["class_distribution"].values()) == len(df)


def test_clean_removes_duplicates_implausible_rows_and_marks_missing(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(5)], ignore_index=True)
    df.loc[10, "bmi"] = 500  # implausible (row not duplicated)
    cleaned, stats = clean(df, ZERO_COLS)
    assert stats["duplicates_removed"] == 5
    assert stats["implausible_removed"] == 1
    assert cleaned["insulin"].isna().any()
    assert not (cleaned[ZERO_COLS] == 0).any().any()
    assert (cleaned["pregnancies"] == 0).any()


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns=["glucose"]), "Missing required columns"),
    (lambda d: d.assign(glucose=-d["glucose"]), "Negative values"),
    (lambda d: d.assign(outcome=2), "binary"),
    (lambda d: d.head(10), "rows"),
])
def test_validate_dataset_blocks_bad_data(synthetic_df, mutate, message):
    with pytest.raises(DataValidationError, match=message):
        validate_dataset(mutate(synthetic_df.copy()))


def test_validate_dataset_accepts_good_data(synthetic_df):
    assert validate_dataset(synthetic_df)["passed"]


def test_split_is_stratified_disjoint_and_reproducible(synthetic_df):
    df, _ = clean(synthetic_df, ZERO_COLS)
    a = split(df, 0.2, 0.2, seed=42)
    b = split(df, 0.2, 0.2, seed=42)
    train, val, test = a
    assert len(train) + len(val) + len(test) == len(df)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    overall = df[TARGET].mean()
    for part in a:
        assert abs(part[TARGET].mean() - overall) < 0.05
    keys = [set(map(tuple, p[FEATURES].fillna(-1).values)) for p in a]
    assert not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_imputer_is_fitted_on_training_data_only(synthetic_df, params):
    """No leakage: imputation medians come from the training split alone."""
    df, _ = clean(synthetic_df, ZERO_COLS)
    train, _val, test = split(df, 0.2, 0.2, seed=1)
    pipe = build_pipelines(params, train[TARGET])["logistic_regression"].fit(train[FEATURES], train[TARGET])
    medians = pipe.named_steps["imputer"].statistics_[: len(FEATURES)]
    np.testing.assert_allclose(medians, train[FEATURES].median().values)
    assert not np.allclose(medians, pd.concat([train, test])[FEATURES].median().values)


def test_pipeline_handles_zero_coded_missing_input(synthetic_df, params):
    df, _ = clean(synthetic_df, ZERO_COLS)
    pipe = build_pipelines(params, df[TARGET])["random_forest"].fit(df[FEATURES], df[TARGET])
    row = pd.DataFrame([[1, 120, 70, 0, 0, 30.0, 0.5, 40]], columns=FEATURES, dtype=float)
    prob = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= prob <= 1.0
