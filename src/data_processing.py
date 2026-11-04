"""Data preparation stage: load, profile, validate, clean, explore and split the dataset.

Usage:
    python -m src.data_processing --download   # fetch the public CSV and verify its checksum
    python -m src.data_processing              # profile + EDA + train/val/test split
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Docker / CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.common import (  # noqa: E402
    FEATURES,
    ORIGINAL_COLUMN_MAP,
    TARGET,
    get_logger,
    load_params,
    resolve,
    sha256_file,
    write_json,
)

log = get_logger("data_processing")

# Hard plausibility bounds used to flag invalid records in the dataset (wider than the
# API input limits, which are deliberately conservative for user-facing input).
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "pregnancies": (0, 25),
    "glucose": (0, 400),
    "blood_pressure": (0, 200),
    "skin_thickness": (0, 120),
    "insulin": (0, 1000),
    "bmi": (0, 80),
    "diabetes_pedigree_function": (0, 3),
    "age": (10, 120),
}
MIN_ROWS = 100


class DataValidationError(ValueError):
    """Raised when a dataset fails a blocking quality check."""


def download_dataset(url: str, dest: str | Path, expected_sha256: str | None = None) -> Path:
    dest = resolve(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", url)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL from params.yaml
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the dataset from {url}: {exc}. "
            f"Download it manually and save it as {dest} (see README, 'Dataset setup')."
        ) from exc
    digest = sha256_file(dest)
    if expected_sha256 and digest != expected_sha256:
        raise DataValidationError(
            f"Checksum mismatch for {dest}: expected {expected_sha256}, got {digest}. "
            "The upstream file changed or the download is corrupted."
        )
    log.info("Saved %s (sha256=%s)", dest, digest)
    return dest


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the CSV with or without the original header and normalise column names."""
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m src.data_processing --download` "
            "or place the Pima Indians Diabetes CSV there."
        )
    first = path.read_text(encoding="utf-8").splitlines()[0]
    has_header = any(c.isalpha() for c in first)
    if has_header:
        df = pd.read_csv(path)
        df = df.rename(columns=ORIGINAL_COLUMN_MAP)
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None, names=FEATURES + [TARGET])
    return df


def profile(df: pd.DataFrame, zero_as_missing: list[str]) -> dict[str, Any]:
    """Collect a JSON-serialisable profile of the raw dataset."""
    numeric = df.select_dtypes(include="number")
    report: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "positive_rate": float(df[TARGET].mean()),
        "zero_counts": {c: int((df[c] == 0).sum()) for c in FEATURES},
        "zero_treated_as_missing": zero_as_missing,
        "describe": numeric.describe().round(3).to_dict(),
    }
    report["out_of_plausible_range"] = {
        c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in PLAUSIBLE_RANGES.items() if c in df
    }
    return report


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Blocking quality checks. Raises DataValidationError; returns warnings otherwise."""
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    non_numeric = [c for c in FEATURES + [TARGET] if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns: {non_numeric}")
    if len(df) < MIN_ROWS:
        raise DataValidationError(f"Dataset has {len(df)} rows; at least {MIN_ROWS} are required.")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise DataValidationError("Target column 'outcome' must be binary (0/1).")
    if df[TARGET].isna().any():
        raise DataValidationError("Target column 'outcome' contains missing labels.")
    negatives = {c: int((df[c] < 0).sum()) for c in FEATURES if (df[c] < 0).any()}
    if negatives:
        raise DataValidationError(f"Negative values found: {negatives}")
    pos_rate = df[TARGET].mean()
    if not 0.05 <= pos_rate <= 0.95:
        raise DataValidationError(f"Implausible class balance (positive rate {pos_rate:.2%}).")

    warnings = []
    for c, (lo, hi) in PLAUSIBLE_RANGES.items():
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n:
            warnings.append(f"{n} rows with {c} outside plausible range [{lo}, {hi}] will be dropped")
    return {"passed": True, "warnings": warnings}


def clean(df: pd.DataFrame, zero_as_missing: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicates / implausible rows and mark zero-coded missing values as NaN.

    Imputation is NOT done here: it lives inside the model pipeline and is fitted on the
    training split only, which prevents leakage from validation/test rows.
    """
    stats = {"rows_in": int(len(df))}
    df = df[FEATURES + [TARGET]].drop_duplicates().copy()
    stats["duplicates_removed"] = stats["rows_in"] - int(len(df))

    mask = pd.Series(True, index=df.index)
    for c, (lo, hi) in PLAUSIBLE_RANGES.items():
        mask &= df[c].between(lo, hi) | df[c].isna()
    stats["implausible_removed"] = int((~mask).sum())
    df = df[mask]

    for c in zero_as_missing:
        df[c] = df[c].astype(float).replace(0.0, np.nan)
    stats["rows_out"] = int(len(df))
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True), stats


def make_eda_figures(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out_dir: str | Path) -> list[str]:
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved = []

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df_raw[TARGET].value_counts().sort_index()
    ax.bar(["No diabetes (0)", "Diabetes (1)"], counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_ylabel("Records")
    saved.append(_save(fig, out_dir / "class_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    zeros = pd.Series({c: (df_raw[c] == 0).sum() for c in FEATURES}).sort_values()
    zeros.plot.barh(ax=ax, color="#8172B2")
    ax.set_title("Zero values per feature (raw data)")
    ax.set_xlabel("Rows with value 0")
    saved.append(_save(fig, out_dir / "zero_counts.png"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), FEATURES, strict=True):
        sns.histplot(data=df_clean, x=c, hue=TARGET, kde=True, ax=ax, element="step", stat="density")
        ax.set_title(c)
    fig.suptitle("Feature distributions by outcome (zero-coded missing values excluded)")
    saved.append(_save(fig, out_dir / "feature_distributions.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix (pairwise complete observations)")
    saved.append(_save(fig, out_dir / "correlation_heatmap.png"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), FEATURES, strict=True):
        sns.boxplot(data=df_clean, x=TARGET, y=c, ax=ax)
        ax.set_title(c)
    fig.suptitle("Feature boxplots by outcome")
    saved.append(_save(fig, out_dir / "boxplots.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    (df_clean[FEATURES].isna().mean() * 100).sort_values().plot.barh(ax=ax, color="#C44E52")
    ax.set_title("Missing values after zero-code treatment (%)")
    saved.append(_save(fig, out_dir / "missing_after_treatment.png"))
    return saved


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.relative_to(resolve(".")).as_posix() if path.is_relative_to(resolve(".")) else str(path)


def split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Stratified train / validation / test split (fractions of the full dataset)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare(params: dict[str, Any], raw_path: str | Path | None = None, out_dir: str | Path | None = None,
            figures_dir: str | Path = "reports/figures") -> dict[str, Any]:
    raw_path = resolve(raw_path or params["data"]["raw_path"])
    out_dir = resolve(out_dir or params["data"]["processed_dir"])
    zero_cols = params["zero_as_missing"]

    df_raw = load_raw(raw_path)
    log.info("Loaded %s: %d rows x %d columns", raw_path.name, *df_raw.shape)
    prof = profile(df_raw, zero_cols)
    validation = validate_dataset(df_raw)
    for w in validation["warnings"]:
        log.warning(w)

    df, clean_stats = clean(df_raw, zero_cols)
    figures = make_eda_figures(df_raw, df, figures_dir)
    train, val, test = split(df, params["split"]["validation_size"], params["split"]["test_size"],
                             params["split"]["random_seed"])

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "validation.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    summary = {
        "raw_path": str(raw_path.relative_to(resolve("."))) if raw_path.is_relative_to(resolve(".")) else str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "profile": prof,
        "validation": validation,
        "cleaning": clean_stats,
        "splits": {
            name: {"rows": int(len(part)), "positive_rate": round(float(part[TARGET].mean()), 4)}
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "figures": figures,
    }
    write_json(out_dir / "data_summary.json", summary)
    write_json(resolve(figures_dir).parent / "data_summary.json", summary)
    log.info("Rows: train=%d validation=%d test=%d", len(train), len(val), len(test))
    log.info("Zero counts (raw): %s", prof["zero_counts"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Download the public dataset and exit.")
    parser.add_argument("--raw-path", default=None, help="Override the raw CSV path from params.yaml.")
    args = parser.parse_args()
    params = load_params()
    if args.download:
        download_dataset(params["data"]["source_url"], params["data"]["raw_path"], params["data"]["expected_sha256"])
        return
    prepare(params, raw_path=args.raw_path)


if __name__ == "__main__":
    main()
