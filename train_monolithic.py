"""
Enterprise MLOps Pipeline — Monolithic Training Script
=======================================================
A self-contained, single-file ML workflow that mirrors what
many data science teams start with before adopting modular
MLOps practices. This script serves as the BASELINE for
benchmarking against the DVC-orchestrated modular pipeline.

Architecture: All stages (load → clean → encode → split →
              train → evaluate → save) are executed linearly
              within a single process with no caching,
              no partial re-runs, and no dependency tracking.

Benchmark Role:
  - Measures full pipeline execution time (wall clock)
  - Demonstrates the "re-run everything" problem
  - Contrasts with DVC caching behavior

CLI Usage:
    python train_monolithic.py
    python train_monolithic.py --n-estimators 200 --max-depth 15
    python train_monolithic.py --output-dir /tmp/my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ── Logging Setup ──────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | monolithic | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "monolithic.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("monolithic")

# ── Constants ──────────────────────────────────────────────
COLUMN_NAMES: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]

DEFAULT_DATA_PATH = Path("data/adult.csv")
DEFAULT_MODEL_PATH = Path("models/model_monolithic.joblib")
DEFAULT_METRICS_PATH = Path("metrics/scores_monolithic.json")


# ══════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════

def _load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the UCI Adult dataset from disk.

    Handles both headered and headerless CSV variants.

    Parameters
    ----------
    data_path : Path
        Path to the raw adult.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with canonical column names.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist at the given path.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Please place adult.csv in the data/ directory."
        )

    with data_path.open("r") as fh:
        first_line = fh.readline().strip()

    is_header = (
        first_line.split(",")[0].strip().lower() in ("age", "age ")
        or not first_line.split(",")[0].strip().lstrip("-").isdigit()
    )

    if is_header:
        df = pd.read_csv(data_path, na_values=["?", " ?"])
        df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
        if len(df.columns) == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
    else:
        df = pd.read_csv(
            data_path,
            names=COLUMN_NAMES,
            na_values=["?", " ?"],
            skipinitialspace=True,
        )

    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Steps performed:
    1. Strip whitespace from all string columns.
    2. Normalize target labels (remove trailing periods).
    3. Drop rows containing missing values.
    4. Remove duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".")

    rows_before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    logger.info(
        "Cleaned data: removed %d rows >> %d remaining",
        rows_before - len(df), len(df),
    )
    return df


def _encode_features(
    df: pd.DataFrame,
    target_col: str = "income",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode features and target for model training.

    Uses OrdinalEncoder for categorical columns and maps the
    binary income target to {0, 1}.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X feature matrix, y label vector)
    """
    X_df = df.drop(columns=[target_col])
    y_raw = df[target_col]

    income_map = {"<=50K": 0, ">50K": 1}
    y = y_raw.map(income_map)
    if y.isnull().any():
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y_raw))

    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(exclude="object").columns.tolist()

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X_df[cat_cols])
        X_num = X_df[num_cols].to_numpy(dtype=np.float64)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_df.to_numpy(dtype=np.float64)

    logger.info("Encoded feature matrix: %s | Labels: %s", X.shape, y.shape)
    return X, y.to_numpy(dtype=np.int32)


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training set.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth per tree.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 = use all cores).

    Returns
    -------
    RandomForestClassifier
        Fitted model instance.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced",
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    logger.info(
        "Training RandomForest — n_estimators=%d | max_depth=%s | n_jobs=%d",
        n_estimators, max_depth, n_jobs,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    logger.info("Training complete in %.3fs", time.perf_counter() - t0)
    return model


def _evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Generate predictions and compute evaluation metrics.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    dict[str, Any]
        Metrics dictionary including accuracy, AUC, F1 variants.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "auc": round(float(roc_auc_score(y_test, y_prob)), 6),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "f1_binary": round(float(f1_score(y_test, y_pred, average="binary")), 6),
        "n_test_samples": int(len(y_test)),
        "model_type": "monolithic_random_forest",
    }

    logger.info(
        "Evaluation — accuracy=%.4f | auc=%.4f | f1_macro=%.4f",
        metrics["accuracy"], metrics["auc"], metrics["f1_macro"],
    )
    return metrics


# ══════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the monolithic run.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Enterprise MLOps — Monolithic Training Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_monolithic.py\n"
            "  python train_monolithic.py --n-estimators 200 --max-depth 15\n"
            "  python train_monolithic.py --data-path data/adult.csv\n"
        ),
    )
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to the raw UCI Adult CSV dataset.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Destination path for the serialized model artifact.",
    )
    parser.add_argument(
        "--metrics-path", type=Path, default=DEFAULT_METRICS_PATH,
        help="Destination path for the metrics JSON file.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees in the RandomForest (default: 100).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum tree depth (default: 10).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def run_monolithic(args: argparse.Namespace) -> dict[str, Any]:
    """
    Execute the complete monolithic ML workflow end-to-end.

    Stages (all executed sequentially with no caching):
    1. Load raw dataset
    2. Clean and normalize data
    3. Encode features
    4. Split train/test
    5. Train model
    6. Evaluate metrics
    7. Save model artifact
    8. Save metrics JSON

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict[str, Any]
        Final evaluation metrics.
    """
    pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("MONOLITHIC ML PIPELINE — START")
    logger.info("Config: n_estimators=%d | max_depth=%d | test_size=%.2f | seed=%d",
                args.n_estimators, args.max_depth, args.test_size, args.random_state)
    logger.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────
    t0 = time.perf_counter()
    df = _load_data(args.data_path)
    logger.info("Step 1 — Load: %.3fs", time.perf_counter() - t0)

    # ── 2. Clean ──────────────────────────────────────────
    t0 = time.perf_counter()
    df = _clean_data(df)
    logger.info("Step 2 — Clean: %.3fs", time.perf_counter() - t0)

    # ── 3. Encode ─────────────────────────────────────────
    t0 = time.perf_counter()
    X, y = _encode_features(df, target_col="income")
    logger.info("Step 3 — Encode: %.3fs", time.perf_counter() - t0)

    # ── 4. Split ──────────────────────────────────────────
    t0 = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    logger.info("Step 4 — Split: %.3fs | train=%d | test=%d",
                time.perf_counter() - t0, len(X_train), len(X_test))

    # ── 5. Train ──────────────────────────────────────────
    t0 = time.perf_counter()
    model = _train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    logger.info("Step 5 — Train: %.3fs", time.perf_counter() - t0)

    # ── 6. Evaluate ───────────────────────────────────────
    t0 = time.perf_counter()
    metrics = _evaluate_model(model, X_test, y_test)
    logger.info("Step 6 — Evaluate: %.3fs", time.perf_counter() - t0)

    # ── 7. Save model ─────────────────────────────────────
    t0 = time.perf_counter()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    logger.info("Step 7 - Save model: %.3fs >> '%s'", time.perf_counter() - t0, args.model_path)

    # ── 8. Save metrics ───────────────────────────────────
    t0 = time.perf_counter()
    total_wall_clock = time.perf_counter() - pipeline_start
    metrics["total_wall_clock_seconds"] = round(total_wall_clock, 3)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Step 8 - Save metrics: %.3fs >> '%s'",
                time.perf_counter() - t0, args.metrics_path)

    # ── Summary ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MONOLITHIC PIPELINE COMPLETE")
    logger.info("  Total wall-clock time : %.3fs", total_wall_clock)
    logger.info("  Accuracy              : %.4f", metrics["accuracy"])
    logger.info("  AUC                   : %.4f", metrics["auc"])
    logger.info("  F1 Macro              : %.4f", metrics["f1_macro"])
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    args = parse_args()
    try:
        run_monolithic(args)
    except Exception as exc:
        logger.error("Monolithic pipeline FAILED: %s", exc, exc_info=True)
        sys.exit(1)
