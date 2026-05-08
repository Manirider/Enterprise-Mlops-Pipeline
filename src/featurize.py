"""
Enterprise MLOps Pipeline — Stage 2: Feature Engineering
=========================================================
Loads the cleaned processed.csv, applies categorical encoding
(OrdinalEncoder for simplicity and speed with tree models),
performs stratified train/test split, and serializes the
resulting NumPy arrays to an NPZ archive consumed by train.py.

CLI Usage:
    python src/featurize.py

DVC Usage:
    dvc repro featurize
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import get_param
from src.utils.helpers import log_dataset_info, set_seed, timer
from src.utils.logger import get_logger
from src.utils.paths import Paths

logger = get_logger(__name__)


def load_processed(path: Path) -> pd.DataFrame:
    """
    Load the cleaned processed dataset from disk.

    Parameters
    ----------
    path : Path
        Path to processed.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.

    Raises
    ------
    FileNotFoundError
        If processed.csv does not exist (run prepare stage first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{path}'. "
            "Run 'dvc repro prepare' first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded processed data: %d rows × %d columns", *df.shape)
    return df


def encode_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Encode all categorical columns using OrdinalEncoder and
    return the feature matrix X and label vector y.

    OrdinalEncoder is preferred over OneHotEncoder for tree-based
    models (Random Forest) as it preserves feature count and
    avoids high-dimensional sparse matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with a mix of numeric and categorical columns.
    target_col : str
        Name of the binary target column (e.g., ``"income"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        - X : float64 feature matrix
        - y : int32 binary label vector (0 or 1)
        - feature_names : list of column names matching X
    """
    logger.info("Encoding features — target column: '%s'", target_col)

    # ── Separate features and target ──────────────────────
    X_df = df.drop(columns=[target_col])
    y_raw = df[target_col]

    feature_names = list(X_df.columns)

    # ── Encode target ─────────────────────────────────────
    # ">50K" → 1, "<=50K" → 0
    income_map = {"<=50K": 0, ">50K": 1}
    y = y_raw.map(income_map)
    if y.isnull().any():
        # Fallback: numeric encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), name=target_col)
        logger.warning(
            "Target column encoded via LabelEncoder. "
            "Classes: %s", list(le.classes_)
        )

    y_array = y.to_numpy(dtype=np.int32)

    # ── Encode categorical features ───────────────────────
    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(exclude="object").columns.tolist()

    logger.info(
        "Feature types — numeric: %d | categorical: %d",
        len(num_cols), len(cat_cols),
    )

    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X_cat = enc.fit_transform(X_df[cat_cols])
        X_num = X_df[num_cols].to_numpy(dtype=np.float64)
        X = np.hstack([X_num, X_cat])
        # Reorder feature names to match hstack
        feature_names = num_cols + cat_cols
    else:
        X = X_df.to_numpy(dtype=np.float64)

    logger.info("Feature matrix shape: %s | Label vector shape: %s", X.shape, y_array.shape)
    return X, y_array, feature_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified train/test split.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary label vector.
    test_size : float
        Fraction of data to reserve for testing (e.g., 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Split — train: %d rows | test: %d rows | test_size=%.2f",
        len(X_train), len(X_test), test_size,
    )
    return X_train, X_test, y_train, y_test


@timer
def run_featurize() -> None:
    """
    Execute the full feature engineering stage.

    Reads params.yaml for configuration, loads processed CSV,
    encodes categorical features, splits train/test, and saves
    all arrays to an NPZ archive.
    """
    random_state = get_param("base.random_state")
    processed_path = Path(get_param("data.processed_data_path"))
    features_path = Path(get_param("data.features_path"))
    target_col = get_param("data.target_column")
    test_size = get_param("data.test_size")

    set_seed(random_state)
    Paths.ensure_dirs()

    # ── Load ──────────────────────────────────────────────
    df = load_processed(processed_path)
    log_dataset_info(df, label="processed")

    # ── Encode ────────────────────────────────────────────
    X, y, feature_names = encode_features(df, target_col)

    # ── Split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── Persist NPZ ───────────────────────────────────────
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        features_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names),
    )
    logger.info(
        "Features saved to '%s' — arrays: X_train%s, X_test%s, y_train%s, y_test%s",
        features_path,
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STAGE 2 — Feature Engineering")
    logger.info("=" * 60)
    try:
        run_featurize()
        logger.info("Stage 2 completed successfully ✓")
    except Exception as exc:
        logger.error("Stage 2 FAILED: %s", exc, exc_info=True)
        sys.exit(1)
