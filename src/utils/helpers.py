"""
Enterprise MLOps Pipeline — General Helper Utilities
====================================================
Reusable helper functions used across pipeline stages:
timing decorators, reproducibility seeders, joblib I/O,
and dataset introspection utilities.

Usage:
    from src.utils.helpers import timer, set_seed, save_model, load_model
"""

from __future__ import annotations

import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Timing Decorator ───────────────────────────────────────

def timer(func: Callable) -> Callable:
    """
    Decorator that logs the execution time of a function.

    Example
    -------
    @timer
    def my_stage():
        ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        logger.info(">> Starting '%s'", func.__name__)
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("[OK] Finished '%s' in %.3fs", func.__name__, elapsed)
        return result
    return wrapper


# ── Reproducibility ────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Covers Python's random module, NumPy, and common
    environment variables for deterministic behaviour.

    Parameters
    ----------
    seed : int
        Seed value to apply globally.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Global random seed set to %d", seed)


# ── Joblib Model I/O ───────────────────────────────────────

def save_model(model: Any, path: Path) -> None:
    """
    Serialize a fitted scikit-learn model using joblib.

    Parameters
    ----------
    model : Any
        Fitted estimator object.
    path : Path
        Destination file path (e.g., ``models/model.joblib``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    size_kb = path.stat().st_size / 1024
    logger.info("Model saved to '%s' (%.1f KB)", path, size_kb)


def load_model(path: Path) -> Any:
    """
    Deserialize a joblib-serialized model.

    Parameters
    ----------
    path : Path
        Path to the ``.joblib`` artifact.

    Returns
    -------
    Any
        The deserialized sklearn estimator.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Run the train stage first."
        )
    model = joblib.load(path, mmap_mode="r")
    logger.info("Model loaded from '%s'", path)
    return model


# ── Dataset Introspection ──────────────────────────────────

def log_dataset_info(df: Any, label: str = "dataset") -> None:
    """
    Log shape, dtypes, and null counts of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to introspect.
    label : str
        Descriptive label for logging context.
    """
    logger.info(
        "[%s] shape=%s | nulls=%d | dtypes=%s",
        label,
        df.shape,
        int(df.isnull().sum().sum()),
        dict(df.dtypes.value_counts()),
    )
