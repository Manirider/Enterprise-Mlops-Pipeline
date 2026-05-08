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

def timer(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        logger.info(">> Starting '%s'", func.__name__)
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("[OK] Finished '%s' in %.3fs", func.__name__, elapsed)
        return result
    return wrapper

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    logger.debug('Global random seed set to %d', seed)

def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    size_kb = path.stat().st_size / 1024
    logger.info("Model saved to '%s' (%.1f KB)", path, size_kb)

def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{path}'. Run the train stage first.")
    model = joblib.load(path, mmap_mode='r')
    logger.info("Model loaded from '%s'", path)
    return model

def log_dataset_info(df: Any, label: str='dataset') -> None:
    logger.info('[%s] shape=%s | nulls=%d | dtypes=%s', label, df.shape, int(df.isnull().sum().sum()), dict(df.dtypes.value_counts()))