from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_param
from src.utils.helpers import log_dataset_info, set_seed, timer
from src.utils.logger import get_logger
from src.utils.paths import Paths
logger = get_logger(__name__)

def load_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at '{path}'. Run 'dvc repro prepare' first.")
    df = pd.read_csv(path)
    logger.info('Loaded processed data: %d rows × %d columns', *df.shape)
    return df

def encode_features(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logger.info("Encoding features — target column: '%s'", target_col)
    X_df = df.drop(columns=[target_col])
    y_raw = df[target_col]
    feature_names = list(X_df.columns)
    income_map = {'<=50K': 0, '>50K': 1}
    y = y_raw.map(income_map)
    if y.isnull().any():
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), name=target_col)
        logger.warning('Target column encoded via LabelEncoder. Classes: %s', list(le.classes_))
    y_array = y.to_numpy(dtype=np.int32)
    cat_cols = X_df.select_dtypes(include='object').columns.tolist()
    num_cols = X_df.select_dtypes(exclude='object').columns.tolist()
    logger.info('Feature types — numeric: %d | categorical: %d', len(num_cols), len(cat_cols))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat = enc.fit_transform(X_df[cat_cols])
        X_num = X_df[num_cols].to_numpy(dtype=np.float64)
        X = np.hstack([X_num, X_cat])
        feature_names = num_cols + cat_cols
    else:
        X = X_df.to_numpy(dtype=np.float64)
    logger.info('Feature matrix shape: %s | Label vector shape: %s', X.shape, y_array.shape)
    return (X, y_array, feature_names)

def split_data(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info('Split — train: %d rows | test: %d rows | test_size=%.2f', len(X_train), len(X_test), test_size)
    return (X_train, X_test, y_train, y_test)

@timer
def run_featurize() -> None:
    random_state = get_param('base.random_state')
    processed_path = Path(get_param('data.processed_data_path'))
    features_path = Path(get_param('data.features_path'))
    target_col = get_param('data.target_column')
    test_size = get_param('data.test_size')
    set_seed(random_state)
    Paths.ensure_dirs()
    df = load_processed(processed_path)
    log_dataset_info(df, label='processed')
    X, y, feature_names = encode_features(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=np.array(feature_names))
    logger.info("Features saved to '%s' — arrays: X_train%s, X_test%s, y_train%s, y_test%s", features_path, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('STAGE 2 — Feature Engineering')
    logger.info('=' * 60)
    try:
        run_featurize()
        logger.info('Stage 2 completed successfully ✓')
    except Exception as exc:
        logger.error('Stage 2 FAILED: %s', exc, exc_info=True)
        sys.exit(1)