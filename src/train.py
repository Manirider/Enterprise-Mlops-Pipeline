from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_param
from src.utils.helpers import save_model, set_seed, timer
from src.utils.logger import get_logger
from src.utils.paths import Paths
logger = get_logger(__name__)

def load_features(features_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not features_path.exists():
        raise FileNotFoundError(f"Features archive not found at '{features_path}'. Run 'dvc repro featurize' first.")
    data = np.load(features_path, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = list(data['feature_names'])
    logger.info('Features loaded — X_train%s | X_test%s | y_train%s | y_test%s', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return (X_train, X_test, y_train, y_test, feature_names)

def build_lightgbm_model(params: dict):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    model = lgb.LGBMClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        subsample=float(params['subsample']),
        colsample_bytree=float(params['colsample_bytree']),
        min_child_samples=int(params['min_child_samples']),
        reg_alpha=float(params['reg_alpha']),
        reg_lambda=float(params['reg_lambda']),
        n_jobs=int(params['n_jobs']),
        random_state=int(params['random_state']),
        verbose=-1,
    )
    logger.info('LGBMClassifier — n_estimators=%d | max_depth=%d | lr=%.3f | subsample=%.2f',
                model.n_estimators, model.max_depth, model.learning_rate, model.subsample)
    return model

def build_random_forest_model(params: dict) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=params['max_depth'] if params['max_depth'] != 'null' else None,
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        n_jobs=int(params['n_jobs']),
        random_state=int(params['random_state']),
        class_weight='balanced',
    )
    logger.info('RandomForestClassifier — n_estimators=%d | max_depth=%s | max_features=%s',
                model.n_estimators, model.max_depth, model.max_features)
    return model

def build_model(model_type: str, params: dict):
    if model_type == 'lightgbm':
        return build_lightgbm_model(params)
    elif model_type == 'random_forest':
        return build_random_forest_model(params)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lightgbm' or 'random_forest'.")

@timer
def run_train() -> None:
    random_state = get_param('base.random_state')
    features_path = Path(get_param('data.features_path'))
    model_path = Path(get_param('evaluate.model_path'))
    model_type = get_param('train.model_type')
    model_params = {
        'model_type':        model_type,
        'n_estimators':      get_param('train.n_estimators'),
        'max_depth':         get_param('train.max_depth'),
        'learning_rate':     get_param('train.learning_rate'),
        'subsample':         get_param('train.subsample'),
        'colsample_bytree':  get_param('train.colsample_bytree'),
        'min_child_samples': get_param('train.min_child_samples'),
        'reg_alpha':         get_param('train.reg_alpha'),
        'reg_lambda':        get_param('train.reg_lambda'),
        'min_samples_split': get_param('train.min_samples_split'),
        'min_samples_leaf':  get_param('train.min_samples_leaf'),
        'max_features':      get_param('train.max_features'),
        'n_jobs':            get_param('train.n_jobs'),
        'random_state':      random_state,
    }
    set_seed(random_state)
    Paths.ensure_dirs()
    X_train, X_test, y_train, y_test, feature_names = load_features(features_path)
    model = build_model(model_type, model_params)
    logger.info('Training %s on %d samples × %d features...', model_type, X_train.shape[0], X_train.shape[1])
    t0 = time.perf_counter()
    # Use DataFrame to preserve feature names — avoids sklearn validation warnings
    import pandas as pd
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_names)
    model.fit(X_train_df, y_train)
    train_time = time.perf_counter() - t0
    logger.info('Training complete in %.2fs', train_time)
    train_score = model.score(X_train_df, y_train)
    test_score  = model.score(X_test_df,  y_test)
    logger.info('Quick scores — train_acc=%.4f | test_acc=%.4f', train_score, test_score)
    save_model(model, model_path)

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('STAGE 3 — Model Training')
    logger.info('=' * 60)
    try:
        run_train()
        logger.info('Stage 3 completed successfully ✓')
    except Exception as exc:
        logger.error('Stage 3 FAILED: %s', exc, exc_info=True)
        sys.exit(1)