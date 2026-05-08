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

def build_model(params: dict) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=params['max_depth'] if params['max_depth'] != 'null' else None, min_samples_split=int(params['min_samples_split']), min_samples_leaf=int(params['min_samples_leaf']), max_features=params['max_features'], n_jobs=int(params['n_jobs']), random_state=int(params['random_state']), class_weight='balanced')
    logger.info('Model instantiated — n_estimators=%d | max_depth=%s | max_features=%s', model.n_estimators, model.max_depth, model.max_features)
    return model

@timer
def run_train() -> None:
    random_state = get_param('base.random_state')
    features_path = Path(get_param('data.features_path'))
    model_path = Path(get_param('evaluate.model_path'))
    model_params = {'n_estimators': get_param('train.n_estimators'), 'max_depth': get_param('train.max_depth'), 'min_samples_split': get_param('train.min_samples_split'), 'min_samples_leaf': get_param('train.min_samples_leaf'), 'max_features': get_param('train.max_features'), 'n_jobs': get_param('train.n_jobs'), 'random_state': random_state}
    set_seed(random_state)
    Paths.ensure_dirs()
    X_train, X_test, y_train, y_test, feature_names = load_features(features_path)
    model = build_model(model_params)
    logger.info('Training RandomForestClassifier on %d samples × %d features...', X_train.shape[0], X_train.shape[1])
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info('Training complete in %.2fs | OOB score: N/A (oob_score=False)', train_time)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
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