from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def small_features():
    rng = np.random.default_rng(42)
    X = rng.random((200, 14)).astype(np.float64)
    y = rng.integers(0, 2, size=200).astype(np.int32)
    return (X[:160], X[160:], y[:160], y[160:])

@pytest.fixture
def features_npz(small_features, tmp_path):
    X_train, X_test, y_train, y_test = small_features
    path = tmp_path / 'features.npz'
    np.savez_compressed(path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=np.array([f'f{i}' for i in range(14)]))
    return path

class TestBuildModel:

    def test_returns_random_forest(self):
        from src.train import build_model
        params = {'n_estimators': 10, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'random_state': 42}
        assert isinstance(build_model(params), RandomForestClassifier)

    def test_params_applied(self):
        from src.train import build_model
        params = {'n_estimators': 25, 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 99}
        model = build_model(params)
        assert model.n_estimators == 25
        assert model.max_depth == 7
        assert model.random_state == 99

    def test_model_is_unfitted(self):
        from sklearn.exceptions import NotFittedError
        from src.train import build_model
        params = {'n_estimators': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'random_state': 0}
        model = build_model(params)
        with pytest.raises(NotFittedError):
            model.predict(np.random.rand(10, 5))

class TestLoadFeatures:

    def test_raises_when_missing(self, tmp_path):
        from src.train import load_features
        with pytest.raises(FileNotFoundError):
            load_features(tmp_path / 'nonexistent.npz')

    def test_correct_shapes(self, features_npz):
        from src.train import load_features
        X_train, X_test, y_train, y_test, names = load_features(features_npz)
        assert X_train.shape == (160, 14)
        assert X_test.shape == (40, 14)
        assert len(names) == 14

    def test_returns_five_values(self, features_npz):
        from src.train import load_features
        assert len(load_features(features_npz)) == 5

class TestModelTraining:

    def test_fits_without_error(self, small_features):
        from src.train import build_model
        X_train, X_test, y_train, _ = small_features
        params = {'n_estimators': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'random_state': 42}
        model = build_model(params)
        model.fit(X_train, y_train)
        assert len(model.predict(X_test)) == len(X_test)

    def test_predictions_binary(self, small_features):
        from src.train import build_model
        X_train, X_test, y_train, _ = small_features
        params = {'n_estimators': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'random_state': 42}
        model = build_model(params)
        model.fit(X_train, y_train)
        assert set(np.unique(model.predict(X_test))).issubset({0, 1})

    def test_serialization_roundtrip(self, small_features, tmp_path):
        from src.train import build_model
        X_train, X_test, y_train, _ = small_features
        params = {'n_estimators': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'random_state': 42}
        model = build_model(params)
        model.fit(X_train, y_train)
        path = tmp_path / 'model.joblib'
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(model.predict(X_test), loaded.predict(X_test))