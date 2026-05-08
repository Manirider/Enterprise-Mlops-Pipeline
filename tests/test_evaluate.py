"""
Enterprise MLOps Pipeline — Test Suite: Model Evaluation
=========================================================
Unit tests for src/evaluate.py covering:
  - compute_metrics correctness
  - metrics JSON persistence and loading
  - test data loading from NPZ
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def synthetic_preds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100).astype(np.int32)
    y_pred = y_true.copy()
    # Introduce ~10% error
    flip = rng.choice(100, size=10, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_prob = rng.random(100)
    return y_true, y_pred, y_prob


@pytest.fixture
def test_npz(tmp_path):
    rng = np.random.default_rng(42)
    X_test = rng.random((40, 14)).astype(np.float64)
    y_test = rng.integers(0, 2, size=40).astype(np.int32)
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        X_train=rng.random((160, 14)),
        X_test=X_test,
        y_train=rng.integers(0, 2, size=160),
        y_test=y_test,
        feature_names=np.array([f"f{i}" for i in range(14)]),
    )
    return path


@pytest.fixture
def fitted_model(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((200, 14))
    y = rng.integers(0, 2, size=200)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, path)
    return path


class TestComputeMetrics:
    def test_returns_all_keys(self, synthetic_preds):
        from src.utils.metrics import compute_metrics
        y_true, y_pred, y_prob = synthetic_preds
        m = compute_metrics(y_true, y_pred, y_prob)
        for key in ["accuracy", "auc", "f1_macro", "f1_binary", "precision", "recall"]:
            assert key in m

    def test_accuracy_range(self, synthetic_preds):
        from src.utils.metrics import compute_metrics
        y_true, y_pred, y_prob = synthetic_preds
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_auc_range(self, synthetic_preds):
        from src.utils.metrics import compute_metrics
        y_true, y_pred, y_prob = synthetic_preds
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m["auc"] <= 1.0

    def test_perfect_predictions(self):
        from src.utils.metrics import compute_metrics
        y = np.array([0, 1, 0, 1, 1])
        m = compute_metrics(y, y)
        assert m["accuracy"] == 1.0

    def test_no_auc_without_proba(self):
        from src.utils.metrics import compute_metrics
        y = np.array([0, 1, 0, 1])
        m = compute_metrics(y, y, y_prob=None)
        assert m["auc"] is None


class TestSaveLoadMetrics:
    def test_save_creates_file(self, tmp_path, synthetic_preds):
        from src.utils.metrics import compute_metrics, save_metrics
        y_true, y_pred, y_prob = synthetic_preds
        m = compute_metrics(y_true, y_pred, y_prob)
        path = tmp_path / "scores.json"
        save_metrics(m, path)
        assert path.exists()

    def test_load_matches_saved(self, tmp_path, synthetic_preds):
        from src.utils.metrics import compute_metrics, save_metrics, load_metrics
        y_true, y_pred, y_prob = synthetic_preds
        m = compute_metrics(y_true, y_pred, y_prob)
        path = tmp_path / "scores.json"
        save_metrics(m, path)
        loaded = load_metrics(path)
        assert abs(loaded["accuracy"] - m["accuracy"]) < 1e-6

    def test_load_raises_when_missing(self, tmp_path):
        from src.utils.metrics import load_metrics
        with pytest.raises(FileNotFoundError):
            load_metrics(tmp_path / "nonexistent.json")


class TestLoadTestData:
    def test_returns_correct_shapes(self, test_npz):
        from src.evaluate import load_test_data
        X_test, y_test = load_test_data(test_npz)
        assert X_test.shape == (40, 14)
        assert y_test.shape == (40,)

    def test_raises_when_missing(self, tmp_path):
        from src.evaluate import load_test_data
        with pytest.raises(FileNotFoundError):
            load_test_data(tmp_path / "missing.npz")
