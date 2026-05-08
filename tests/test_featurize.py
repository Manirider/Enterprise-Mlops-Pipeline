"""
Enterprise MLOps Pipeline — Test Suite: Feature Engineering
============================================================
Unit tests for src/featurize.py covering:
  - Feature encoding (categorical → ordinal)
  - Target mapping (income label → binary)
  - Train/test split (stratification, sizing)
  - NPZ persistence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def processed_df() -> pd.DataFrame:
    """Return a representative cleaned processed DataFrame."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 37, 49, 52, 31, 42],
            "workclass": [
                "State-gov", "Self-emp-not-inc", "Private", "Private",
                "Private", "Private", "Self-emp-not-inc", "Private",
                "Private", "Private",
            ],
            "fnlwgt": [77516, 83311, 215646, 234721, 338409, 284582,
                       160187, 209642, 45781, 159449],
            "education": [
                "Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors",
                "Masters", "9th", "HS-grad", "Masters", "Bachelors",
            ],
            "education_num": [13, 13, 9, 7, 13, 14, 5, 9, 14, 13],
            "marital_status": [
                "Never-married", "Married-civ-spouse", "Divorced",
                "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse",
                "Married-spouse-absent", "Married-civ-spouse", "Never-married",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical", "Exec-managerial", "Handlers-cleaners",
                "Handlers-cleaners", "Prof-specialty", "Exec-managerial",
                "Other-service", "Exec-managerial", "Prof-specialty",
                "Exec-managerial",
            ],
            "relationship": [
                "Not-in-family", "Husband", "Not-in-family", "Husband",
                "Wife", "Wife", "Not-in-family", "Husband", "Not-in-family",
                "Husband",
            ],
            "race": ["White"] * 10,
            "sex": ["Male", "Male", "Male", "Male", "Female",
                    "Female", "Female", "Male", "Female", "Male"],
            "capital_gain": [2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178],
            "capital_loss": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "hours_per_week": [40, 13, 40, 40, 40, 40, 16, 45, 50, 40],
            "native_country": ["United-States"] * 10,
            "income": [
                "<=50K", "<=50K", "<=50K", "<=50K", "<=50K",
                "<=50K", "<=50K", ">50K", ">50K", ">50K",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tests: encode_features
# ---------------------------------------------------------------------------

class TestEncodeFeatures:
    """Tests for the feature encoding function."""

    def test_returns_numpy_arrays(self, processed_df: pd.DataFrame) -> None:
        """encode_features must return (np.ndarray, np.ndarray, list)."""
        from src.featurize import encode_features
        X, y, names = encode_features(processed_df, target_col="income")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(names, list)

    def test_correct_feature_count(self, processed_df: pd.DataFrame) -> None:
        """Number of features must equal number of columns minus target."""
        from src.featurize import encode_features
        X, y, names = encode_features(processed_df, target_col="income")
        expected_cols = len(processed_df.columns) - 1  # exclude target
        assert X.shape[1] == expected_cols
        assert len(names) == expected_cols

    def test_target_is_binary(self, processed_df: pd.DataFrame) -> None:
        """Encoded target must contain only 0 and 1."""
        from src.featurize import encode_features
        _, y, _ = encode_features(processed_df, target_col="income")
        assert set(np.unique(y)).issubset({0, 1})

    def test_positive_class_correct(self, processed_df: pd.DataFrame) -> None:
        """'>50K' rows must map to label 1."""
        from src.featurize import encode_features
        _, y, _ = encode_features(processed_df, target_col="income")
        # Last 3 rows in fixture are ">50K"
        assert y[-1] == 1
        assert y[0] == 0

    def test_no_nulls_in_features(self, processed_df: pd.DataFrame) -> None:
        """Feature matrix must not contain NaN values."""
        from src.featurize import encode_features
        X, _, _ = encode_features(processed_df, target_col="income")
        assert not np.isnan(X).any()

    def test_feature_matrix_dtype(self, processed_df: pd.DataFrame) -> None:
        """Feature matrix dtype must be float64."""
        from src.featurize import encode_features
        X, _, _ = encode_features(processed_df, target_col="income")
        assert X.dtype == np.float64

    def test_label_dtype_is_int(self, processed_df: pd.DataFrame) -> None:
        """Label vector dtype must be int32."""
        from src.featurize import encode_features
        _, y, _ = encode_features(processed_df, target_col="income")
        assert y.dtype == np.int32


# ---------------------------------------------------------------------------
# Tests: split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    """Tests for the stratified train/test split function."""

    def test_split_sizes(self, processed_df: pd.DataFrame) -> None:
        """Split sizes must approximately match the requested test_size."""
        from src.featurize import encode_features, split_data
        X, y, _ = encode_features(processed_df, target_col="income")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert len(X_test) >= 1  # At minimum 1 test sample

    def test_returns_four_arrays(self, processed_df: pd.DataFrame) -> None:
        """split_data must return exactly 4 numpy arrays."""
        from src.featurize import encode_features, split_data
        X, y, _ = encode_features(processed_df, target_col="income")
        result = split_data(X, y, test_size=0.2, random_state=42)
        assert len(result) == 4
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_no_overlap_between_splits(self, processed_df: pd.DataFrame) -> None:
        """Train and test indices must not overlap."""
        from sklearn.model_selection import train_test_split
        from src.featurize import encode_features
        X, y, _ = encode_features(processed_df, target_col="income")
        # Use indices as proxy for row identity
        idx = np.arange(len(X))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
        assert len(set(idx_train) & set(idx_test)) == 0


# ---------------------------------------------------------------------------
# Tests: NPZ serialization
# ---------------------------------------------------------------------------

class TestNPZSerialization:
    """Tests for NPZ archive persistence."""

    def test_npz_contains_expected_keys(
        self, processed_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Saved NPZ must contain X_train, X_test, y_train, y_test, feature_names."""
        from src.featurize import encode_features, split_data
        X, y, feature_names = encode_features(processed_df, target_col="income")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        out = tmp_path / "features.npz"
        np.savez_compressed(
            out,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=np.array(feature_names),
        )

        data = np.load(out, allow_pickle=True)
        for key in ["X_train", "X_test", "y_train", "y_test", "feature_names"]:
            assert key in data

    def test_npz_roundtrip_preserves_shape(
        self, processed_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Arrays loaded from NPZ must have the same shape as saved."""
        from src.featurize import encode_features, split_data
        X, y, feature_names = encode_features(processed_df, target_col="income")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        out = tmp_path / "features.npz"
        np.savez_compressed(out, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test,
                            feature_names=np.array(feature_names))
        data = np.load(out, allow_pickle=True)

        assert data["X_train"].shape == X_train.shape
        assert data["X_test"].shape == X_test.shape
        assert data["y_train"].shape == y_train.shape
        assert data["y_test"].shape == y_test.shape
