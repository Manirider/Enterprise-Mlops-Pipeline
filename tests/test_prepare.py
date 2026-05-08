"""
Enterprise MLOps Pipeline — Test Suite: Data Preparation
=========================================================
Unit tests for src/prepare.py covering:
  - Raw data loading (UCI Adult format variants)
  - Data cleaning (null removal, whitespace, duplicates)
  - Data validation (schema, target classes)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Return a small representative raw dataset."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, None],
            "workclass": [" State-gov", " Self-emp-not-inc", " Private", " Private", " ?"],
            "fnlwgt": [77516, 83311, 215646, 234721, 100000],
            "education": [" Bachelors", " Bachelors", " HS-grad", " 11th", " Bachelors"],
            "education_num": [13, 13, 9, 7, 13],
            "marital_status": [" Never-married", " Married-civ-spouse", " Divorced",
                               " Married-civ-spouse", " Never-married"],
            "occupation": [" Adm-clerical", " Exec-managerial", " Handlers-cleaners",
                           " Handlers-cleaners", " ?"],
            "relationship": [" Not-in-family", " Husband", " Not-in-family",
                             " Husband", " Not-in-family"],
            "race": [" White", " White", " White", " Black", " White"],
            "sex": [" Male", " Male", " Male", " Male", " Female"],
            "capital_gain": [2174, 0, 0, 0, 0],
            "capital_loss": [0, 0, 0, 0, 0],
            "hours_per_week": [40, 13, 40, 40, 40],
            "native_country": [" United-States"] * 5,
            "income": [" <=50K", " <=50K", " <=50K", " >50K", " >50K."],
        }
    )


@pytest.fixture
def clean_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned version of the sample DataFrame."""
    from src.prepare import clean_data
    return clean_data(sample_raw_df.copy(), target_col="income")


# ---------------------------------------------------------------------------
# Tests: load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:
    """Tests for the raw data loader."""

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """Must raise FileNotFoundError for a non-existent path."""
        from src.prepare import load_raw_data
        with pytest.raises(FileNotFoundError, match="Raw data not found"):
            load_raw_data(tmp_path / "missing.csv")

    def test_loads_csv_without_header(self, tmp_path: Path) -> None:
        """Must correctly parse a headerless UCI-format CSV."""
        from src.prepare import load_raw_data, COLUMN_NAMES

        # Write a headerless CSV
        csv_content = (
            "39, State-gov, 77516, Bachelors, 13, Never-married, "
            "Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, "
            "United-States, <=50K\n"
        )
        data_file = tmp_path / "adult.csv"
        data_file.write_text(csv_content)

        df = load_raw_data(data_file)
        assert list(df.columns) == COLUMN_NAMES
        assert len(df) == 1

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """Returned value must be a pandas DataFrame."""
        from src.prepare import load_raw_data

        csv_content = (
            "39, State-gov, 77516, Bachelors, 13, Never-married, "
            "Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, "
            "United-States, <=50K\n"
        )
        data_file = tmp_path / "adult.csv"
        data_file.write_text(csv_content)

        result = load_raw_data(data_file)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    """Tests for the data cleaning function."""

    def test_drops_null_rows(self, sample_raw_df: pd.DataFrame) -> None:
        """Rows containing nulls must be removed."""
        from src.prepare import clean_data
        cleaned = clean_data(sample_raw_df.copy(), target_col="income")
        assert cleaned.isnull().sum().sum() == 0

    def test_strips_whitespace(self, sample_raw_df: pd.DataFrame) -> None:
        """String columns must have leading/trailing whitespace stripped."""
        from src.prepare import clean_data
        cleaned = clean_data(sample_raw_df.copy(), target_col="income")
        for col in cleaned.select_dtypes(include="object").columns:
            assert not any(cleaned[col].str.startswith(" "))
            assert not any(cleaned[col].str.endswith(" "))

    def test_normalizes_income_target(self, sample_raw_df: pd.DataFrame) -> None:
        """Income label trailing periods (UCI artifact) must be removed."""
        from src.prepare import clean_data
        cleaned = clean_data(sample_raw_df.copy(), target_col="income")
        assert not any(cleaned["income"].str.endswith("."))

    def test_result_is_dataframe(self, sample_raw_df: pd.DataFrame) -> None:
        """Return type must be a pandas DataFrame."""
        from src.prepare import clean_data
        result = clean_data(sample_raw_df.copy(), target_col="income")
        assert isinstance(result, pd.DataFrame)

    def test_non_empty_after_cleaning(self, sample_raw_df: pd.DataFrame) -> None:
        """Dataset must not be empty after cleaning valid data."""
        from src.prepare import clean_data
        result = clean_data(sample_raw_df.copy(), target_col="income")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    """Tests for the data validation function."""

    def test_passes_on_clean_data(self, clean_df: pd.DataFrame) -> None:
        """Validation must pass silently on a properly cleaned DataFrame."""
        from src.prepare import validate_data
        validate_data(clean_df, target_col="income")  # Should not raise

    def test_raises_on_empty_dataframe(self) -> None:
        """Must raise ValueError when the DataFrame is empty."""
        from src.prepare import validate_data
        with pytest.raises(ValueError, match="empty"):
            validate_data(pd.DataFrame(), target_col="income")

    def test_raises_when_target_missing(self, clean_df: pd.DataFrame) -> None:
        """Must raise ValueError when target column is absent."""
        from src.prepare import validate_data
        df_no_target = clean_df.drop(columns=["income"])
        with pytest.raises(ValueError, match="missing from dataset"):
            validate_data(df_no_target, target_col="income")

    def test_raises_on_remaining_nulls(self, clean_df: pd.DataFrame) -> None:
        """Must raise ValueError when nulls are still present."""
        from src.prepare import validate_data
        clean_df.loc[0, "age"] = None
        with pytest.raises(ValueError, match="null values"):
            validate_data(clean_df, target_col="income")
