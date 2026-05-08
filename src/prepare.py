from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_param
from src.utils.helpers import log_dataset_info, set_seed, timer
from src.utils.logger import get_logger
from src.utils.paths import Paths
logger = get_logger(__name__)
COLUMN_NAMES: list[str] = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at '{path}'. Place the UCI Adult dataset (adult.csv) in the data/ directory.")
    logger.info("Loading raw data from '%s'", path)
    with path.open('r', encoding='utf-8') as fh:
        first_line = fh.readline().strip()
    has_header = first_line.split(',')[0].strip().lower() in ('age', 'age ') or not first_line.split(',')[0].strip().lstrip('-').isdigit()
    if has_header:
        df = pd.read_csv(path, na_values=['?', ' ?'])
        df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]
        if len(df.columns) == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
    else:
        df = pd.read_csv(path, names=COLUMN_NAMES, na_values=['?', ' ?'], skipinitialspace=True)
    logger.info('Raw data loaded: %d rows × %d columns', *df.shape)
    return df

def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    logger.info('Cleaning data — initial shape: %s', df.shape)
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    if target_col in df.columns:
        df[target_col] = df[target_col].str.rstrip('.')
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    logger.info('Dropped %d rows with missing values (%.1f%%)', rows_dropped, 100 * rows_dropped / rows_before)
    dups_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dups_removed = dups_before - len(df)
    if dups_removed:
        logger.warning('Removed %d duplicate rows', dups_removed)
    logger.info('Cleaned data shape: %s', df.shape)
    return df

def validate_data(df: pd.DataFrame, target_col: str) -> None:
    if df.empty:
        raise ValueError('Cleaned dataset is empty — check raw data source.')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset. Available columns: {list(df.columns)}")
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f'Dataset still contains {null_count} null values after cleaning.')
    unique_targets = df[target_col].unique()
    logger.info('Target class distribution: %s', df[target_col].value_counts().to_dict())
    if len(unique_targets) < 2:
        raise ValueError(f'Target column must have at least 2 classes; found: {unique_targets}')
    logger.info('Data validation passed ✓')

@timer
def run_prepare() -> None:
    random_state = get_param('base.random_state')
    raw_path = Path(get_param('data.raw_data_path'))
    processed_path = Path(get_param('data.processed_data_path'))
    target_col = get_param('data.target_column')
    set_seed(random_state)
    Paths.ensure_dirs()
    df = load_raw_data(raw_path)
    log_dataset_info(df, label='raw')
    df = clean_data(df, target_col)
    log_dataset_info(df, label='cleaned')
    validate_data(df, target_col)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Processed data saved to '%s' (%d rows)", processed_path, len(df))
if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('STAGE 1 — Data Preparation')
    logger.info('=' * 60)
    try:
        run_prepare()
        logger.info('Stage 1 completed successfully ✓')
    except Exception as exc:
        logger.error('Stage 1 FAILED: %s', exc, exc_info=True)
        sys.exit(1)