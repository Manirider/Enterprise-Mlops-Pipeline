from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def pipeline_dir(tmp_path):
    import yaml, os, shutil
    params = {'base': {'random_state': 42, 'log_level': 'INFO', 'project_name': 'test'}, 'data': {'raw_data_path': str(tmp_path / 'data' / 'adult.csv'), 'processed_data_path': str(tmp_path / 'data' / 'processed.csv'), 'features_path': str(tmp_path / 'data' / 'features.npz'), 'target_column': 'income', 'test_size': 0.2}, 'featurize': {'add_interaction_features': True}, 'train': {'model_type': 'random_forest', 'n_estimators': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': 1, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_samples': 20, 'reg_alpha': 0.0, 'reg_lambda': 0.0}, 'evaluate': {'metrics_path': str(tmp_path / 'metrics' / 'scores.json'), 'model_path': str(tmp_path / 'models' / 'model.joblib'), 'threshold': 0.5}}
    (tmp_path / 'params.yaml').write_text(yaml.dump(params))
    for d in ['data', 'models', 'metrics', 'logs']:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({'age': rng.integers(18, 90, n), 'workclass': rng.choice(['Private', 'Self-emp-not-inc', 'State-gov'], n), 'fnlwgt': rng.integers(50000, 500000, n), 'education': rng.choice(['Bachelors', 'HS-grad', 'Masters'], n), 'education_num': rng.integers(5, 16, n), 'marital_status': rng.choice(['Never-married', 'Married-civ-spouse', 'Divorced'], n), 'occupation': rng.choice(['Exec-managerial', 'Prof-specialty', 'Adm-clerical'], n), 'relationship': rng.choice(['Husband', 'Not-in-family', 'Wife'], n), 'race': rng.choice(['White', 'Black', 'Asian-Pac-Islander'], n), 'sex': rng.choice(['Male', 'Female'], n), 'capital_gain': rng.integers(0, 100000, n), 'capital_loss': rng.integers(0, 4000, n), 'hours_per_week': rng.integers(10, 80, n), 'native_country': ['United-States'] * n, 'income': rng.choice(['<=50K', '>50K'], n)})
    df.to_csv(tmp_path / 'data' / 'adult.csv', index=False)
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)

class TestFullPipelineIntegration:

    def test_prepare_creates_processed_csv(self, pipeline_dir):
        import importlib, sys
        if 'src.utils.config' in sys.modules:
            del sys.modules['src.utils.config']
        if 'src.utils.paths' in sys.modules:
            del sys.modules['src.utils.paths']
        if 'src.prepare' in sys.modules:
            del sys.modules['src.prepare']
        from src.prepare import run_prepare
        run_prepare()
        assert (pipeline_dir / 'data' / 'processed.csv').exists()

    def test_featurize_creates_npz(self, pipeline_dir):
        import sys
        for mod in ['src.utils.config', 'src.utils.paths', 'src.prepare', 'src.featurize']:
            if mod in sys.modules:
                del sys.modules[mod]
        from src.prepare import run_prepare
        from src.featurize import run_featurize
        run_prepare()
        run_featurize()
        assert (pipeline_dir / 'data' / 'features.npz').exists()

    def test_train_creates_model(self, pipeline_dir):
        import sys
        for mod in list(sys.modules.keys()):
            if mod.startswith('src.'):
                del sys.modules[mod]
        from src.prepare import run_prepare
        from src.featurize import run_featurize
        from src.train import run_train
        run_prepare()
        run_featurize()
        run_train()
        assert (pipeline_dir / 'models' / 'model.joblib').exists()

    def test_evaluate_creates_metrics(self, pipeline_dir):
        import sys, json
        for mod in list(sys.modules.keys()):
            if mod.startswith('src.'):
                del sys.modules[mod]
        from src.prepare import run_prepare
        from src.featurize import run_featurize
        from src.train import run_train
        from src.evaluate import run_evaluate
        run_prepare()
        run_featurize()
        run_train()
        run_evaluate()
        metrics_path = pipeline_dir / 'metrics' / 'scores.json'
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        for key in ['accuracy', 'f1_macro', 'f1_binary']:
            assert key in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0