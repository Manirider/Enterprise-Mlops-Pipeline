from __future__ import annotations
from pathlib import Path
from src.utils.config import get_param

class Paths:
    RAW_DATA: Path = Path(get_param('data.raw_data_path'))
    PROCESSED_DATA: Path = Path(get_param('data.processed_data_path'))
    FEATURES: Path = Path(get_param('data.features_path'))
    MODEL: Path = Path(get_param('evaluate.model_path'))
    METRICS: Path = Path(get_param('evaluate.metrics_path'))
    LOGS: Path = Path('logs')
    LOG_FILE: Path = LOGS / 'pipeline.log'
    REPORTS: Path = Path('reports')

    @classmethod
    def ensure_dirs(cls) -> None:
        for p in [cls.PROCESSED_DATA.parent, cls.MODEL.parent, cls.METRICS.parent, cls.LOGS, cls.REPORTS]:
            p.mkdir(parents=True, exist_ok=True)