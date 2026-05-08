"""
Enterprise MLOps Pipeline — Path Registry
==========================================
Centralizes all project path constants derived from params.yaml
so every module references the same canonical locations.

Usage:
    from src.utils.paths import Paths
    df = pd.read_csv(Paths.PROCESSED_DATA)
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import get_param


class Paths:
    """
    Static namespace for all canonical project paths.

    All values are resolved from params.yaml at import time,
    ensuring that a single change in params propagates everywhere.
    """

    # ── Raw and processed data ─────────────────────────────
    RAW_DATA: Path = Path(get_param("data.raw_data_path"))
    PROCESSED_DATA: Path = Path(get_param("data.processed_data_path"))
    FEATURES: Path = Path(get_param("data.features_path"))

    # ── Model artifacts ────────────────────────────────────
    MODEL: Path = Path(get_param("evaluate.model_path"))

    # ── Metrics output ─────────────────────────────────────
    METRICS: Path = Path(get_param("evaluate.metrics_path"))

    # ── Log directory ──────────────────────────────────────
    LOGS: Path = Path("logs")
    LOG_FILE: Path = LOGS / "pipeline.log"

    # ── Reports directory ──────────────────────────────────
    REPORTS: Path = Path("reports")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all output directories if they do not exist."""
        for p in [
            cls.PROCESSED_DATA.parent,
            cls.MODEL.parent,
            cls.METRICS.parent,
            cls.LOGS,
            cls.REPORTS,
        ]:
            p.mkdir(parents=True, exist_ok=True)
