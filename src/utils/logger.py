"""
Enterprise MLOps Pipeline — Centralized Logger
===============================================
Provides a structured, file+console logging system used
across all pipeline stages and scripts. Follows Python
logging best practices with stage-aware formatting.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Stage started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "pipeline.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Internal registry to avoid duplicate handlers ──────────
_configured_loggers: set[str] = set()


def _ensure_log_dir() -> None:
    """Create the logs directory if it does not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Return a named logger with file and console handlers attached.

    The function is idempotent — calling it multiple times with
    the same ``name`` will not add duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int, optional
        Logging level (e.g., ``logging.DEBUG``).
        Defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    _ensure_log_dir()

    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File handler ───────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ── Console (stdout) handler ───────────────────────────
    import io
    stdout_stream = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    console_handler = logging.StreamHandler(stdout_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _configured_loggers.add(name)
    return logger
