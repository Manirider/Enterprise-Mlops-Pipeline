import logging
import sys
from pathlib import Path
from typing import Optional
LOG_DIR = Path('logs')
LOG_FILE = LOG_DIR / 'pipeline.log'
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
_configured_loggers: set[str] = set()

def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, level: Optional[int]=None) -> logging.Logger:
    _ensure_log_dir()
    if level is None:
        level = logging.INFO
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    import io
    stdout_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'buffer') else sys.stdout
    console_handler = logging.StreamHandler(stdout_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    _configured_loggers.add(name)
    return logger