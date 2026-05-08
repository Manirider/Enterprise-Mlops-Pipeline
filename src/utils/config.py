from __future__ import annotations
import functools
from pathlib import Path
from typing import Any
import yaml
from src.utils.logger import get_logger
logger = get_logger(__name__)
DEFAULT_PARAMS_PATH = Path('params.yaml')

@functools.lru_cache(maxsize=1)
def load_params(params_path: Path=DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found at '{params_path}'. Ensure you are running from the project root.")
    logger.debug("Loading parameters from '%s'", params_path)
    with params_path.open('r', encoding='utf-8') as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    logger.info('Parameters loaded successfully (%d top-level keys)', len(params))
    return params

def get_param(dotted_key: str, params_path: Path=DEFAULT_PARAMS_PATH) -> Any:
    params = load_params(params_path)
    keys = dotted_key.split('.')
    value: Any = params
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"Parameter key '{dotted_key}' not found. Failed at segment '{key}'.") from exc
    return value