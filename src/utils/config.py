"""
Enterprise MLOps Pipeline — Configuration Loader
=================================================
Loads and validates params.yaml centrally so every pipeline
stage reads from a single, consistent source of truth.

Usage:
    from src.utils.config import load_params, get_param
    params = load_params()
    n_est = get_param("model.n_estimators")
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Default params file location ───────────────────────────
DEFAULT_PARAMS_PATH = Path("params.yaml")


@functools.lru_cache(maxsize=1)
def load_params(params_path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """
    Load and cache params.yaml into a nested dictionary.

    Parameters
    ----------
    params_path : Path
        Path to the YAML parameters file.

    Returns
    -------
    dict[str, Any]
        Parsed parameters as a nested Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the params file does not exist.
    yaml.YAMLError
        If the file cannot be parsed.
    """
    if not params_path.exists():
        raise FileNotFoundError(
            f"Parameters file not found at '{params_path}'. "
            "Ensure you are running from the project root."
        )

    logger.debug("Loading parameters from '%s'", params_path)
    with params_path.open("r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Parameters loaded successfully (%d top-level keys)", len(params))
    return params


def get_param(dotted_key: str, params_path: Path = DEFAULT_PARAMS_PATH) -> Any:
    """
    Retrieve a single parameter using dot-notation.

    Example
    -------
    >>> get_param("model.n_estimators")
    100

    Parameters
    ----------
    dotted_key : str
        Dot-separated key path, e.g., ``"model.n_estimators"``.
    params_path : Path
        Path to the YAML parameters file.

    Returns
    -------
    Any
        The resolved parameter value.

    Raises
    ------
    KeyError
        If the key path does not exist in the parameters.
    """
    params = load_params(params_path)
    keys = dotted_key.split(".")
    value: Any = params
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"Parameter key '{dotted_key}' not found. "
                f"Failed at segment '{key}'."
            ) from exc
    return value
