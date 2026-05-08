"""
Enterprise MLOps Pipeline — Metrics Utilities
=============================================
Reusable functions for computing, formatting, and persisting
ML evaluation metrics in JSON format.

Usage:
    from src.utils.metrics import compute_metrics, save_metrics
    m = compute_metrics(y_true, y_pred, y_prob)
    save_metrics(m, path=Paths.METRICS)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray, optional
        Predicted probabilities for the positive class.
        Required for AUC computation.

    Returns
    -------
    dict[str, Any]
        Dictionary containing accuracy, auc, f1_macro, f1_binary,
        precision, recall, and the confusion matrix.
    """
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "f1_binary": round(float(f1_score(y_true, y_pred, average="binary")), 6),
        "precision": round(float(precision_score(y_true, y_pred)), 6),
        "recall": round(float(recall_score(y_true, y_pred)), 6),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        except ValueError as exc:
            logger.warning("AUC computation failed: %s", exc)
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = report

    logger.info(
        "Metrics — accuracy=%.4f | auc=%s | f1_macro=%.4f",
        metrics["accuracy"],
        f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A",
        metrics["f1_macro"],
    )
    return metrics


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """
    Persist metrics dictionary as a formatted JSON file.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics to save.
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Metrics saved to '%s'", path)


def load_metrics(path: Path) -> dict[str, Any]:
    """
    Load metrics from a JSON file.

    Parameters
    ----------
    path : Path
        Path to the metrics JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed metrics dictionary.

    Raises
    ------
    FileNotFoundError
        If the metrics file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at '{path}'")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
