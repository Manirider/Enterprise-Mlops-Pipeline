"""
Enterprise MLOps Pipeline — Stage 4: Model Evaluation
======================================================
Loads the trained model and held-out test features, computes
comprehensive classification metrics (accuracy, AUC, F1-macro,
precision, recall, confusion matrix), and persists results to
metrics/scores.json for DVC experiment tracking and comparison.

CLI Usage:
    python src/evaluate.py

DVC Usage:
    dvc repro evaluate
    dvc metrics show
    dvc exp show
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import get_param
from src.utils.helpers import load_model, timer
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics, save_metrics
from src.utils.paths import Paths

logger = get_logger(__name__)


def load_test_data(
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the held-out test partition from the NPZ archive.

    Parameters
    ----------
    features_path : Path
        Path to the features.npz archive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X_test, y_test arrays.

    Raises
    ------
    FileNotFoundError
        If the features archive does not exist.
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features archive not found at '{features_path}'. "
            "Run 'dvc repro featurize' first."
        )
    data = np.load(features_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    logger.info(
        "Test data loaded — X_test%s | y_test%s | positive_rate=%.3f",
        X_test.shape, y_test.shape, y_test.mean(),
    )
    return X_test, y_test


@timer
def run_evaluate() -> None:
    """
    Execute the full model evaluation stage.

    Loads the serialized model and test features, generates
    predictions and probability scores, computes metrics,
    and saves results to metrics/scores.json.
    """
    features_path = Path(get_param("data.features_path"))
    model_path = Path(get_param("evaluate.model_path"))
    metrics_path = Path(get_param("evaluate.metrics_path"))

    Paths.ensure_dirs()

    # ── Load model + test data ────────────────────────────
    model = load_model(model_path)
    X_test, y_test = load_test_data(features_path)

    # ── Generate predictions ──────────────────────────────
    y_pred = model.predict(X_test)
    y_prob: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    # ── Compute metrics ───────────────────────────────────
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # ── Add pipeline metadata ─────────────────────────────
    metrics["stage"] = "evaluate"
    metrics["n_test_samples"] = int(len(y_test))
    metrics["model_path"] = str(model_path)

    # ── Persist metrics ───────────────────────────────────
    save_metrics(metrics, path=metrics_path)

    # ── Human-readable summary ────────────────────────────
    logger.info(
        "\n%s\n"
        "  ✦ Accuracy  : %.4f\n"
        "  ✦ AUC       : %s\n"
        "  ✦ F1 Macro  : %.4f\n"
        "  ✦ F1 Binary : %.4f\n"
        "  ✦ Precision : %.4f\n"
        "  ✦ Recall    : %.4f\n"
        "%s",
        "=" * 50,
        metrics["accuracy"],
        f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A",
        metrics["f1_macro"],
        metrics["f1_binary"],
        metrics["precision"],
        metrics["recall"],
        "=" * 50,
    )


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STAGE 4 — Model Evaluation")
    logger.info("=" * 60)
    try:
        run_evaluate()
        logger.info("Stage 4 completed successfully ✓")
    except Exception as exc:
        logger.error("Stage 4 FAILED: %s", exc, exc_info=True)
        sys.exit(1)
