"""
Enterprise MLOps Pipeline — Dataset Bootstrap Script
=====================================================
Downloads the UCI Adult Income dataset and generates
placeholder report files. Run this once before dvc repro.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def download_adult_dataset(dest: Path = Path("data/adult.csv")) -> None:
    """Download the UCI Adult Income dataset if not already present."""
    if dest.exists():
        print(f"Dataset already exists at '{dest}'. Skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    print(f"Downloading UCI Adult dataset from:\n  {url}")
    print("Please wait...")

    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"✓ Downloaded to '{dest}' ({size_kb:.1f} KB)")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("\nManual download instructions:")
        print(f"  1. Visit: https://archive.ics.uci.edu/ml/datasets/adult")
        print(f"  2. Download 'adult.data' and save as '{dest}'")
        sys.exit(1)


def create_placeholder_dirs() -> None:
    """Create all required output directories."""
    for d in ["data", "models", "metrics", "logs", "reports", "notebooks"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directory ready: {d}/")


if __name__ == "__main__":
    print("=" * 50)
    print("Enterprise MLOps Pipeline — Bootstrap")
    print("=" * 50)
    create_placeholder_dirs()
    download_adult_dataset()
    print("\n[OK] Bootstrap complete. Run: dvc repro")
