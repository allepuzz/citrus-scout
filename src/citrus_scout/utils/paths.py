"""Canonical project paths.

Resolved from this file's location rather than the working directory, so scripts
behave the same whether they are launched from the repo root, from a notebook, or
from a Colab cell after cloning.
"""

from __future__ import annotations

from pathlib import Path

# src/citrus_scout/utils/paths.py -> repo root is four levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"


def ensure_dir(path: Path) -> Path:
    """Create `path` if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
