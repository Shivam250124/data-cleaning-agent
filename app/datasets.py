from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.config import DIFFICULTY_CONFIG
from app.exceptions import DatasetLoadError
from app.models import DifficultyLevel


class DatasetRegistry:
    """Loads and caches all six CSV files at startup."""

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir is None:
            # Default: project root (parent of the app/ directory)
            base_dir = str(Path(__file__).parent.parent)

        self._dirty: dict[DifficultyLevel, pd.DataFrame] = {}
        self._clean: dict[DifficultyLevel, pd.DataFrame] = {}

        for difficulty, cfg in DIFFICULTY_CONFIG.items():
            dirty_path = os.path.join(base_dir, cfg["dirty"])
            clean_path = os.path.join(base_dir, cfg["clean"])

            for path in (dirty_path, clean_path):
                if not os.path.exists(path):
                    raise DatasetLoadError(f"Dataset file not found: {path}")

            self._dirty[difficulty] = pd.read_csv(dirty_path)
            self._clean[difficulty] = pd.read_csv(clean_path)

    def get_dirty(self, difficulty: DifficultyLevel) -> pd.DataFrame:
        """Return a deep copy of the cached dirty DataFrame for the given difficulty."""
        return self._dirty[difficulty].copy(deep=True)

    def get_clean(self, difficulty: DifficultyLevel) -> pd.DataFrame:
        """Return a deep copy of the cached clean DataFrame for the given difficulty."""
        return self._clean[difficulty].copy(deep=True)
