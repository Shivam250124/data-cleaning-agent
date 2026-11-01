"""Unit tests for DatasetRegistry — Requirements 3.3, 3.4."""
import os
import tempfile

import pandas as pd
import pytest

from app.datasets import DatasetRegistry
from app.exceptions import DatasetLoadError
from app.models import DifficultyLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path: str, rows: int = 5) -> None:
    df = pd.DataFrame({"a": range(rows), "b": [f"v{i}" for i in range(rows)]})
    df.to_csv(path, index=False)


def _setup_fake_data_dir(tmp_path) -> str:
    """Create a minimal fake data directory with all 6 CSVs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in (
        "easy_dirty.csv", "easy_clean.csv",
        "medium_dirty.csv", "medium_clean.csv",
        "hard_dirty.csv", "hard_clean.csv",
    ):
        _make_csv(str(data_dir / name))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests: successful load
# ---------------------------------------------------------------------------

class TestDatasetRegistryLoad:
    def test_loads_all_difficulties(self, tmp_path):
        base_dir = _setup_fake_data_dir(tmp_path)
        registry = DatasetRegistry(base_dir=base_dir)
        for difficulty in DifficultyLevel:
            dirty = registry.get_dirty(difficulty)
            clean = registry.get_clean(difficulty)
            assert isinstance(dirty, pd.DataFrame)
            assert isinstance(clean, pd.DataFrame)

    def test_returns_deep_copies(self, tmp_path):
        base_dir = _setup_fake_data_dir(tmp_path)
        registry = DatasetRegistry(base_dir=base_dir)
        df1 = registry.get_dirty(DifficultyLevel.easy)
        df2 = registry.get_dirty(DifficultyLevel.easy)
        # Mutating one copy must not affect the other
        df1.iloc[0, 0] = 9999
        df3 = registry.get_dirty(DifficultyLevel.easy)
        assert df3.iloc[0, 0] != 9999

    def test_real_data_row_counts(self):
        """Load from the actual data/ directory and verify dirty row counts match config."""
        from app.config import DIFFICULTY_CONFIG
        registry = DatasetRegistry()
        for difficulty, cfg in DIFFICULTY_CONFIG.items():
            dirty = registry.get_dirty(difficulty)
            assert len(dirty) == cfg["rows"], (
                f"{difficulty} dirty: expected {cfg['rows']} rows, got {len(dirty)}"
            )


# ---------------------------------------------------------------------------
# Tests: missing file raises DatasetLoadError
# ---------------------------------------------------------------------------

class TestDatasetRegistryMissingFile:
    def test_missing_dirty_csv_raises(self, tmp_path):
        base_dir = _setup_fake_data_dir(tmp_path)
        # Remove one dirty file
        os.remove(str(tmp_path / "data" / "easy_dirty.csv"))
        with pytest.raises(DatasetLoadError) as exc_info:
            DatasetRegistry(base_dir=base_dir)
        assert "easy_dirty.csv" in str(exc_info.value)

    def test_missing_clean_csv_raises(self, tmp_path):
        base_dir = _setup_fake_data_dir(tmp_path)
        os.remove(str(tmp_path / "data" / "medium_clean.csv"))
        with pytest.raises(DatasetLoadError) as exc_info:
            DatasetRegistry(base_dir=base_dir)
        assert "medium_clean.csv" in str(exc_info.value)

    def test_error_message_contains_full_path(self, tmp_path):
        base_dir = _setup_fake_data_dir(tmp_path)
        missing = str(tmp_path / "data" / "hard_dirty.csv")
        os.remove(missing)
        with pytest.raises(DatasetLoadError) as exc_info:
            DatasetRegistry(base_dir=base_dir)
        assert missing in str(exc_info.value)

    def test_empty_base_dir_raises(self, tmp_path):
        """A base_dir with no data/ subdirectory should raise DatasetLoadError."""
        with pytest.raises(DatasetLoadError):
            DatasetRegistry(base_dir=str(tmp_path))
