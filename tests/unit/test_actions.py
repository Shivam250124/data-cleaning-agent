"""Unit tests for action executors."""

import numpy as np
import pandas as pd
import pytest

from app.actions import execute_action
from app.exceptions import ColumnNotFoundError
from app.models import ActionType


class TestDropDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = execute_action(df, ActionType.drop_duplicates, {})
        assert len(result) == 2
        assert list(result["a"]) == [1, 2]

    def test_subset_parameter(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "y"]})
        result = execute_action(df, ActionType.drop_duplicates, {"subset": ["a"]})
        assert len(result) == 2

    def test_keep_last(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["first", "last", "other"]})
        result = execute_action(df, ActionType.drop_duplicates, {"subset": ["a"], "keep": "last"})
        assert result.iloc[0]["b"] == "last"

    def test_invalid_subset_column_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            execute_action(df, ActionType.drop_duplicates, {"subset": ["nonexistent"]})


class TestFillMissing:
    def test_fill_with_value(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = execute_action(df, ActionType.fill_missing, {"column": "a", "value": 0})
        assert result["a"].isna().sum() == 0
        assert result.iloc[1]["a"] == 0

    def test_fill_with_mean(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = execute_action(df, ActionType.fill_missing, {"column": "a", "strategy": "mean"})
        assert result.iloc[1]["a"] == 2.0

    def test_fill_with_mode(self):
        df = pd.DataFrame({"a": ["x", "x", np.nan, "y"]})
        result = execute_action(df, ActionType.fill_missing, {"column": "a", "strategy": "mode"})
        assert result.iloc[2]["a"] == "x"

    def test_missing_column_param_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="column"):
            execute_action(df, ActionType.fill_missing, {"value": 0})

    def test_missing_value_and_strategy_raises(self):
        df = pd.DataFrame({"a": [1, np.nan]})
        with pytest.raises(ValueError):
            execute_action(df, ActionType.fill_missing, {"column": "a"})

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            execute_action(df, ActionType.fill_missing, {"column": "b", "value": 0})


class TestCastColumn:
    def test_cast_to_int(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = execute_action(df, ActionType.cast_column, {"column": "a", "dtype": "int"})
        assert result["a"].dtype == "Int64"

    def test_cast_to_float(self):
        df = pd.DataFrame({"a": ["1.5", "2.5"]})
        result = execute_action(df, ActionType.cast_column, {"column": "a", "dtype": "float"})
        assert result["a"].dtype == float

    def test_cast_coerces_invalid(self):
        df = pd.DataFrame({"a": ["1", "abc", "3"]})
        result = execute_action(df, ActionType.cast_column, {"column": "a", "dtype": "float"})
        assert pd.isna(result.iloc[1]["a"])

    def test_missing_params_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="column"):
            execute_action(df, ActionType.cast_column, {"dtype": "int"})
        with pytest.raises(ValueError, match="dtype"):
            execute_action(df, ActionType.cast_column, {"column": "a"})


class TestRenameColumn:
    def test_renames_column(self):
        df = pd.DataFrame({"old": [1, 2]})
        result = execute_action(df, ActionType.rename_column, {"old_name": "old", "new_name": "new"})
        assert "new" in result.columns
        assert "old" not in result.columns

    def test_missing_old_name_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="old_name"):
            execute_action(df, ActionType.rename_column, {"new_name": "b"})

    def test_invalid_old_name_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            execute_action(df, ActionType.rename_column, {"old_name": "b", "new_name": "c"})


class TestStripWhitespace:
    def test_strips_whitespace(self):
        df = pd.DataFrame({"a": ["  hello  ", "  world  "]})
        result = execute_action(df, ActionType.strip_whitespace, {"column": "a"})
        assert result.iloc[0]["a"] == "hello"
        assert result.iloc[1]["a"] == "world"

    def test_handles_non_strings(self):
        df = pd.DataFrame({"a": ["  text  ", 123, np.nan]})
        result = execute_action(df, ActionType.strip_whitespace, {"column": "a"})
        assert result.iloc[0]["a"] == "text"
        assert result.iloc[1]["a"] == 123

    def test_missing_column_param_raises(self):
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(ValueError, match="column"):
            execute_action(df, ActionType.strip_whitespace, {})


class TestDropColumn:
    def test_drops_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = execute_action(df, ActionType.drop_column, {"column": "b"})
        assert "b" not in result.columns
        assert "a" in result.columns

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            execute_action(df, ActionType.drop_column, {"column": "b"})
