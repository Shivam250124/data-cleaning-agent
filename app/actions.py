"""Action executors for the Data Cleaning Agent environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from app.exceptions import ColumnNotFoundError
from app.models import ActionType


def _require_column(df: pd.DataFrame, column: str) -> None:
    """Raise ColumnNotFoundError if the column does not exist."""
    if column not in df.columns:
        raise ColumnNotFoundError(f"Column '{column}' not found in DataFrame")


def execute_action(
    df: pd.DataFrame,
    action_type: ActionType,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute a data cleaning action on the DataFrame.

    Args:
        df: The DataFrame to modify (will be modified in place and returned).
        action_type: The type of action to execute.
        params: Action-specific parameters.

    Returns:
        The modified DataFrame.

    Raises:
        ColumnNotFoundError: If a required column does not exist.
        ValueError: If parameters are invalid.
    """
    executors = {
        ActionType.drop_duplicates: _drop_duplicates,
        ActionType.fill_missing: _fill_missing,
        ActionType.cast_column: _cast_column,
        ActionType.rename_column: _rename_column,
        ActionType.strip_whitespace: _strip_whitespace,
        ActionType.drop_column: _drop_column,
    }

    executor = executors.get(action_type)
    if executor is None:
        raise ValueError(f"Unknown action type: {action_type}")

    return executor(df, params)


def _drop_duplicates(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Params:
        subset (optional): List of column names to consider for duplicates.
        keep (optional): 'first', 'last', or False. Default 'first'.
    """
    subset = params.get("subset")
    keep = params.get("keep", "first")

    if subset is not None:
        for col in subset:
            _require_column(df, col)

    df.drop_duplicates(subset=subset, keep=keep, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _fill_missing(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Fill missing values in a column.

    Params:
        column (required): Column name to fill.
        value (optional): Static value to fill with.
        strategy (optional): 'mean', 'median', 'mode', 'ffill', 'bfill'.
            If both value and strategy are provided, value takes precedence.
    """
    column = params.get("column")
    if column is None:
        raise ValueError("fill_missing requires 'column' parameter")

    _require_column(df, column)

    value = params.get("value")
    strategy = params.get("strategy")

    if value is not None:
        df[column] = df[column].fillna(value)
    elif strategy == "mean":
        numeric_col = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(numeric_col.mean())
    elif strategy == "median":
        numeric_col = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(numeric_col.median())
    elif strategy == "mode":
        mode_val = df[column].mode()
        if len(mode_val) > 0:
            df[column] = df[column].fillna(mode_val.iloc[0])
    elif strategy == "ffill":
        df[column] = df[column].ffill()
    elif strategy == "bfill":
        df[column] = df[column].bfill()
    elif strategy is None:
        raise ValueError("fill_missing requires either 'value' or 'strategy' parameter")
    else:
        raise ValueError(f"Unknown fill strategy: {strategy}")

    return df


def _cast_column(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Cast a column to a specified type, coercing invalid values.

    Params:
        column (required): Column name to cast.
        dtype (required): Target type - 'int', 'float', 'str', 'datetime'.
        errors (optional): 'coerce' (default), 'ignore', or 'raise'.
    """
    column = params.get("column")
    dtype = params.get("dtype")

    if column is None:
        raise ValueError("cast_column requires 'column' parameter")
    if dtype is None:
        raise ValueError("cast_column requires 'dtype' parameter")

    _require_column(df, column)

    errors = params.get("errors", "coerce")

    if dtype == "int":
        df[column] = pd.to_numeric(df[column], errors=errors)
        if errors == "coerce":
            # Convert to Int64 (nullable integer) to preserve NaN
            df[column] = df[column].astype("Int64")
    elif dtype == "float":
        df[column] = pd.to_numeric(df[column], errors=errors)
    elif dtype == "str":
        df[column] = df[column].astype(str)
    elif dtype == "datetime":
        df[column] = pd.to_datetime(df[column], errors=errors)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    return df


def _rename_column(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Rename a column.

    Params:
        old_name (required): Current column name.
        new_name (required): New column name.
    """
    old_name = params.get("old_name")
    new_name = params.get("new_name")

    if old_name is None:
        raise ValueError("rename_column requires 'old_name' parameter")
    if new_name is None:
        raise ValueError("rename_column requires 'new_name' parameter")

    _require_column(df, old_name)

    df.rename(columns={old_name: new_name}, inplace=True)
    return df


def _strip_whitespace(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from string values in a column.

    Params:
        column (required): Column name to strip.
    """
    column = params.get("column")

    if column is None:
        raise ValueError("strip_whitespace requires 'column' parameter")

    _require_column(df, column)

    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    return df


def _drop_column(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Drop a column from the DataFrame.

    Params:
        column (required): Column name to drop.
    """
    column = params.get("column")

    if column is None:
        raise ValueError("drop_column requires 'column' parameter")

    _require_column(df, column)

    df.drop(columns=[column], inplace=True)
    return df
