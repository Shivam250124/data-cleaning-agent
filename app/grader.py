"""Grader / reward function for the Data Cleaning Agent environment."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_score(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """
    Compute a similarity score between the current and target DataFrames.

    The score is a weighted combination of:
    - Row count accuracy (10%): Penalizes having wrong number of rows
    - Column match accuracy (10%): Penalizes missing/extra columns
    - Cell-level accuracy (80%): Fraction of matching cells

    Args:
        current: The agent's current DataFrame state.
        target: The clean/target DataFrame.

    Returns:
        A float score strictly in [0.0, 1.0].
    """
    # Handle edge cases
    if current.empty and target.empty:
        return 1.0
    if current.empty or target.empty:
        return 0.0

    # Component 1: Row count accuracy (10%)
    current_rows = len(current)
    target_rows = len(target)
    if target_rows == 0:
        row_score = 1.0 if current_rows == 0 else 0.0
    else:
        row_diff = abs(current_rows - target_rows)
        row_score = max(0.0, 1.0 - (row_diff / target_rows))

    # Component 2: Column match accuracy (10%)
    current_cols = set(current.columns)
    target_cols = set(target.columns)
    if len(target_cols) == 0:
        col_score = 1.0 if len(current_cols) == 0 else 0.0
    else:
        matching_cols = current_cols & target_cols
        extra_cols = current_cols - target_cols
        missing_cols = target_cols - current_cols
        # Penalize both missing and extra columns
        col_score = len(matching_cols) / (len(target_cols) + len(extra_cols))

    # Component 3: Cell-level accuracy (80%)
    # Only compare shared columns and up to min row count
    shared_cols = list(current_cols & target_cols)
    if not shared_cols:
        cell_score = 0.0
    else:
        min_rows = min(current_rows, target_rows)
        if min_rows == 0:
            cell_score = 0.0
        else:
            # Subset both DataFrames to shared columns and comparable rows
            curr_subset = current[shared_cols].head(min_rows).reset_index(drop=True)
            tgt_subset = target[shared_cols].head(min_rows).reset_index(drop=True)

            total_cells = min_rows * len(shared_cols)
            matching_cells = 0

            for col in shared_cols:
                for i in range(min_rows):
                    curr_val = curr_subset.at[i, col]
                    tgt_val = tgt_subset.at[i, col]

                    if _values_match(curr_val, tgt_val):
                        matching_cells += 1

            cell_score = matching_cells / total_cells

    # Weighted combination
    score = 0.10 * row_score + 0.10 * col_score + 0.80 * cell_score

    # Guarantee strict [0.0, 1.0] bounds
    return float(max(0.0, min(1.0, score)))


def _values_match(val1, val2) -> bool:
    """
    Check if two values match, handling NaN and type differences.
    """
    # Both NaN
    if pd.isna(val1) and pd.isna(val2):
        return True

    # One NaN, one not
    if pd.isna(val1) or pd.isna(val2):
        return False

    # Convert to comparable types
    try:
        # Try numeric comparison first
        num1 = float(val1) if not isinstance(val1, (int, float, np.number)) else val1
        num2 = float(val2) if not isinstance(val2, (int, float, np.number)) else val2
        if isinstance(num1, (int, float, np.number)) and isinstance(num2, (int, float, np.number)):
            return np.isclose(num1, num2, rtol=1e-5, atol=1e-8)
    except (ValueError, TypeError):
        pass

    # String comparison (case-sensitive, whitespace-sensitive)
    str1 = str(val1).strip()
    str2 = str(val2).strip()
    return str1 == str2


def compute_reward(
    prev_score: float,
    curr_score: float,
    done: bool,
    steps_remaining: int,
    max_steps: int,
) -> float:
    """
    Compute the reward for a single step.

    Uses dense shaping: reward is based on score improvement plus a bonus
    for completion.

    Args:
        prev_score: Score before the action.
        curr_score: Score after the action.
        done: Whether the episode is complete.
        steps_remaining: Number of steps remaining after this action.
        max_steps: Maximum steps for this difficulty.

    Returns:
        A float reward (can be negative for regression).
    """
    # Base reward: improvement in score (scaled)
    improvement = curr_score - prev_score
    reward = improvement

    # Completion bonus if done with high score
    if done and curr_score >= 0.95:
        # Bonus for finishing early with good score
        efficiency_bonus = 0.1 * (steps_remaining / max_steps)
        reward += efficiency_bonus

    # Guarantee reward is reasonable (not infinite)
    return float(max(-1.0, min(1.0, reward)))
