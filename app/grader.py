"""Grader / reward function for the Data Cleaning Agent environment."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_score(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """
    Compute a similarity score between the current and target DataFrames.

    Strict scoring that penalizes:
    - Wrong row count (duplicates not removed)
    - Missing values not filled
    - Whitespace not stripped
    - Wrong data types
    - Cell value mismatches

    Returns a float in [0.0, 1.0].
    """
    if current.empty and target.empty:
        return 1.0
    if current.empty or target.empty:
        return 0.0

    current_rows = len(current)
    target_rows = len(target)
    target_cols = set(target.columns)
    current_cols = set(current.columns)

    # Sub-score 1: Row count (20%) — penalize duplicates heavily
    if target_rows == 0:
        row_score = 1.0 if current_rows == 0 else 0.0
    else:
        if current_rows > target_rows:
            # Extra rows = duplicates not removed
            row_score = max(0.0, 1.0 - (current_rows - target_rows) / target_rows)
        else:
            row_score = max(0.0, 1.0 - abs(current_rows - target_rows) / target_rows)

    # Sub-score 2: Missing values (20%) — penalize any nulls in current
    shared_cols = list(current_cols & target_cols)
    if shared_cols:
        total_nulls_in_current = current[shared_cols].isnull().sum().sum()
        total_nulls_in_target = target[shared_cols].isnull().sum().sum()
        total_cells = len(current) * len(shared_cols)
        if total_cells > 0:
            null_score = max(0.0, 1.0 - (total_nulls_in_current - total_nulls_in_target) / total_cells)
        else:
            null_score = 1.0
    else:
        null_score = 0.0

    # Sub-score 3: Cell-level exact match (60%) — strict, no whitespace tolerance
    if not shared_cols:
        cell_score = 0.0
    else:
        min_rows = min(current_rows, target_rows)
        if min_rows == 0:
            cell_score = 0.0
        else:
            curr_subset = current[shared_cols].head(min_rows).reset_index(drop=True)
            tgt_subset = target[shared_cols].head(min_rows).reset_index(drop=True)

            total_cells = min_rows * len(shared_cols)
            matching_cells = 0

            for col in shared_cols:
                for i in range(min_rows):
                    curr_val = curr_subset.at[i, col]
                    tgt_val = tgt_subset.at[i, col]
                    if _values_match_strict(curr_val, tgt_val):
                        matching_cells += 1

            cell_score = matching_cells / total_cells

    score = 0.20 * row_score + 0.20 * null_score + 0.60 * cell_score
    return float(max(0.0, min(1.0, score)))


def _values_match_strict(val1, val2) -> bool:
    """Strict value matching — whitespace matters, NaN != NaN."""
    # Both NaN — in dirty data NaN is bad, so don't reward it
    if pd.isna(val1) and pd.isna(val2):
        return True
    if pd.isna(val1) or pd.isna(val2):
        return False

    # Numeric comparison
    try:
        n1 = float(val1)
        n2 = float(val2)
        return np.isclose(n1, n2, rtol=1e-5, atol=1e-8)
    except (ValueError, TypeError):
        pass

    # Strict string comparison — whitespace NOT stripped (agent must strip it)
    return str(val1) == str(val2)


def _values_match(val1, val2) -> bool:
    """Lenient value matching (used in tests)."""
    if pd.isna(val1) and pd.isna(val2):
        return True
    if pd.isna(val1) or pd.isna(val2):
        return False
    try:
        n1 = float(val1)
        n2 = float(val2)
        return np.isclose(n1, n2, rtol=1e-5, atol=1e-8)
    except (ValueError, TypeError):
        pass
    return str(val1).strip() == str(val2).strip()


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
    for completion. Can be negative for regression.

    Args:
        prev_score: Score before the action.
        curr_score: Score after the action.
        done: Whether the episode is complete.
        steps_remaining: Number of steps remaining after this action.
        max_steps: Maximum steps for this difficulty.

    Returns:
        A float reward in [-1.0, 1.0]. Negative for regression.
    """
    # Base reward: improvement in score (can be negative)
    reward = curr_score - prev_score

    # Completion bonus if done with high score
    if done and curr_score >= 0.95:
        # Bonus for finishing early with good score
        efficiency_bonus = 0.1 * (steps_remaining / max(max_steps, 1))
        reward += efficiency_bonus

    # Guarantee reward is bounded
    return float(max(-1.0, min(1.0, reward)))
