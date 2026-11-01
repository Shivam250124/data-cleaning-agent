"""Unit tests for the grader module."""

import numpy as np
import pandas as pd
import pytest

from app.grader import compute_score, compute_reward, _values_match


class TestValuesMatch:
    def test_both_nan(self):
        assert _values_match(np.nan, np.nan) == True
        assert _values_match(None, None) == True

    def test_one_nan(self):
        assert _values_match(np.nan, 1) == False
        assert _values_match(1, np.nan) == False

    def test_numeric_match(self):
        assert _values_match(1, 1) == True
        assert _values_match(1.0, 1) == True
        assert _values_match(1.5, 1.5) == True

    def test_numeric_close(self):
        assert _values_match(1.0000001, 1.0) == True
        assert _values_match(1.1, 1.0) == False

    def test_string_match(self):
        assert _values_match("hello", "hello") == True
        assert _values_match("hello", "world") == False

    def test_string_whitespace_stripped(self):
        assert _values_match("hello  ", "  hello") == True


class TestComputeScore:
    def test_identical_dataframes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        score = compute_score(df.copy(), df.copy())
        assert score == 1.0

    def test_empty_dataframes(self):
        df = pd.DataFrame()
        score = compute_score(df, df)
        assert score == 1.0

    def test_completely_different(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        score = compute_score(df1, df2)
        assert score < 0.3  # Low score due to column mismatch

    def test_partial_match(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6]})  # One cell different
        score = compute_score(df1, df2)
        assert 0.8 < score < 1.0

    def test_row_count_mismatch(self):
        df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        score = compute_score(df1, df2)
        assert score < 1.0  # Penalized for extra rows

    def test_score_in_bounds(self):
        # Test many edge cases
        cases = [
            (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]})),
            (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})),
            (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]})),
            (pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]})),
            (pd.DataFrame(), pd.DataFrame({"a": [1]})),
        ]
        for df1, df2 in cases:
            score = compute_score(df1, df2)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {df1}, {df2}"


class TestComputeReward:
    def test_improvement_positive_reward(self):
        reward = compute_reward(
            prev_score=0.5,
            curr_score=0.7,
            done=False,
            steps_remaining=5,
            max_steps=10,
        )
        assert reward > 0

    def test_regression_negative_reward(self):
        reward = compute_reward(
            prev_score=0.7,
            curr_score=0.5,
            done=False,
            steps_remaining=5,
            max_steps=10,
        )
        assert reward < 0

    def test_no_change_zero_reward(self):
        reward = compute_reward(
            prev_score=0.5,
            curr_score=0.5,
            done=False,
            steps_remaining=5,
            max_steps=10,
        )
        assert reward == 0.0

    def test_completion_bonus(self):
        # High score completion should get bonus
        reward_done = compute_reward(
            prev_score=0.94,
            curr_score=0.96,
            done=True,
            steps_remaining=5,
            max_steps=10,
        )
        reward_not_done = compute_reward(
            prev_score=0.94,
            curr_score=0.96,
            done=False,
            steps_remaining=5,
            max_steps=10,
        )
        assert reward_done > reward_not_done

    def test_reward_bounded(self):
        # Even extreme cases should be bounded
        reward = compute_reward(
            prev_score=0.0,
            curr_score=1.0,
            done=True,
            steps_remaining=10,
            max_steps=10,
        )
        assert -1.0 <= reward <= 1.0
