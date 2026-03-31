"""Property-based tests for the grader module using Hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume

from app.grader import compute_score, compute_reward


# Strategy for generating simple DataFrames
@st.composite
def dataframes(draw, min_rows=0, max_rows=20, min_cols=1, max_cols=5):
    """Generate random DataFrames for testing."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    columns = [f"col_{i}" for i in range(n_cols)]
    data = {}
    
    for col in columns:
        col_type = draw(st.sampled_from(["int", "float", "str"]))
        if col_type == "int":
            data[col] = draw(st.lists(
                st.integers(min_value=-100, max_value=100),
                min_size=n_rows, max_size=n_rows
            ))
        elif col_type == "float":
            data[col] = draw(st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=n_rows, max_size=n_rows
            ))
        else:
            data[col] = draw(st.lists(
                st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L', 'N'))),
                min_size=n_rows, max_size=n_rows
            ))
    
    return pd.DataFrame(data)


class TestComputeScoreProperties:
    """Property-based tests for compute_score."""

    @given(dataframes(min_rows=1, max_rows=10))
    @settings(max_examples=50)
    def test_score_always_in_bounds(self, df):
        """Score must always be in [0.0, 1.0]."""
        score = compute_score(df, df)
        assert 0.0 <= score <= 1.0

    @given(dataframes(min_rows=1, max_rows=10), dataframes(min_rows=1, max_rows=10))
    @settings(max_examples=50)
    def test_score_between_different_dfs_in_bounds(self, df1, df2):
        """Score between any two DataFrames must be in [0.0, 1.0]."""
        score = compute_score(df1, df2)
        assert 0.0 <= score <= 1.0

    @given(dataframes(min_rows=1, max_rows=10))
    @settings(max_examples=30)
    def test_identical_dataframes_high_score(self, df):
        """Identical DataFrames should have score 1.0."""
        score = compute_score(df.copy(), df.copy())
        assert score == 1.0

    def test_empty_dataframes_score_one(self):
        """Two empty DataFrames should score 1.0."""
        df = pd.DataFrame()
        assert compute_score(df, df) == 1.0

    def test_empty_vs_nonempty_score_zero(self):
        """Empty vs non-empty should score 0.0."""
        empty = pd.DataFrame()
        nonempty = pd.DataFrame({"a": [1, 2, 3]})
        assert compute_score(empty, nonempty) == 0.0
        assert compute_score(nonempty, empty) == 0.0


class TestComputeRewardProperties:
    """Property-based tests for compute_reward."""

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.booleans(),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_reward_always_bounded(self, prev_score, curr_score, done, steps_remaining, max_steps):
        """Reward must always be in [-1.0, 1.0]."""
        reward = compute_reward(prev_score, curr_score, done, steps_remaining, max_steps)
        assert -1.0 <= reward <= 1.0

    @given(
        st.floats(min_value=0.0, max_value=0.9),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_improvement_gives_positive_reward(self, prev_score, steps_remaining, max_steps):
        """Improving the score should give positive reward."""
        curr_score = min(1.0, prev_score + 0.1)
        assume(curr_score > prev_score)
        reward = compute_reward(prev_score, curr_score, False, steps_remaining, max_steps)
        assert reward > 0

    @given(
        st.floats(min_value=0.1, max_value=1.0),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_regression_gives_negative_reward(self, prev_score, steps_remaining, max_steps):
        """Making the score worse should give negative reward."""
        curr_score = max(0.0, prev_score - 0.1)
        assume(curr_score < prev_score)
        reward = compute_reward(prev_score, curr_score, False, steps_remaining, max_steps)
        assert reward < 0

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_no_change_gives_zero_reward(self, score, steps_remaining, max_steps):
        """No change in score should give zero reward (when not done)."""
        reward = compute_reward(score, score, False, steps_remaining, max_steps)
        assert reward == 0.0


class TestScoreRewardConsistency:
    """Tests for consistency between score and reward."""

    @given(
        st.floats(min_value=0.0, max_value=0.5),
        st.floats(min_value=0.5, max_value=1.0),
    )
    @settings(max_examples=30)
    def test_bigger_improvement_bigger_reward(self, low_score, high_score):
        """Larger improvements should yield larger rewards."""
        assume(high_score > low_score + 0.1)
        
        mid_score = (low_score + high_score) / 2
        
        small_improvement = compute_reward(low_score, mid_score, False, 5, 10)
        big_improvement = compute_reward(low_score, high_score, False, 5, 10)
        
        assert big_improvement > small_improvement
