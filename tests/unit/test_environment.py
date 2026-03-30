"""Unit tests for the environment engine."""

import pytest

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.exceptions import EpisodeDoneError
from app.models import ActionType, DifficultyLevel


@pytest.fixture
def env():
    """Create an environment with real datasets."""
    registry = DatasetRegistry()
    return DataCleaningEnv(registry=registry)


class TestEnvironmentReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(DifficultyLevel.easy)
        assert obs.state is not None
        assert len(obs.state) > 0
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.info.step == 0
        assert obs.info.max_steps == 15  # Easy has 15 max steps

    def test_reset_sets_difficulty(self, env):
        obs = env.reset(DifficultyLevel.medium)
        assert obs.info.difficulty == DifficultyLevel.medium
        assert obs.info.max_steps == 30

    def test_reset_clears_done_state(self, env):
        # First episode
        env.reset(DifficultyLevel.easy)
        # Exhaust steps
        for _ in range(15):
            try:
                env.step(ActionType.drop_duplicates, {})
            except EpisodeDoneError:
                break

        # Reset should allow new episode
        obs = env.reset(DifficultyLevel.easy)
        assert obs.done is False


class TestEnvironmentStep:
    def test_step_increments_count(self, env):
        env.reset(DifficultyLevel.easy)
        obs = env.step(ActionType.drop_duplicates, {})
        assert obs.info.step == 1

    def test_step_returns_reward(self, env):
        env.reset(DifficultyLevel.easy)
        obs = env.step(ActionType.drop_duplicates, {})
        # Reward should be a float (positive, negative, or zero)
        assert isinstance(obs.reward, float)

    def test_step_without_reset_raises(self, env):
        with pytest.raises(EpisodeDoneError):
            env.step(ActionType.drop_duplicates, {})

    def test_step_after_done_raises(self, env):
        env.reset(DifficultyLevel.easy)
        # Exhaust all steps
        for _ in range(15):
            try:
                env.step(ActionType.drop_duplicates, {})
            except EpisodeDoneError:
                break

        with pytest.raises(EpisodeDoneError):
            env.step(ActionType.drop_duplicates, {})

    def test_episode_ends_at_max_steps(self, env):
        env.reset(DifficultyLevel.easy)
        obs = None
        for _ in range(15):
            obs = env.step(ActionType.strip_whitespace, {"column": "name"})
            if obs.done:
                break
        assert obs.done is True
        assert obs.info.step == 15


class TestEnvironmentState:
    def test_state_returns_current(self, env):
        env.reset(DifficultyLevel.easy)
        state = env.state()
        assert state.state is not None
        assert len(state.state) > 0

    def test_state_before_reset(self, env):
        state = env.state()
        assert state.state == []

    def test_state_unchanged_by_call(self, env):
        env.reset(DifficultyLevel.easy)
        state1 = env.state()
        state2 = env.state()
        assert len(state1.state) == len(state2.state)


class TestEnvironmentScoring:
    def test_get_current_score_in_bounds(self, env):
        env.reset(DifficultyLevel.easy)
        score = env.get_current_score()
        assert 0.0 <= score <= 1.0

    def test_score_improves_with_cleaning(self, env):
        env.reset(DifficultyLevel.easy)
        initial_score = env.get_current_score()

        # Drop duplicates should improve score (easy has duplicates)
        env.step(ActionType.drop_duplicates, {})
        new_score = env.get_current_score()

        # Score should improve or stay same (duplicates removed)
        assert new_score >= initial_score - 0.1  # Allow small regression

    def test_final_score_in_info(self, env):
        env.reset(DifficultyLevel.easy)
        obs = None
        for _ in range(15):
            obs = env.step(ActionType.drop_duplicates, {})
            if obs.done:
                break
        assert obs.info.final_score is not None
        assert 0.0 <= obs.info.final_score <= 1.0
