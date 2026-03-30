"""Core environment engine for the Data Cleaning Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.actions import execute_action
from app.config import DIFFICULTY_CONFIG
from app.datasets import DatasetRegistry
from app.exceptions import EpisodeDoneError
from app.grader import compute_score, compute_reward
from app.models import ActionType, DifficultyLevel, InfoPayload, ObservationResponse, StateResponse


class DataCleaningEnv:
    """
    OpenEnv-compatible environment for data cleaning tasks.

    The agent receives a dirty DataFrame and must clean it to match
    a target clean DataFrame using a series of actions.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        """
        Initialize the environment.

        Args:
            registry: Optional DatasetRegistry instance. If None, creates one.
        """
        self._registry = registry or DatasetRegistry()
        self._difficulty: DifficultyLevel | None = None
        self._current_df: pd.DataFrame | None = None
        self._target_df: pd.DataFrame | None = None
        self._step_count: int = 0
        self._max_steps: int = 0
        self._done: bool = True
        self._prev_score: float = 0.0

    def reset(self, difficulty: DifficultyLevel) -> ObservationResponse:
        """
        Reset the environment for a new episode.

        Args:
            difficulty: The difficulty level for the episode.

        Returns:
            ObservationResponse with initial state.
        """
        self._difficulty = difficulty
        config = DIFFICULTY_CONFIG[difficulty]

        self._current_df = self._registry.get_dirty(difficulty)
        self._target_df = self._registry.get_clean(difficulty)
        self._step_count = 0
        self._max_steps = config["max_steps"]
        self._done = False

        # Compute initial score
        self._prev_score = compute_score(self._current_df, self._target_df)

        return ObservationResponse(
            state=self._df_to_state(),
            reward=0.0,
            done=False,
            info=self._make_info(),
        )

    def step(self, action_type: ActionType, params: Dict[str, Any]) -> ObservationResponse:
        """
        Execute an action and return the result.

        Args:
            action_type: The type of action to execute.
            params: Action-specific parameters.

        Returns:
            ObservationResponse with new state, reward, and done flag.

        Raises:
            EpisodeDoneError: If the episode has already ended.
        """
        if self._done:
            raise EpisodeDoneError("Episode has ended. Call reset() to start a new episode.")

        if self._current_df is None:
            raise EpisodeDoneError("Environment not initialized. Call reset() first.")

        # Execute the action
        self._current_df = execute_action(self._current_df, action_type, params)
        self._step_count += 1

        # Compute new score
        curr_score = compute_score(self._current_df, self._target_df)

        # Compute reward
        steps_remaining = self._max_steps - self._step_count
        reward = compute_reward(
            prev_score=self._prev_score,
            curr_score=curr_score,
            done=self._step_count >= self._max_steps or curr_score >= 1.0,
            steps_remaining=steps_remaining,
            max_steps=self._max_steps,
        )

        # Update state
        self._prev_score = curr_score

        # Check if done
        if self._step_count >= self._max_steps or curr_score >= 1.0:
            self._done = True

        info = self._make_info()
        if self._done:
            info.final_score = curr_score

        return ObservationResponse(
            state=self._df_to_state(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> StateResponse:
        """
        Get the current state without taking an action.

        Returns:
            StateResponse with current state.
        """
        if self._current_df is None:
            # Return empty state if not initialized
            return StateResponse(
                state=[],
                info=InfoPayload(
                    step=0,
                    max_steps=0,
                    difficulty=DifficultyLevel.easy,
                    final_score=None,
                ),
            )

        return StateResponse(
            state=self._df_to_state(),
            info=self._make_info(),
        )

    def get_current_score(self) -> float:
        """Get the current score without modifying state."""
        if self._current_df is None or self._target_df is None:
            return 0.0
        return compute_score(self._current_df, self._target_df)

    def _df_to_state(self) -> List[Dict[str, Any]]:
        """Convert the current DataFrame to a list of dicts for JSON serialization."""
        if self._current_df is None:
            return []

        # Convert to records, replacing NaN with None for JSON compliance
        df_copy = self._current_df.copy()
        # Replace NaN/NaT with None
        df_copy = df_copy.where(df_copy.notna(), None)
        return df_copy.to_dict(orient="records")

    def _make_info(self) -> InfoPayload:
        """Create an InfoPayload for the current state."""
        return InfoPayload(
            step=self._step_count,
            max_steps=self._max_steps,
            difficulty=self._difficulty or DifficultyLevel.easy,
            final_score=None,
        )
