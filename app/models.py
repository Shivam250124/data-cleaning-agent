from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ActionType(str, Enum):
    drop_duplicates = "drop_duplicates"
    fill_missing = "fill_missing"
    cast_column = "cast_column"
    rename_column = "rename_column"
    strip_whitespace = "strip_whitespace"
    drop_column = "drop_column"


class ResetRequest(BaseModel):
    difficulty: DifficultyLevel = DifficultyLevel.easy  # Default to easy


class ActionRequest(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = {}


class InfoPayload(BaseModel):
    step: int
    max_steps: int
    difficulty: DifficultyLevel
    final_score: Optional[float] = None


class ObservationResponse(BaseModel):
    state: List[Dict[str, Any]]
    reward: float
    done: bool
    info: InfoPayload


class StateResponse(BaseModel):
    state: List[Dict[str, Any]]
    info: InfoPayload


class HealthResponse(BaseModel):
    status: str
