"""FastAPI application for the Data Cleaning Agent OpenEnv."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import DIFFICULTY_CONFIG
from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.exceptions import ColumnNotFoundError, DatasetLoadError, EpisodeDoneError
from app.models import (
    ActionRequest,
    ActionType,
    DifficultyLevel,
    HealthResponse,
    ObservationResponse,
    ResetRequest,
    StateResponse,
)


# Global environment instance
_env: Optional[DataCleaningEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment on startup."""
    global _env
    try:
        registry = DatasetRegistry()
        _env = DataCleaningEnv(registry=registry)
    except DatasetLoadError as e:
        # Log error but allow app to start (health check will fail)
        print(f"Warning: Failed to load datasets: {e}")
        _env = None
    yield
    # Cleanup on shutdown
    _env = None


app = FastAPI(
    title="Data Cleaning Agent OpenEnv",
    description="OpenEnv environment for evaluating AI agents on data cleaning tasks.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_env() -> DataCleaningEnv:
    """Get the environment instance, raising error if not initialized."""
    if _env is None:
        raise HTTPException(
            status_code=503,
            detail="Environment not initialized. Check that datasets are loaded.",
        )
    return _env


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Data Cleaning Agent OpenEnv",
        "version": "1.0.0",
        "description": "OpenEnv environment for evaluating AI agents on data cleaning tasks",
        "endpoints": {
            "health": "GET /health",
            "tasks": "GET /tasks",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "grader": "GET /grader",
            "baseline": "POST /baseline",
            "docs": "GET /docs",
        },
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for deployment verification."""
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")
    return HealthResponse(status="healthy")


@app.post("/reset", response_model=ObservationResponse)
async def reset(request: ResetRequest = ResetRequest()) -> ObservationResponse:
    """
    Reset the environment for a new episode.

    Args:
        request: Contains the difficulty level (defaults to 'easy' if not provided).

    Returns:
        Initial observation with state, reward, done flag, and info.
    """
    env = get_env()
    try:
        return env.reset(request.difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=ObservationResponse)
async def step(request: ActionRequest) -> ObservationResponse:
    """
    Execute an action in the environment.

    Args:
        request: Contains action_type and params.

    Returns:
        Observation with new state, reward, done flag, and info.
    """
    env = get_env()
    try:
        return env.step(request.action_type, request.params)
    except EpisodeDoneError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """
    Get the current state without taking an action.

    Returns:
        Current state and info.
    """
    env = get_env()
    return env.state()


# ---------------------------------------------------------------------------
# /tasks endpoint
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks():
    """Return list of tasks and the action schema."""
    tasks = []
    for difficulty, cfg in DIFFICULTY_CONFIG.items():
        tasks.append({
            "id": difficulty.value,
            "name": f"{difficulty.value.capitalize()} Data Cleaning",
            "difficulty": difficulty.value,
            "max_steps": cfg["max_steps"],
            "dataset_rows": cfg["rows"],
        })

    action_schema = {
        "action_type": {
            "type": "string",
            "enum": [a.value for a in ActionType],
            "description": "The cleaning action to perform",
        },
        "params": {
            "type": "object",
            "description": "Action-specific parameters",
            "examples": {
                "drop_duplicates": {},
                "fill_missing": {"column": "age", "strategy": "mean"},
                "cast_column": {"column": "age", "dtype": "int"},
                "rename_column": {"old_name": "old", "new_name": "new"},
                "strip_whitespace": {"column": "name"},
                "drop_column": {"column": "unwanted_col"},
            },
        },
    }

    return {"tasks": tasks, "action_schema": action_schema}


# ---------------------------------------------------------------------------
# /grader endpoint
# ---------------------------------------------------------------------------

@app.get("/grader")
async def get_grader_score():
    """Return the current grader score for the active episode."""
    env = get_env()
    score = env.get_current_score()
    state = env.state()
    return {
        "score": score,
        "step": state.info.step,
        "max_steps": state.info.max_steps,
        "difficulty": state.info.difficulty,
        "done": state.info.final_score is not None,
    }


# ---------------------------------------------------------------------------
# /baseline endpoint
# ---------------------------------------------------------------------------

class BaselineRequest(BaseModel):
    difficulties: list[str] = ["easy", "medium", "hard"]


@app.post("/baseline")
async def run_baseline(request: BaselineRequest = BaselineRequest()):
    """Return baseline scores for all tasks (quick version using grader only)."""
    env = get_env()
    scores = {}

    for diff_str in request.difficulties:
        try:
            difficulty = DifficultyLevel(diff_str)
        except ValueError:
            continue

        # Reset and apply best known cleaning sequence deterministically
        obs = env.reset(difficulty)

        # Apply a fixed sequence of cleaning actions (deterministic baseline)
        actions = [
            (ActionType.drop_duplicates, {}),
            (ActionType.strip_whitespace, {"column": list(obs.state[0].keys())[1]}),
            (ActionType.fill_missing, {"column": list(obs.state[0].keys())[1], "strategy": "mode"}),
        ]

        for action_type, params in actions:
            if obs.done:
                break
            try:
                obs = env.step(action_type, params)
            except Exception:
                pass

        scores[diff_str] = env.get_current_score()

    return {
        "scores": scores,
        "all_in_range": all(0.0 <= s <= 1.0 for s in scores.values()),
        "note": "Deterministic baseline. For LLM baseline run: python scripts/baseline.py",
    }


# Exception handlers for better error responses
@app.exception_handler(EpisodeDoneError)
async def episode_done_handler(request, exc: EpisodeDoneError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(ColumnNotFoundError)
async def column_not_found_handler(request, exc: ColumnNotFoundError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )
