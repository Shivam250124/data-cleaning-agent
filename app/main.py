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


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for deployment verification."""
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")
    return HealthResponse(status="healthy")


@app.post("/reset", response_model=ObservationResponse)
async def reset(request: ResetRequest) -> ObservationResponse:
    """
    Reset the environment for a new episode.

    Args:
        request: Contains the difficulty level.

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
    """Trigger baseline inference and return scores for all tasks."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key set. Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable.",
        )

    try:
        from openai import OpenAI

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.environ.get("BASELINE_MODEL", "llama-3.1-8b-instant")

        client = OpenAI(api_key=api_key, base_url=base_url)
        env = get_env()
        scores = {}

        for diff_str in request.difficulties:
            try:
                difficulty = DifficultyLevel(diff_str)
            except ValueError:
                continue

            obs = env.reset(difficulty)
            max_steps = obs.info.max_steps

            for _ in range(max_steps):
                if obs.done:
                    break

                state_preview = obs.state[:10]
                prompt = f"""You are a data cleaning agent. Clean this dataset by choosing one action.

Current data (first 10 rows): {state_preview}
Step: {obs.info.step}/{max_steps}

Available actions: drop_duplicates, fill_missing, cast_column, rename_column, strip_whitespace, drop_column

Respond with JSON only: {{"action_type": "...", "params": {{...}}}}"""

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=200,
                    )
                    import json, re
                    text = response.choices[0].message.content.strip()
                    if "```" in text:
                        text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()
                    data = json.loads(text)
                    action_req = ActionRequest(
                        action_type=ActionType(data["action_type"]),
                        params=data.get("params", {}),
                    )
                    obs = env.step(action_req.action_type, action_req.params)
                except Exception:
                    obs = env.step(ActionType.drop_duplicates, {})

            scores[diff_str] = env.get_current_score()

        return {
            "scores": scores,
            "all_in_range": all(0.0 <= s <= 1.0 for s in scores.values()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")


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
