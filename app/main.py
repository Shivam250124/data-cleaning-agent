"""FastAPI application for the Data Cleaning Agent OpenEnv."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.exceptions import ColumnNotFoundError, DatasetLoadError, EpisodeDoneError
from app.models import (
    ActionRequest,
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
