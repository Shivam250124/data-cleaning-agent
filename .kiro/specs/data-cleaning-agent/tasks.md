# Implementation Plan: Data Cleaning Agent OpenEnv

## Overview

Incremental build of the FastAPI environment server, synthetic datasets, grader, action handlers, and Gemini baseline script. Each task builds on the previous, ending with a fully wired and deployable system.

## Tasks

- [x] 1. Project scaffold and dependency setup
  - Create directory structure: `app/`, `data/`, `scripts/`, `tests/unit/`, `tests/property/`
  - Create `requirements.txt` with pinned versions: `fastapi`, `uvicorn`, `pandas`, `hypothesis`, `pytest`, `httpx`, `google-generativeai`
  - Create `app/__init__.py`, `app/models.py` with all Pydantic models and enums (`DifficultyLevel`, `ActionType`, `ResetRequest`, `ActionRequest`, `ObservationResponse`, `InfoPayload`, `StateResponse`, `HealthResponse`)
  - Create `app/config.py` with `DIFFICULTY_CONFIG` dict
  - _Requirements: 2.1, 2.2, 2.3, 9.5_

- [ ] 2. Synthetic dataset generation
  - [x] 2.1 Generate Easy dataset (Customer, 50 rows)
    - Write `scripts/generate_datasets.py` that creates `data/easy_dirty.csv` and `data/easy_clean.csv`
    - Dirty CSV must contain: missing values, duplicate rows, type errors (e.g., numeric stored as string), formatting inconsistencies (e.g., extra whitespace in names)
    - Clean CSV is the fully corrected ground truth
    - _Requirements: 3.1, 3.2_
  - [x] 2.2 Generate Medium dataset (Sales, 200 rows)
    - Add Medium generation to `scripts/generate_datasets.py`: `data/medium_dirty.csv` and `data/medium_clean.csv`
    - _Requirements: 3.1, 3.2_
  - [x] 2.3 Generate Hard dataset (HR, 500 rows)
    - Add Hard generation to `scripts/generate_datasets.py`: `data/hard_dirty.csv` and `data/hard_clean.csv`
    - _Requirements: 3.1, 3.2_
  - [ ]* 2.4 Write unit tests for dataset integrity
    - Test each dirty CSV contains nulls, duplicates, type errors, and whitespace issues
    - Test each clean CSV has no nulls, no duplicates, correct dtypes
    - _Requirements: 3.1, 3.2_

- [x] 3. DatasetRegistry and startup validation
  - Implement `app/datasets.py` with `DatasetRegistry` class that loads all 6 CSVs at startup using `DIFFICULTY_CONFIG`
  - Raise `DatasetLoadError` (custom exception in `app/exceptions.py`) with the missing file path if any CSV is absent
  - Expose `get_dirty(difficulty)` and `get_clean(difficulty)` returning deep copies of cached DataFrames
  - _Requirements: 3.3, 3.4_
  - [-]* 3.1 Write unit tests for DatasetRegistry
    - Test successful load returns correct row counts per difficulty
    - Test missing file raises `DatasetLoadError` with correct path in message
    - _Requirements: 3.3, 3.4_

- [ ] 4. ActionHandler implementation
  - [x] 4.1 Implement all 6 action functions in `app/actions.py`
    - `drop_duplicates(df)`, `fill_missing(df, column, strategy)`, `cast_column(df, column, dtype)`, `rename_column(df, old_name, new_name)`, `strip_whitespace(df, column)`, `drop_column(df, column)`
    - All column-referencing functions raise `ColumnNotFoundError` if column absent; return unchanged df
    - Implement `apply_action(df, action: ActionRequest) -> pd.DataFrame` dispatch function
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [ ]* 4.2 Write unit tests for ActionHandler
    - Happy path for each of the 6 actions with known inputs and expected outputs
    - `ColumnNotFoundError` raised for each column-referencing action when column is absent
    - `cast_column` with incompatible dtype leaves df unchanged and raises error
    - _Requirements: 5.1–5.7_
  - [ ]* 4.3 Write property test: drop_duplicates idempotent
    - **Property 4: drop_duplicates is idempotent**
    - **Validates: Requirements 5.1**
    - Generate DataFrames with random duplicate rows; assert `drop_duplicates(drop_duplicates(df)) == drop_duplicates(df)`
  - [ ]* 4.4 Write property test: action on missing column leaves DataFrame unchanged
    - **Property 5: Action on missing column leaves DataFrame unchanged**
    - **Validates: Requirements 5.7**
    - Generate DataFrames and column names not present in the DataFrame; assert df is unchanged and error is raised
  - [ ]* 4.5 Write property test: fill_missing removes all nulls in target column
    - **Property 9: fill_missing removes all nulls in target column**
    - **Validates: Requirements 5.2**
    - Generate DataFrames with nulls in a column and a valid fill value; assert zero nulls remain after action

- [x] 5. Grader implementation
  - Implement `app/grader.py` with `compute_reward(current_df, clean_df) -> float`
  - Compute four sub-scores: `duplicate_score`, `missing_score`, `type_score`, `value_score`
  - Return weighted average clamped to [0.0, 1.0]
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ]* 5.1 Write unit tests for Grader
    - Test `compute_reward(clean_df, clean_df)` returns 1.0 for each difficulty
    - Test `compute_reward(dirty_df, clean_df)` returns < 1.0 for each difficulty
    - Test known intermediate states return expected reward ranges
    - _Requirements: 4.3, 4.4_
  - [ ]* 5.2 Write property test: reward always in [0.0, 1.0]
    - **Property 1: Reward is always in [0.0, 1.0]**
    - **Validates: Requirements 4.1**
    - Generate random pairs of DataFrames; assert `0.0 <= compute_reward(df1, df2) <= 1.0`
  - [ ]* 5.3 Write property test: perfect state yields 1.0
    - **Property 2: Perfect state yields reward 1.0**
    - **Validates: Requirements 4.3**
    - Generate random DataFrames; assert `compute_reward(df, df) == 1.0`
  - [ ]* 5.4 Write property test: grader is deterministic
    - **Property 3: Grader is deterministic**
    - **Validates: Requirements 4.6**
    - Generate random DataFrame pairs; assert two calls return identical floats
  - [ ]* 5.5 Write property test: reward increases after correct cleaning action
    - **Property 10: Reward increases after a correct cleaning action**
    - **Validates: Requirements 4.5**
    - For a known dirty/clean pair, apply a known-correct action and assert reward after >= reward before

- [ ] 6. Checkpoint — core logic complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. EnvironmentManager implementation
  - Implement `app/environment.py` with `EnvironmentManager` class
  - `reset(difficulty)`: deep-copy dirty df into `working_df`, set `step_count=0`, `done=False`, return initial `ObservationResponse`
  - `step(action)`: validate not done (raise `EpisodeDoneError` if done), call `apply_action`, call `compute_reward`, increment `step_count`, check termination, return `ObservationResponse`
  - `get_state()`: return `StateResponse` with current `working_df` and info
  - _Requirements: 1.5, 1.6, 1.7, 2.4, 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ]* 7.1 Write unit tests for EnvironmentManager
    - Test reset returns initial observation with correct difficulty and step=0
    - Test step increments step count and returns reward in [0.0, 1.0]
    - Test step after done=True raises/returns 400 error
    - Test done=True is set when step_count reaches max_steps
    - Test final_score present in info when done=True
    - _Requirements: 1.5, 1.6, 1.7, 2.4, 6.5_
  - [ ]* 7.2 Write property test: step count monotonically increases and terminates
    - **Property 6: Step count monotonically increases and terminates**
    - **Validates: Requirements 2.4, 6.3, 6.4**
    - Simulate N steps for each difficulty; assert step count equals N and done=True at max_steps
  - [ ]* 7.3 Write property test: reset restores dirty dataset
    - **Property 8: Reset restores dirty dataset**
    - **Validates: Requirements 1.5**
    - Apply random action sequences, call reset, assert working_df equals original dirty_df
  - [ ]* 7.4 Write property test: observation schema completeness
    - **Property 7: Observation schema completeness**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    - Generate random valid actions, call step/reset, assert all required fields present and typed correctly

- [x] 8. FastAPI router and app wiring
  - Create `app/routers/env.py` with POST `/reset`, POST `/step`, GET `/state`, GET `/health` endpoints
  - Create `app/main.py` that instantiates `DatasetRegistry` and `EnvironmentManager` at startup, mounts the router, and runs on port 7860
  - Add startup event that raises `DatasetLoadError` if any CSV is missing
  - Wire `EnvironmentManager` as a FastAPI dependency or app-level singleton
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.7, 1.8, 9.1, 9.3_
  - [ ]* 8.1 Write unit tests for API endpoints
    - Test GET `/health` returns 200 with `{"status": "ok"}`
    - Test POST `/reset` with each difficulty returns valid ObservationResponse
    - Test POST `/step` with valid action returns valid ObservationResponse
    - Test POST `/step` after done=True returns 400
    - Test POST `/step` with invalid body returns 422
    - Test POST `/step` with unknown action_type returns 422
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.7, 1.8_

- [ ] 9. Checkpoint — API complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Baseline inference script
  - Implement `scripts/baseline.py`:
    - Read `GOOGLE_API_KEY` from `os.environ`; exit with code 1 and stderr message if absent
    - Initialize `google.generativeai` with the key; use `gemini-1.5-flash` model at `temperature=0`
    - Implement `run_episode(difficulty, env_url, model) -> float` that loops reset→step until done
    - Call `run_episode` for all 3 difficulties sequentially; print scores to stdout
  - _Requirements: 7.1, 7.2, 7.3, 7.5, 8.1, 8.2_
  - [ ]* 10.1 Write unit tests for baseline script
    - Test script exits with code 1 and prints to stderr when `GOOGLE_API_KEY` is unset (mock env)
    - Test `run_episode` calls `/reset` then `/step` in a loop until `done=True` (mock HTTP server)
    - Test model is called with `temperature=0` (mock `google.generativeai`)
    - _Requirements: 7.1, 7.5, 8.1_

- [x] 11. Docker and deployment configuration
  - Create `Dockerfile`: base `python:3.10-slim`, copy source, install `requirements.txt`, expose port 7860, `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]`
  - Create `.env.example` with `GOOGLE_API_KEY=your_key_here` (never commit a real key)
  - Verify `GOOGLE_API_KEY` is absent from all response models and log statements
  - _Requirements: 8.2, 9.1, 9.2, 9.4, 9.5_

- [ ] 12. Final checkpoint — full suite
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` minimum
- Unit tests use pytest with `httpx.AsyncClient` for endpoint testing
- The `GOOGLE_API_KEY` must never appear in any response body, header, or log — enforced by Requirement 8.2
