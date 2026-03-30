# Design Document: Data Cleaning Agent OpenEnv

## Overview

The Data Cleaning Agent OpenEnv is a FastAPI HTTP server that exposes a reinforcement-learning-style environment for evaluating AI agents on multi-step data cleaning tasks. It follows the classic `reset → step → step → ... → done` loop. Each episode operates on one of three synthetic dirty CSV datasets; the agent submits cleaning actions, and a deterministic grader scores progress against a clean ground-truth CSV, returning a shaped reward in [0.0, 1.0] after every step.

The baseline inference script (`scripts/baseline.py`) drives the Gemini 1.5 Flash model against all three difficulty levels and prints reproducible scores, serving as the reference evaluation for the OpenEnv hackathon submission.

---

## Architecture

```mermaid
graph TD
    subgraph Client
        A[Agent / Baseline Script]
    end

    subgraph FastAPI Server (port 7860)
        B[Router Layer]
        C[EnvironmentManager]
        D[ActionHandler]
        E[Grader]
        F[DatasetRegistry]
    end

    subgraph Assets
        G[data/easy_dirty.csv]
        H[data/easy_clean.csv]
        I[data/medium_dirty.csv]
        J[data/medium_clean.csv]
        K[data/hard_dirty.csv]
        L[data/hard_clean.csv]
    end

    subgraph Baseline
        M[scripts/baseline.py]
        N[Google Gemini API]
    end

    A -->|POST /reset, POST /step, GET /state, GET /health| B
    B --> C
    C --> D
    C --> E
    C --> F
    F -->|loads at startup| G
    F -->|loads at startup| H
    F -->|loads at startup| I
    F -->|loads at startup| J
    F -->|loads at startup| K
    F -->|loads at startup| L
    M -->|REST calls| B
    M -->|inference| N
```

**Request flow for `/step`:**
1. Router receives POST `/step` with an `ActionRequest` body.
2. `EnvironmentManager` validates episode state (not done).
3. `ActionHandler` applies the action to the in-memory working DataFrame.
4. `Grader` scores the updated DataFrame against the clean DataFrame.
5. `EnvironmentManager` increments step count, checks termination.
6. Router returns `ObservationResponse`.

---

## Components and Interfaces

### 1. Router Layer (`app/routers/env.py`)

Thin FastAPI router. Delegates all logic to `EnvironmentManager`. No business logic here.

```python
POST /reset   -> ResetRequest  -> ObservationResponse
POST /step    -> ActionRequest -> ObservationResponse
GET  /state   ->               -> StateResponse
GET  /health  ->               -> HealthResponse
```

### 2. EnvironmentManager (`app/environment.py`)

Central stateful component. Holds one active episode at a time (single-session server).

```python
class EnvironmentManager:
    def reset(self, difficulty: DifficultyLevel) -> ObservationResponse
    def step(self, action: ActionRequest) -> ObservationResponse
    def get_state(self) -> StateResponse
```

Internal state:
- `working_df: pd.DataFrame` — mutable copy of the dirty dataset for the current episode
- `clean_df: pd.DataFrame` — immutable reference to the clean dataset
- `step_count: int`
- `max_steps: int`
- `done: bool`
- `difficulty: DifficultyLevel`

### 3. ActionHandler (`app/actions.py`)

Pure functions — no side effects beyond transforming a DataFrame. Each action function takes a DataFrame and action parameters, returns a new DataFrame.

```python
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame
def fill_missing(df: pd.DataFrame, column: str, strategy: str | Any) -> pd.DataFrame
def cast_column(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame
def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame
def strip_whitespace(df: pd.DataFrame, column: str) -> pd.DataFrame
def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame
```

Dispatch function:
```python
def apply_action(df: pd.DataFrame, action: ActionRequest) -> pd.DataFrame
```

All functions raise `ColumnNotFoundError` if a referenced column is absent.

### 4. Grader (`app/grader.py`)

Deterministic scoring. Computes a weighted composite of sub-scores.

```python
def compute_reward(current_df: pd.DataFrame, clean_df: pd.DataFrame) -> float
```

Sub-scores (each in [0.0, 1.0]):
- `duplicate_score`: fraction of duplicate rows removed
- `missing_score`: fraction of missing values filled correctly
- `type_score`: fraction of columns with correct dtypes
- `value_score`: cell-level exact match fraction against clean_df

Final reward: weighted average of sub-scores, clamped to [0.0, 1.0].

### 5. DatasetRegistry (`app/datasets.py`)

Loads and caches all six CSV files at startup.

```python
class DatasetRegistry:
    def get_dirty(self, difficulty: DifficultyLevel) -> pd.DataFrame
    def get_clean(self, difficulty: DifficultyLevel) -> pd.DataFrame
```

Raises `DatasetLoadError` at startup if any file is missing.

### 6. Baseline Script (`scripts/baseline.py`)

Standalone script. Uses `google-generativeai` SDK.

```python
def run_episode(difficulty: str, env_url: str, model) -> float
def main() -> None
```

Flow per episode:
1. POST `/reset` with difficulty
2. Loop: send observation to Gemini, parse action JSON from response, POST `/step`
3. Break when `done=True`
4. Print final score

---

## Data Models

### Enums

```python
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
```

### Request Models

```python
class ResetRequest(BaseModel):
    difficulty: DifficultyLevel

class ActionRequest(BaseModel):
    action_type: ActionType
    params: dict[str, Any] = {}
```

### Response Models

```python
class ObservationResponse(BaseModel):
    state: list[dict[str, Any]]   # records-oriented JSON of current DataFrame
    reward: float                  # in [0.0, 1.0]
    done: bool
    info: InfoPayload

class InfoPayload(BaseModel):
    step: int
    max_steps: int
    difficulty: DifficultyLevel
    final_score: float | None = None  # present only when done=True

class StateResponse(BaseModel):
    state: list[dict[str, Any]]
    info: InfoPayload

class HealthResponse(BaseModel):
    status: str   # "ok"
```

### Dataset Configuration

```python
DIFFICULTY_CONFIG = {
    DifficultyLevel.easy:   {"rows": 50,  "max_steps": 15, "dirty": "data/easy_dirty.csv",   "clean": "data/easy_clean.csv"},
    DifficultyLevel.medium: {"rows": 200, "max_steps": 30, "dirty": "data/medium_dirty.csv", "clean": "data/medium_clean.csv"},
    DifficultyLevel.hard:   {"rows": 500, "max_steps": 60, "dirty": "data/hard_dirty.csv",   "clean": "data/hard_clean.csv"},
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Reward is always in [0.0, 1.0]

*For any* current DataFrame and clean DataFrame, `compute_reward` SHALL return a float value `r` such that `0.0 <= r <= 1.0`.

**Validates: Requirements 4.1**

---

### Property 2: Perfect state yields reward 1.0

*For any* clean DataFrame `df`, `compute_reward(df, df)` SHALL return exactly `1.0`.

**Validates: Requirements 4.3**

---

### Property 3: Grader is deterministic

*For any* current DataFrame and clean DataFrame, calling `compute_reward` twice with the same inputs SHALL return the same float value.

**Validates: Requirements 4.6**

---

### Property 4: drop_duplicates is idempotent

*For any* DataFrame `df`, applying `drop_duplicates` twice SHALL produce the same result as applying it once: `drop_duplicates(drop_duplicates(df)) == drop_duplicates(df)`.

**Validates: Requirements 5.1**

---

### Property 5: Action on missing column leaves DataFrame unchanged

*For any* DataFrame `df` and any action that references a column name not present in `df`, the DataFrame returned SHALL be identical to `df` (no mutation), and a `ColumnNotFoundError` SHALL be raised.

**Validates: Requirements 5.7**

---

### Property 6: Step count monotonically increases and terminates

*For any* episode, the step count after N calls to `/step` SHALL equal N, and `done` SHALL become true when step count reaches `max_steps` for the selected difficulty.

**Validates: Requirements 2.4, 6.3, 6.4**

---

### Property 7: Observation schema completeness

*For any* call to `/step` or `/reset`, the returned Observation SHALL contain all required fields: `state`, `reward`, `done`, `info.step`, `info.difficulty`. When `done=True`, `info.final_score` SHALL be present and non-null.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

---

### Property 8: Reset restores dirty dataset

*For any* sequence of actions applied during an episode, calling `/reset` with the same difficulty SHALL restore the working DataFrame to be identical to the original Dirty_Dataset (row count, column names, and cell values).

**Validates: Requirements 1.5**

---

### Property 9: fill_missing round-trip — no new nulls introduced

*For any* DataFrame `df` and column `col` with a valid fill strategy, after `fill_missing(df, col, strategy)`, the number of null values in `col` SHALL be zero (all nulls filled).

**Validates: Requirements 5.2**

---

### Property 10: Reward increases (or stays equal) after a correct cleaning action

*For any* episode state where the working DataFrame is not yet equal to the clean DataFrame, applying a cleaning action that moves the dataset closer to the clean state SHALL result in a reward greater than or equal to the reward before the action.

**Validates: Requirements 4.5**

---

## Error Handling

| Scenario | HTTP Status | Behavior |
|---|---|---|
| `/step` called when `done=True` | 400 Bad Request | Return error JSON: `{"detail": "Episode is done. Call /reset to start a new episode."}` |
| Action references non-existent column | 200 OK | Return Observation with `reward` unchanged, `info.error` set to descriptive message, dataset unchanged |
| Unsupported `action_type` | 422 Unprocessable Entity | FastAPI automatic Pydantic validation error |
| Invalid request body schema | 422 Unprocessable Entity | FastAPI automatic Pydantic validation error |
| Dataset file missing at startup | 500 / startup crash | `DatasetLoadError` raised with missing file path; server does not start |
| `GOOGLE_API_KEY` not set (baseline) | Script exits 1 | Print to stderr: `"Error: GOOGLE_API_KEY environment variable is not set."` |
| `cast_column` with incompatible dtype | 200 OK | Return Observation with `info.error` describing the cast failure, dataset unchanged |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required and complementary:
- **Unit tests** cover specific examples, edge cases, and error conditions
- **Property-based tests** verify universal properties across randomly generated inputs

### Property-Based Testing

Library: **Hypothesis** (Python)

Each property-based test runs a minimum of 100 iterations. Tests are tagged with the feature and property number for traceability.

```python
# Tag format example:
# Feature: data-cleaning-agent, Property 1: Reward is always in [0.0, 1.0]
@settings(max_examples=100)
@given(st.data())
def test_reward_bounds(data):
    ...
```

**Property test mapping:**

| Property | Test | Strategy |
|---|---|---|
| P1: Reward bounds | Generate random DataFrames (varying rows, nulls, dtypes) | `hypothesis` DataFrame strategies |
| P2: Perfect state → 1.0 | Generate random clean DataFrames, pass as both args | `st.data()` |
| P3: Grader determinism | Generate random pair, call twice, assert equal | `st.data()` |
| P4: drop_duplicates idempotent | Generate DataFrames with random duplicate rows | `st.lists` + `pd.DataFrame` |
| P5: Missing column → unchanged | Generate DataFrame + column name not in df | `st.text()` filtered |
| P6: Step count / termination | Simulate N steps, assert count and done flag | `st.integers` |
| P7: Observation schema | Generate random actions, assert response fields | `st.from_type(ActionRequest)` |
| P8: Reset restores dirty | Apply random action sequence, reset, compare | `st.lists(st.from_type(ActionRequest))` |
| P9: fill_missing removes nulls | Generate DataFrame with nulls in a column | `st.data()` |
| P10: Reward monotonicity | Apply known-correct action, compare before/after reward | `st.data()` |

### Unit Tests

Framework: **pytest**

Key unit test areas:
- Each action function with valid inputs (happy path)
- Each action function with invalid column names (error path)
- `cast_column` with incompatible types
- Grader sub-score calculations with known inputs
- `/health` endpoint returns 200
- `/reset` with each difficulty level
- `/step` after `done=True` returns 400
- Baseline script exits with code 1 when `GOOGLE_API_KEY` is unset
- Observation schema fields present on every response

### Test File Structure

```
tests/
  unit/
    test_actions.py
    test_grader.py
    test_endpoints.py
    test_datasets.py
    test_baseline.py
  property/
    test_grader_properties.py
    test_action_properties.py
    test_episode_properties.py
    test_observation_properties.py
```
