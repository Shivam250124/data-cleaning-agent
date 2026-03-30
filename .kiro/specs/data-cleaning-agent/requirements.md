# Requirements Document

## Introduction

The Data Cleaning Agent OpenEnv environment is a REST API-based evaluation platform for AI agents performing multi-step data cleaning tasks. It mirrors real-world data analyst workflows across three difficulty levels (Easy, Medium, Hard), provides synthetic dirty/clean CSV dataset pairs, and scores agent performance using a dense shaped reward function. The baseline inference script uses the Google Gemini API to ensure a zero-cost development and evaluation cycle. The environment is deployed as a FastAPI HTTP server on Hugging Face Spaces.

## Glossary

- **Environment**: The Data Cleaning OpenEnv system — the FastAPI HTTP server that manages task state, accepts agent actions, and returns observations and rewards.
- **Agent**: An AI model (e.g., Gemini 1.5 Flash) that interacts with the Environment via the REST API.
- **Episode**: A single run of a task from reset to terminal state or max steps reached.
- **Step**: One action-observation-reward cycle within an Episode.
- **Dirty_Dataset**: A synthetic CSV file containing intentional data quality issues (missing values, duplicates, type errors, formatting inconsistencies).
- **Clean_Dataset**: The ground-truth CSV file corresponding to a Dirty_Dataset after all issues are resolved.
- **Grader**: A deterministic scoring function that compares the Agent's current dataset state against the Clean_Dataset and returns a scalar reward in [0.0, 1.0].
- **Reward**: A scalar float in [0.0, 1.0] returned by the Grader after each Step.
- **Observation**: The JSON payload returned to the Agent after each Step, containing current dataset state, reward, done flag, and info.
- **Action**: A JSON payload sent by the Agent to the /step endpoint describing a data cleaning operation.
- **Difficulty_Level**: One of Easy, Medium, or Hard, determining dataset size and max steps.
- **Baseline_Script**: `scripts/baseline.py` — the reference inference script using the Gemini API.
- **GOOGLE_API_KEY**: The environment variable holding the Google Gemini API key.

---

## Requirements

### Requirement 1: REST API Endpoints

**User Story:** As an AI agent, I want to interact with the environment through a REST API, so that I can perform data cleaning tasks in a standardized, model-agnostic way.

#### Acceptance Criteria

1. THE Environment SHALL expose a `/step` POST endpoint that accepts an Action payload and returns an Observation.
2. THE Environment SHALL expose a `/reset` POST endpoint that initializes a new Episode and returns the initial Observation.
3. THE Environment SHALL expose a `/state` GET endpoint that returns the current dataset state without advancing the Episode.
4. THE Environment SHALL expose a `/health` GET endpoint that returns a 200 OK response confirming the server is running.
5. WHEN a `/reset` request is received, THE Environment SHALL reset the dataset to the Dirty_Dataset for the specified Difficulty_Level and return the initial Observation.
6. WHEN a `/step` request is received with a valid Action, THE Environment SHALL apply the Action to the current dataset, compute the Reward via the Grader, and return the updated Observation.
7. IF a `/step` request is received after the Episode has ended (done=true), THEN THE Environment SHALL return an error response indicating the Episode must be reset.
8. IF a request body fails schema validation, THEN THE Environment SHALL return a 422 Unprocessable Entity response with a descriptive error message.

---

### Requirement 2: Task Difficulty Levels

**User Story:** As an evaluator, I want three difficulty levels with distinct dataset sizes and step budgets, so that I can assess agent capability across a range of task complexities.

#### Acceptance Criteria

1. THE Environment SHALL support an Easy difficulty level using a Customer dataset of 50 rows with a maximum of 15 steps per Episode.
2. THE Environment SHALL support a Medium difficulty level using a Sales dataset of 200 rows with a maximum of 30 steps per Episode.
3. THE Environment SHALL support a Hard difficulty level using an HR dataset of 500 rows with a maximum of 60 steps per Episode.
4. WHEN the number of Steps in an Episode reaches the maximum for the selected Difficulty_Level, THE Environment SHALL set the done flag to true in the next Observation.
5. THE Environment SHALL include all three Dirty_Dataset and Clean_Dataset CSV pairs as static assets bundled with the server.

---

### Requirement 3: Synthetic Datasets

**User Story:** As an evaluator, I want synthetic dirty/clean CSV dataset pairs, so that the Grader can deterministically score agent performance against a known ground truth.

#### Acceptance Criteria

1. THE Environment SHALL provide a Dirty_Dataset for each Difficulty_Level containing at least four categories of data quality issues: missing values, duplicate rows, type errors, and formatting inconsistencies.
2. THE Environment SHALL provide a corresponding Clean_Dataset for each Difficulty_Level that represents the fully corrected ground-truth version of the Dirty_Dataset.
3. THE Environment SHALL load Dirty_Dataset and Clean_Dataset files at server startup and keep them in memory for the duration of the server process.
4. IF a dataset file is missing at startup, THEN THE Environment SHALL raise a startup error with a descriptive message identifying the missing file.

---

### Requirement 4: Dense Shaped Reward Function (Grader)

**User Story:** As an AI agent, I want a dense shaped reward signal after each step, so that I can learn incrementally rather than only receiving feedback at the end of an episode.

#### Acceptance Criteria

1. THE Grader SHALL compute a scalar Reward strictly within the range [0.0, 1.0] after every Step.
2. THE Grader SHALL compare the Agent's current dataset state against the Clean_Dataset using deterministic, reproducible logic.
3. WHEN the Agent's dataset state exactly matches the Clean_Dataset, THE Grader SHALL return a Reward of 1.0.
4. WHEN the Agent's dataset state is identical to the Dirty_Dataset (no changes made), THE Grader SHALL return a Reward strictly less than 1.0.
5. THE Grader SHALL produce shaped (intermediate) rewards that increase monotonically as the dataset state approaches the Clean_Dataset.
6. THE Grader SHALL be deterministic: given the same dataset state and Clean_Dataset, THE Grader SHALL always return the same Reward.

---

### Requirement 5: Supported Cleaning Actions

**User Story:** As an AI agent, I want a well-defined set of data cleaning actions, so that I can systematically address data quality issues in the dataset.

#### Acceptance Criteria

1. THE Environment SHALL support a `drop_duplicates` action that removes all duplicate rows from the current dataset.
2. THE Environment SHALL support a `fill_missing` action that fills missing values in a specified column with a specified fill value or strategy (e.g., mean, median, mode, or a literal value).
3. THE Environment SHALL support a `cast_column` action that converts a specified column to a specified data type (e.g., int, float, str, datetime).
4. THE Environment SHALL support a `rename_column` action that renames a specified column to a new name.
5. THE Environment SHALL support a `strip_whitespace` action that removes leading and trailing whitespace from all string values in a specified column.
6. THE Environment SHALL support a `drop_column` action that removes a specified column from the current dataset.
7. IF an Action references a column that does not exist in the current dataset, THEN THE Environment SHALL return an error Observation with a descriptive message and leave the dataset unchanged.
8. IF an Action specifies an unsupported action type, THEN THE Environment SHALL return a 422 Unprocessable Entity response.

---

### Requirement 6: Observation Schema

**User Story:** As an AI agent, I want a structured observation after each step, so that I can understand the current state of the dataset and decide on the next action.

#### Acceptance Criteria

1. THE Environment SHALL include a `state` field in every Observation containing a JSON-serializable representation of the current dataset (e.g., records-oriented JSON).
2. THE Environment SHALL include a `reward` field in every Observation containing the scalar Reward computed by the Grader.
3. THE Environment SHALL include a `done` field in every Observation containing a boolean indicating whether the Episode has ended.
4. THE Environment SHALL include an `info` field in every Observation containing at minimum the current step count and the selected Difficulty_Level.
5. WHEN `done` is true, THE Environment SHALL include a `final_score` field in the `info` object containing the terminal Reward.

---

### Requirement 7: Baseline Inference Script

**User Story:** As a hackathon evaluator, I want a reproducible baseline script using the Gemini API, so that I can verify the environment produces valid scores without any additional setup cost.

#### Acceptance Criteria

1. THE Baseline_Script SHALL read the GOOGLE_API_KEY exclusively from environment variables and SHALL NOT hardcode any API key value.
2. THE Baseline_Script SHALL invoke the Gemini 1.5 Flash model with temperature=0 for all inference calls.
3. THE Baseline_Script SHALL run all three Difficulty_Level tasks sequentially and print the final Grader score for each task to standard output.
4. THE Baseline_Script SHALL complete all three tasks within 5 minutes under normal network conditions, adhering to Gemini free-tier rate limits.
5. WHEN the GOOGLE_API_KEY environment variable is not set, THE Baseline_Script SHALL exit with a non-zero status code and print a descriptive error message to standard error.
6. THE Baseline_Script SHALL produce the same scores on repeated runs given the same dataset and model configuration (reproducibility).

---

### Requirement 8: Security and API Key Handling

**User Story:** As a system operator, I want the API key to be handled securely, so that credentials are never exposed in source code or API responses.

#### Acceptance Criteria

1. THE Environment SHALL read GOOGLE_API_KEY exclusively from environment variables at runtime.
2. THE Environment SHALL NOT include GOOGLE_API_KEY or any portion of it in any HTTP response body, header, or log output.
3. IF GOOGLE_API_KEY is required and not set, THEN THE Baseline_Script SHALL fail with a clear error rather than proceeding with an empty or invalid key.

---

### Requirement 9: Deployment and Infrastructure

**User Story:** As a hackathon participant, I want the environment to be deployable on Hugging Face Spaces, so that evaluators can access it without local setup.

#### Acceptance Criteria

1. THE Environment SHALL run as a FastAPI HTTP server on port 7860.
2. THE Environment SHALL be containerized using Docker with a Dockerfile that installs all Python dependencies.
3. THE Environment SHALL respond to a GET /health request with HTTP 200 within 5 seconds of server startup completing.
4. THE Environment SHALL be CPU-only and SHALL NOT require a GPU to run.
5. WHERE a `requirements.txt` or equivalent dependency file is provided, THE Environment SHALL pin all dependency versions to ensure reproducible builds.
