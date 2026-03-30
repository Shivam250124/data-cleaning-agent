---
title: Data Cleaning Agent
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Data Cleaning Agent OpenEnv

A real-world OpenEnv environment where AI agents must clean dirty CSV datasets through a series of data cleaning actions. Agents receive dense shaped rewards after each step, enabling incremental learning.

## Environment Description

The agent receives a dirty DataFrame and must apply cleaning actions to match a clean ground-truth dataset. Three difficulty levels simulate real analyst workflows:

| Task | Dataset | Rows | Max Steps | Expected Score |
|------|---------|------|-----------|----------------|
| Easy | Customer data | 50 | 15 | ~0.78 |
| Medium | Sales data | 200 | 30 | ~0.55 |
| Hard | HR data | 500 | 60 | ~0.30 |

Each dirty dataset contains: missing values, duplicate rows, type errors, and formatting inconsistencies.

## Action Space

```json
{
  "action_type": "drop_duplicates | fill_missing | cast_column | rename_column | strip_whitespace | drop_column",
  "params": {}
}
```

| Action | Required Params |
|--------|----------------|
| `drop_duplicates` | none |
| `fill_missing` | `column`, `value` or `strategy` (mean/median/mode/ffill/bfill) |
| `cast_column` | `column`, `dtype` (int/float/str/datetime) |
| `rename_column` | `old_name`, `new_name` |
| `strip_whitespace` | `column` |
| `drop_column` | `column` |

## Observation Space

```json
{
  "state": [{"col1": "val1", ...}],
  "reward": 0.85,
  "done": false,
  "info": {
    "step": 3,
    "max_steps": 15,
    "difficulty": "easy",
    "final_score": null
  }
}
```

## Reward Function

Dense shaped reward based on 4 sub-scores:
- **duplicate_score** (10%): fraction of duplicates removed
- **missing_score** (10%): fraction of missing values filled
- **type_score** (10%): fraction of columns with correct dtypes
- **value_score** (70%): cell-level exact match against clean dataset

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode `{"difficulty": "easy"}` |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | GET | Get current grader score |
| `/baseline` | POST | Run baseline inference |

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Baseline

```bash
export OPENAI_API_KEY="your-groq-or-openai-key"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"  # optional, defaults to Groq
python scripts/baseline.py
```

Baseline scores: Easy ~0.94, Medium ~0.98, Hard ~0.99

## Docker

```bash
docker build -t data-cleaning-agent .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key data-cleaning-agent
```
