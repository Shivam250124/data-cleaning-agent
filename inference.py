#!/usr/bin/env python3
"""
Baseline inference script for Data Cleaning Agent OpenEnv.

Uses the OpenAI API client (compatible with Groq, OpenAI, etc.)
Falls back to deterministic baseline when OPENAI_API_KEY is not set.

Usage:
    # With LLM (requires API key):
    export OPENAI_API_KEY="your-key"
    export OPENAI_BASE_URL="https://api.groq.com/openai/v1"  # optional
    python inference.py
    
    # Without API key (uses deterministic baseline):
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.models import ActionType, DifficultyLevel


REQUEST_DELAY = 1.0
USE_LLM = False  # Will be set to True if API key is available


def get_client() -> Optional["OpenAI"]:
    """Get OpenAI client if API key is available, otherwise return None."""
    global USE_LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: OPENAI_API_KEY not set. Using deterministic baseline instead.")
        USE_LLM = False
        return None

    USE_LLM = True
    from openai import OpenAI
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def parse_action(text: str) -> tuple[str, dict]:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        return data.get("action_type", ""), data.get("params", {})
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("action_type", ""), data.get("params", {})
            except json.JSONDecodeError:
                pass
    return "", {}


def get_deterministic_actions(obs) -> list[tuple[ActionType, dict]]:
    """Generate a deterministic sequence of cleaning actions based on the data."""
    actions = []
    
    # Always start with drop_duplicates
    actions.append((ActionType.drop_duplicates, {}))
    
    if obs.state:
        columns = list(obs.state[0].keys())
        
        # Strip whitespace on all string-like columns
        for col in columns:
            sample_vals = [row.get(col) for row in obs.state[:5] if row.get(col) is not None]
            if sample_vals and isinstance(sample_vals[0], str):
                actions.append((ActionType.strip_whitespace, {"column": col}))
        
        # Fill missing values using mode for categorical, mean for numeric
        for col in columns:
            sample_vals = [row.get(col) for row in obs.state[:10] if row.get(col) is not None]
            if sample_vals:
                if isinstance(sample_vals[0], (int, float)):
                    actions.append((ActionType.fill_missing, {"column": col, "strategy": "mean"}))
                else:
                    actions.append((ActionType.fill_missing, {"column": col, "strategy": "mode"}))
    
    return actions


def run_episode_deterministic(env: DataCleaningEnv, difficulty: DifficultyLevel) -> float:
    """Run episode using deterministic baseline (no LLM)."""
    print(f"\n{'='*50}")
    print(f"Running {difficulty.value.upper()} difficulty (deterministic)")
    print(f"{'='*50}")

    obs = env.reset(difficulty)
    print(f"Max steps: {obs.info.max_steps}, Initial rows: {len(obs.state)}")

    actions = get_deterministic_actions(obs)
    
    for step, (action_type, params) in enumerate(actions):
        if obs.done:
            break
        if step >= obs.info.max_steps:
            break
            
        try:
            print(f"  Step {step+1}: {action_type.value} {params}")
            obs = env.step(action_type, params)
            print(f"  Reward: {obs.reward:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    score = env.get_current_score()
    print(f"Final score: {score:.4f}")
    return score


def run_episode_llm(env: DataCleaningEnv, client, difficulty: DifficultyLevel) -> float:
    """Run episode using LLM-based inference."""
    model = os.environ.get("BASELINE_MODEL", "llama-3.1-8b-instant")
    print(f"\n{'='*50}")
    print(f"Running {difficulty.value.upper()} difficulty (LLM)")
    print(f"{'='*50}")

    obs = env.reset(difficulty)
    print(f"Max steps: {obs.info.max_steps}, Initial rows: {len(obs.state)}")

    for step in range(obs.info.max_steps):
        if obs.done:
            break

        preview = obs.state[:10]
        prompt = f"""You are a data cleaning agent. Choose one action to clean this dataset.

Current data (first 10 rows): {json.dumps(preview, default=str)}
Step: {step + 1}/{obs.info.max_steps}

IMPORTANT - Use EXACTLY these parameter names:
- drop_duplicates: {{}}
- fill_missing: {{"column": "col_name", "strategy": "mean"}} or {{"column": "col_name", "value": "x"}}
- cast_column: {{"column": "col_name", "dtype": "int"}}
- rename_column: {{"old_name": "old", "new_name": "new"}}
- strip_whitespace: {{"column": "col_name"}}
- drop_column: {{"column": "col_name"}}

The key is always "column" (not "column_name", not "col").

Respond with JSON only: {{"action_type": "...", "params": {{...}}}}"""

        time.sleep(REQUEST_DELAY)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            action_type, params = parse_action(response.choices[0].message.content)
            print(f"  Step {step+1}: {action_type} {params}")

            if not action_type or action_type == "done":
                break

            obs = env.step(ActionType(action_type), params)
            print(f"  Reward: {obs.reward:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            obs = env.step(ActionType.drop_duplicates, {})

    score = env.get_current_score()
    print(f"Final score: {score:.4f}")
    return score


def main() -> None:
    print("Data Cleaning Agent - Baseline")
    print("=" * 50)

    client = get_client()
    registry = DatasetRegistry()
    env = DataCleaningEnv(registry=registry)

    scores = {}
    start = time.time()

    for difficulty in DifficultyLevel:
        if USE_LLM and client is not None:
            scores[difficulty.value] = run_episode_llm(env, client, difficulty)
        else:
            scores[difficulty.value] = run_episode_deterministic(env, difficulty)

    elapsed = time.time() - start

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    mode = "LLM" if USE_LLM else "Deterministic"
    print(f"Mode: {mode}")
    for diff, score in scores.items():
        print(f"  {diff.upper():8s}: {score:.4f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"All scores in [0.0, 1.0]: {all(0.0 <= s <= 1.0 for s in scores.values())}")


if __name__ == "__main__":
    main()
