#!/usr/bin/env python3
"""
Baseline inference script for Data Cleaning Agent OpenEnv.

Uses the OpenAI API client (compatible with Groq, OpenAI, etc.)

Usage:
    export OPENAI_API_KEY="your-key"
    export OPENAI_BASE_URL="https://api.groq.com/openai/v1"  # optional
    python scripts/baseline.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.models import ActionType, DifficultyLevel


REQUEST_DELAY = 1.0


def get_client() -> OpenAI | None:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY environment variable.", file=sys.stderr)
        return None

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


def run_episode(env: DataCleaningEnv, client: OpenAI, difficulty: DifficultyLevel) -> float:
    model = os.environ.get("BASELINE_MODEL", "llama-3.1-8b-instant")
    print(f"\n{'='*50}")
    print(f"Running {difficulty.value.upper()} difficulty")
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
    if client is None:
        print("Skipping baseline due to missing API key.")
        print("Set HF_TOKEN, API_KEY, or OPENAI_API_KEY to run the LLM baseline.")
        return

    registry = DatasetRegistry()
    env = DataCleaningEnv(registry=registry)

    scores = {}
    start = time.time()

    for difficulty in DifficultyLevel:
        scores[difficulty.value] = run_episode(env, client, difficulty)

    elapsed = time.time() - start

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for diff, score in scores.items():
        print(f"  {diff.upper():8s}: {score:.4f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"All scores in [0.0, 1.0]: {all(0.0 <= s <= 1.0 for s in scores.values())}")


if __name__ == "__main__":
    main()
