#!/usr/bin/env python3
"""
Baseline inference script using Google Gemini API.

Runs gemini-1.5-flash against all 3 difficulty levels and prints scores.

Usage:
    export GOOGLE_API_KEY="your-api-key"
    python scripts/baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.models import ActionType, DifficultyLevel


# Rate limiting for free tier
REQUEST_DELAY = 4.0  # seconds between API calls


def get_api_key() -> str:
    """Get the Google API key from environment variables."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)
    return api_key


def create_model() -> genai.GenerativeModel:
    """Create and configure the Gemini model."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=1024,
        ),
    )


SYSTEM_PROMPT = """You are a data cleaning agent. Your task is to clean a dirty DataFrame to match a target schema.

Available actions:
1. drop_duplicates - Remove duplicate rows
   params: {"subset": ["col1", "col2"]} (optional - columns to consider)

2. fill_missing - Fill missing/NaN values
   params: {"column": "col_name", "value": "fill_value"} or
   params: {"column": "col_name", "strategy": "mean|median|mode|ffill|bfill"}

3. cast_column - Convert column type
   params: {"column": "col_name", "dtype": "int|float|str|datetime"}

4. rename_column - Rename a column
   params: {"old_name": "old", "new_name": "new"}

5. strip_whitespace - Remove leading/trailing whitespace
   params: {"column": "col_name"}

6. drop_column - Remove a column
   params: {"column": "col_name"}

Respond with ONLY a JSON object in this exact format:
{"action_type": "<action_name>", "params": {<params>}}

If no more cleaning is needed, respond with:
{"action_type": "done", "params": {}}

Analyze the data carefully and choose the most impactful action each step."""


def format_state_for_prompt(state: list[dict[str, Any]], max_rows: int = 20) -> str:
    """Format the state as a readable table for the prompt."""
    if not state:
        return "Empty DataFrame"

    # Get columns
    columns = list(state[0].keys())

    # Build table header
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))

    # Add rows (limit for token efficiency)
    display_rows = state[:max_rows]
    for row in display_rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append(" | ".join(values))

    if len(state) > max_rows:
        lines.append(f"... ({len(state) - max_rows} more rows)")

    return "\n".join(lines)


def parse_action_response(response_text: str) -> tuple[str, dict[str, Any]]:
    """Parse the model's response into action type and params."""
    # Clean up response
    text = response_text.strip()

    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())
        action_type = data.get("action_type", "")
        params = data.get("params", {})
        return action_type, params
    except json.JSONDecodeError:
        # Try to find JSON object in text
        import re
        match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("action_type", ""), data.get("params", {})
            except json.JSONDecodeError:
                pass

        return "", {}


def run_episode(
    env: DataCleaningEnv,
    model: genai.GenerativeModel,
    difficulty: DifficultyLevel,
) -> float:
    """Run a single episode and return the final score."""
    print(f"\n{'='*60}")
    print(f"Running {difficulty.value.upper()} difficulty")
    print(f"{'='*60}")

    # Reset environment
    obs = env.reset(difficulty)
    max_steps = obs.info.max_steps
    print(f"Max steps: {max_steps}")
    print(f"Initial rows: {len(obs.state)}")

    step = 0
    while not obs.done and step < max_steps:
        step += 1
        print(f"\nStep {step}/{max_steps}")

        # Format state for prompt
        state_str = format_state_for_prompt(obs.state)

        # Build prompt
        prompt = f"""{SYSTEM_PROMPT}

Current DataFrame ({len(obs.state)} rows):
{state_str}

Current step: {step}/{max_steps}

What action should be taken next?"""

        # Call Gemini API
        time.sleep(REQUEST_DELAY)  # Rate limiting
        try:
            response = model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            print(f"  API error: {e}")
            break

        # Parse response
        action_type, params = parse_action_response(response_text)
        print(f"  Action: {action_type}")
        print(f"  Params: {params}")

        # Check if done
        if action_type == "done" or not action_type:
            print("  Agent signaled done or invalid action")
            break

        # Validate and execute action
        try:
            action_enum = ActionType(action_type)
            obs = env.step(action_enum, params)
            print(f"  Reward: {obs.reward:.4f}")
            print(f"  Rows: {len(obs.state)}")
        except ValueError as e:
            print(f"  Invalid action: {e}")
            continue
        except Exception as e:
            print(f"  Error executing action: {e}")
            continue

    # Get final score
    final_score = env.get_current_score()
    print(f"\nFinal score: {final_score:.4f}")
    return final_score


def main() -> None:
    """Run baseline against all difficulties."""
    print("Data Cleaning Agent - Gemini Baseline")
    print("=" * 60)

    # Initialize
    model = create_model()
    registry = DatasetRegistry()
    env = DataCleaningEnv(registry=registry)

    # Run all difficulties
    scores = {}
    start_time = time.time()

    for difficulty in [DifficultyLevel.easy, DifficultyLevel.medium, DifficultyLevel.hard]:
        scores[difficulty.value] = run_episode(env, model, difficulty)

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for diff, score in scores.items():
        print(f"  {diff.upper():8s}: {score:.4f}")
    print(f"\nTotal time: {elapsed:.1f} seconds")

    # Verify scores are in range
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    print(f"All scores in [0.0, 1.0]: {all_valid}")


if __name__ == "__main__":
    main()
