#!/usr/bin/env python3
"""
Inference Script for Data Cleaning Agent OpenEnv
===================================

MANDATORY - Before submitting, ensure the following variables are defined
in your environment configuration:

    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment
                     if you are using from_docker_image() method

- Defaults are set only for API_BASE_URL and MODEL_NAME
  (and should reflect your active inference setup):

    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

- The inference script must be named `inference.py` and placed in the root
  directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

Example:
    [START] task=easy env=data-cleaning-agent model=llama-3.1-8b-instant
    [STEP] step=1 action=drop_duplicates({}) reward=0.02 done=false error=null
    [STEP] step=2 action=fill_missing({'column':'age','strategy':'mean'}) reward=0.01 done=false error=null
    [END] success=true steps=2 score=0.95 rewards=0.02,0.01
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.models import ActionType, DifficultyLevel

# Environment variables (MANDATORY)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Benchmark name
BENCHMARK = "data-cleaning-agent"

# Request delay to avoid rate limiting
REQUEST_DELAY = 1.0
TEMPERATURE = 0.0


def get_client() -> OpenAI:
    """Get OpenAI client configured via environment variables."""
    api_key = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def parse_action(text: str) -> tuple[str, dict]:
    """Parse LLM response to extract action_type and params."""
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


def format_action(action_type: str, params: dict) -> str:
    """Format action for logging."""
    return f"{action_type}({params})"


def run_episode(
    env: DataCleaningEnv, client: OpenAI, task_name: str
) -> tuple[bool, int, float, List[float]]:
    """
    Run a single episode with structured logging.
    
    Returns:
        tuple: (success, steps, score, rewards)
    """
    rewards: List[float] = []
    steps = 0
    success = False
    last_error: Optional[str] = None
    
    # [START] line
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
    
    try:
        difficulty = DifficultyLevel(task_name)
        obs = env.reset(difficulty)
        
        for step in range(obs.info.max_steps):
            if obs.done:
                break
            
            steps = step + 1
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
            
            action_type = ""
            params = {}
            error_msg: Optional[str] = None
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=200,
                )
                action_type, params = parse_action(response.choices[0].message.content)
                
                if not action_type or action_type == "done":
                    # No valid action, end episode
                    done_str = "true" if obs.done else "false"
                    print(f"[STEP] step={steps} action=done({{}}) reward=0.00 done={done_str} error=null")
                    rewards.append(0.0)
                    break
                
                obs = env.step(ActionType(action_type), params)
                reward = obs.reward
                rewards.append(reward)
                
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                last_error = error_msg
                # Fallback action on error
                try:
                    obs = env.step(ActionType.drop_duplicates, {})
                    action_type = "drop_duplicates"
                    params = {}
                    reward = obs.reward
                    rewards.append(reward)
                except Exception:
                    reward = 0.0
                    rewards.append(reward)
            
            # [STEP] line
            action_str = format_action(action_type, params)
            done_str = "true" if obs.done else "false"
            error_str = f"{error_msg}" if error_msg else "null"
            print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")
        
        score = env.get_current_score()
        success = True
        
    except Exception as e:
        last_error = str(e).replace("\n", " ")
        score = 0.0
        success = False
    
    # [END] line - always emitted
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")
    
    return success, steps, score, rewards


def main() -> None:
    """Main entry point - runs all tasks."""
    try:
        client = get_client()
    except ValueError as e:
        # If no API key, still run but report error
        print(f"[START] task=easy env={BENCHMARK} model={MODEL_NAME}")
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        print(f"[START] task=medium env={BENCHMARK} model={MODEL_NAME}")
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        print(f"[START] task=hard env={BENCHMARK} model={MODEL_NAME}")
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        sys.exit(0)  # Exit cleanly even without API key
    
    registry = DatasetRegistry()
    env = DataCleaningEnv(registry=registry)
    
    # Run all difficulty levels as separate tasks
    for difficulty in DifficultyLevel:
        run_episode(env, client, difficulty.value)


if __name__ == "__main__":
    main()
