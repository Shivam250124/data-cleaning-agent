"""
Data Cleaning Agent — Arena UI
================================
Gradio app mounted inside FastAPI at /demo.

- On HF Spaces  : served at https://<space>.hf.space/demo  (port 7860)
- Locally       : python demo.py  →  http://localhost:7861

API key priority (option-b deployment):
  1. GROQ_API_KEY env var  (set as HF Space secret)
  2. OPENAI_API_KEY env var
  3. Whatever the visitor types in the UI field
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from queue import Queue, Empty
from typing import Generator

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ── Internal imports (same process as FastAPI) ────────────────────────────────
from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv
from app.models import ActionType, DifficultyLevel
from app.grader import compute_score

# ── One shared env instance (thread-safe via sequential runner) ───────────────
_registry = DatasetRegistry()
_env = DataCleaningEnv(registry=_registry)
_env_lock = threading.Lock()

# ── API key: prefer server-side secret so visitors need nothing ───────────────
_SERVER_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_SERVER_OAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Model registry ────────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    "⚙️ Deterministic Baseline": {"type": "baseline"},
    "🦙 LLaMA 3.1 8B  (Groq)": {
        "type": "llm",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "server_key": _SERVER_GROQ_KEY,
    },
    "✨ GPT-4o Mini  (OpenAI)": {
        "type": "llm",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "server_key": _SERVER_OAI_KEY,
    },
}

DIFFICULTY_LABELS: dict[str, str] = {
    "🟢 Easy   — 50 rows, 15 steps":   "easy",
    "🟡 Medium — 200 rows, 30 steps":  "medium",
    "🔴 Hard   — 500 rows, 60 steps":  "hard",
}

SUB_SCORE_LABELS = ["Row Score (20%)", "Null Score (20%)", "Cell Match (60%)"]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_key(model_cfg: dict, visitor_key: str) -> str:
    """Return server secret if available, otherwise fall back to visitor key."""
    return model_cfg.get("server_key") or visitor_key.strip()


def _dirty_preview(difficulty: str) -> pd.DataFrame:
    """Return first 10 rows of the dirty dataset as a DataFrame."""
    try:
        df = _registry.get_dirty(DifficultyLevel(difficulty))
        return df.head(10)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ═════════════════════════════════════════════════════════════════════════════
# LLM action chooser
# ═════════════════════════════════════════════════════════════════════════════

def _parse_action(text: str) -> tuple[str, dict]:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        return data.get("action_type", ""), data.get("params", {})
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data.get("action_type", ""), data.get("params", {})
            except json.JSONDecodeError:
                pass
    return "", {}


def _llm_pick_action(client, model_id: str, state: list, step: int, max_steps: int) -> tuple[str, dict]:
    preview = state[:8]
    prompt = (
        "You are a data cleaning agent. Pick ONE action to improve this dataset.\n\n"
        f"Data (first 8 rows): {json.dumps(preview, default=str)}\n"
        f"Step: {step}/{max_steps}\n\n"
        "Available actions (use EXACT parameter names):\n"
        '- drop_duplicates: {}\n'
        '- fill_missing:    {"column": "col_name", "strategy": "mean|median|mode|ffill|bfill"}\n'
        '- cast_column:     {"column": "col_name", "dtype": "int|float|str|datetime"}\n'
        '- rename_column:   {"old_name": "old", "new_name": "new"}\n'
        '- strip_whitespace:{"column": "col_name"}\n'
        '- drop_column:     {"column": "col_name"}\n\n'
        "Rules:\n"
        "- Key is always \"column\", never \"column_name\"\n"
        "- Only use columns that exist in the data\n"
        "- Priority: drop_duplicates → fill_missing → strip_whitespace\n\n"
        'Respond with ONLY JSON: {"action_type": "...", "params": {...}}'
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    return _parse_action(resp.choices[0].message.content)


# ═════════════════════════════════════════════════════════════════════════════
# Deterministic baseline
# ═════════════════════════════════════════════════════════════════════════════

def _baseline_pick_action(state: list, step: int) -> tuple[str, dict]:
    cols = list(state[0].keys()) if state else []
    sequence: list[tuple[str, dict]] = [("drop_duplicates", {})]
    for col in cols:
        sequence.append(("strip_whitespace", {"column": col}))
    for col in cols:
        sequence.append(("fill_missing", {"column": col, "strategy": "mode"}))
    idx = step - 1
    return sequence[idx] if idx < len(sequence) else ("drop_duplicates", {})


# ═════════════════════════════════════════════════════════════════════════════
# Episode runner — pushes events into a Queue
# ═════════════════════════════════════════════════════════════════════════════

def _run_episode(model_name: str, model_cfg: dict, difficulty: str,
                 visitor_key: str, q: Queue) -> None:
    """Run one full cleaning episode. All output goes to queue q."""

    def log(text: str):
        q.put({"type": "log", "model": model_name, "text": text})

    def err(text: str):
        q.put({"type": "error", "model": model_name, "text": text})

    start = time.time()

    # ── Reset env (lock so two models don't collide) ──────────────────────────
    with _env_lock:
        try:
            obs = _env.reset(DifficultyLevel(difficulty))
        except Exception as e:
            err(f"Reset failed: {e}")
            return
        max_steps = obs.info.max_steps
        state = obs.state
        done = obs.done

    log(f"🚀 Started | difficulty={difficulty} | max_steps={max_steps}")

    # ── LLM client setup ──────────────────────────────────────────────────────
    client = None
    if model_cfg["type"] == "llm":
        api_key = _resolve_key(model_cfg, visitor_key)
        if not api_key:
            err("❌ No API key available. Add GROQ_API_KEY secret or enter a key in the UI.")
            return
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=model_cfg["base_url"])
        except Exception as e:
            err(f"❌ LLM client init failed: {e}")
            return

    # ── Episode loop ──────────────────────────────────────────────────────────
    step = 0
    while not done and step < max_steps:
        step += 1

        # Choose action
        try:
            if model_cfg["type"] == "baseline":
                action_type, params = _baseline_pick_action(state, step)
                time.sleep(0.25)                   # pacing for live feel
            else:
                action_type, params = _llm_pick_action(
                    client, model_cfg["model"], state, step, max_steps
                )
                time.sleep(0.4)                    # rate-limit buffer
        except Exception as e:
            log(f"  Step {step:2d}: ⚠️  choice failed ({e}) — fallback drop_duplicates")
            action_type, params = "drop_duplicates", {}

        if not action_type or action_type == "done":
            log(f"  Step {step:2d}: 🏁 Model signalled done early")
            break

        # Execute action inside env lock
        with _env_lock:
            try:
                obs = _env.step(ActionType(action_type), params)
                reward = obs.reward
                done = obs.done
                state = obs.state
            except Exception as e:
                log(f"  Step {step:2d}: ⚠️  {action_type} → {e}")
                continue

        icon = "📈" if reward > 0 else ("📉" if reward < -0.001 else "➡️")
        p = json.dumps(params) if params else "{}"
        log(f"  Step {step:2d}: {action_type:<20s} {p:<32s} {icon} reward={reward:+.4f}")

    # ── Final score ───────────────────────────────────────────────────────────
    with _env_lock:
        final_score = _env.get_current_score()
        final_state = _env.state().state

    elapsed = time.time() - start
    sub = _estimate_sub_scores(final_state, final_score)

    log(f"✅ Done  |  score={final_score:.4f}  |  steps={step}/{max_steps}  |  {elapsed:.1f}s")
    q.put({
        "type": "score",
        "model": model_name,
        "score": final_score,
        "steps": step,
        "max_steps": max_steps,
        "elapsed": elapsed,
        "sub_scores": sub,
    })


def _estimate_sub_scores(state: list, total: float) -> list[float]:
    if not state or total == 0:
        return [0.0, 0.0, 0.0]
    df = pd.DataFrame(state)
    null_ratio = df.isnull().sum().sum() / max(df.size, 1)
    null_s = max(0.0, 1.0 - null_ratio)
    return [round(min(total * 1.1, 1.0), 3), round(null_s, 3), round(total, 3)]


# ═════════════════════════════════════════════════════════════════════════════
# Gradio streaming generator
# ═════════════════════════════════════════════════════════════════════════════

def run_arena(difficulty_label: str, visitor_key: str,
              model_choices: list[str]) -> Generator:
    """Yields (log_text, leaderboard_df, chart_fig) progressively."""

    if not model_choices:
        yield "⚠️ Select at least one model.", pd.DataFrame(), go.Figure()
        return

    difficulty = DIFFICULTY_LABELS.get(difficulty_label, "easy")
    q: Queue = Queue()
    results: dict[str, dict] = {}
    lines: list[str] = []

    # Run all models sequentially in one background thread
    def _runner():
        for name in model_choices:
            cfg = MODELS[name]
            q.put({"type": "log", "model": "🎬 Arena",
                   "text": f"━━━━━━━━━━  {name}  ━━━━━━━━━━"})
            _run_episode(name, cfg, difficulty, visitor_key, q)
        q.put({"type": "done"})

    threading.Thread(target=_runner, daemon=True).start()

    while True:
        try:
            ev = q.get(timeout=90)
        except Empty:
            break

        if ev["type"] == "done":
            break
        elif ev["type"] in ("log", "error"):
            lines.append(f"[{ev['model']}]  {ev['text']}")
            yield "\n".join(lines[-120:]), _leaderboard(results), _chart(results)
        elif ev["type"] == "score":
            results[ev["model"]] = ev
            lines.append(
                f"\n🏆 [{ev['model']}]  FINAL SCORE: {ev['score']:.4f}"
                f"  ({ev['steps']}/{ev['max_steps']} steps, {ev['elapsed']:.1f}s)\n"
            )
            yield "\n".join(lines[-120:]), _leaderboard(results), _chart(results)

    yield "\n".join(lines), _leaderboard(results), _chart(results)


# ═════════════════════════════════════════════════════════════════════════════
# Leaderboard + chart
# ═════════════════════════════════════════════════════════════════════════════

_MEDAL = {0: "🥇", 1: "🥈", 2: "🥉"}


def _leaderboard(results: dict) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Rank", "Model", "Score", "Steps", "Time (s)"])
    rows = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    df = pd.DataFrame([{
        "Model":    r["model"],
        "Score":    f"{r['score']:.4f}",
        "Steps":    f"{r['steps']}/{r['max_steps']}",
        "Time (s)": f"{r['elapsed']:.1f}",
    } for r in rows])
    df.insert(0, "Rank", [_MEDAL.get(i, str(i + 1)) for i in range(len(df))])
    return df


def _chart(results: dict) -> go.Figure:
    dark = "#0f172a"
    if not results:
        fig = go.Figure()
        fig.update_layout(
            title="Scores will appear here as models finish",
            plot_bgcolor=dark, paper_bgcolor=dark,
            font={"color": "#e2e8f0"},
        )
        return fig

    ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    names  = [r["model"] for r in ranked]
    totals = [r["score"] for r in ranked]
    subs   = [r["sub_scores"] for r in ranked]
    sub_colors = ["#60a5fa", "#34d399", "#a78bfa"]

    fig = go.Figure()
    for si, label in enumerate(SUB_SCORE_LABELS):
        fig.add_trace(go.Bar(
            name=label,
            x=names,
            y=[subs[mi][si] for mi in range(len(names))],
            marker_color=sub_colors[si],
            opacity=0.8,
        ))
    fig.add_trace(go.Scatter(
        name="Final Score",
        x=names, y=totals,
        mode="markers+text",
        marker=dict(size=18, color="#fbbf24", symbol="diamond"),
        text=[f"{s:.3f}" for s in totals],
        textposition="top center",
        textfont=dict(size=14, color="#fbbf24"),
    ))
    fig.update_layout(
        title=dict(text="Model Score Breakdown", font=dict(size=20, color="#f1f5f9")),
        barmode="group",
        plot_bgcolor=dark, paper_bgcolor=dark,
        font=dict(color="#e2e8f0", size=13),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", range=[0, 1.2], title="Score"),
        height=420,
        margin=dict(t=60, b=20, l=40, r=20),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═════════════════════════════════════════════════════════════════════════════

def on_difficulty_change(label: str) -> pd.DataFrame:
    return _dirty_preview(DIFFICULTY_LABELS.get(label, "easy"))


# ═════════════════════════════════════════════════════════════════════════════
# Gradio Blocks definition
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
body, .gradio-container { background: #0f172a !important; }
.gr-box, .gr-form, .gr-panel { background: #1e293b !important; border-color: #334155 !important; }
h1, h2, h3, label, .gr-markdown { color: #f1f5f9 !important; }
.gr-button-primary { background: #3b82f6 !important; border: none !important;
                     font-weight: 700; font-size: 16px; }
.gr-button-primary:hover { background: #2563eb !important; }
.gr-textbox textarea, .gr-textbox input {
    background: #0f172a !important; color: #e2e8f0 !important;
    border-color: #475569 !important; font-family: monospace; font-size: 13px; }
.gr-dataframe table { background: #1e293b !important; color: #e2e8f0 !important; }
.gr-dataframe th   { background: #334155 !important; color: #93c5fd !important; }
footer { display: none !important; }
"""

_key_hint = (
    "Optional — leave blank if server key is configured (HF Space secret)"
    if (_SERVER_GROQ_KEY or _SERVER_OAI_KEY)
    else "Required for LLM models — get a free Groq key at console.groq.com"
)

with gr.Blocks(css=CSS, title="🧹 Data Cleaning Agent Arena") as gradio_app:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:2.4rem;font-weight:800;color:#f1f5f9;margin:0">
        🧹 Data Cleaning Agent Arena
      </h1>
      <p style="color:#94a3b8;font-size:1.05rem;margin-top:8px">
        Watch AI models compete to clean dirty datasets — scored live, step by step
      </p>
    </div>""")

    # ── Config + preview row ──────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Configuration")
            difficulty_radio = gr.Radio(
                choices=list(DIFFICULTY_LABELS.keys()),
                value=list(DIFFICULTY_LABELS.keys())[0],
                label="Dataset Difficulty",
            )
            model_checkboxes = gr.CheckboxGroup(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys()),
                label="Models to compete",
            )
            api_key_input = gr.Textbox(
                label=f"API Key  ({_key_hint})",
                placeholder="sk-... or gsk_...",
                type="password",
                lines=1,
            )
            run_btn = gr.Button("🚀  Start Arena", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("### 📊 Dirty Data Preview  *(first 10 rows)*")
            preview_table = gr.Dataframe(wrap=True, interactive=False)

    gr.Markdown("---")

    # ── Live log + leaderboard row ────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 📡 Live Action Log")
            log_box = gr.Textbox(
                lines=22, max_lines=22, interactive=False,
                placeholder="Logs stream here as each model runs…",
            )
        with gr.Column(scale=2):
            gr.Markdown("### 🏆 Leaderboard")
            leaderboard_table = gr.Dataframe(interactive=False)

    # ── Chart ─────────────────────────────────────────────────────────────────
    gr.Markdown("### 📈 Score Breakdown")
    chart = gr.Plot()

    # ── Wire events ───────────────────────────────────────────────────────────
    difficulty_radio.change(on_difficulty_change, [difficulty_radio], [preview_table])
    gradio_app.load(on_difficulty_change, [difficulty_radio], [preview_table])
    run_btn.click(
        run_arena,
        inputs=[difficulty_radio, api_key_input, model_checkboxes],
        outputs=[log_box, leaderboard_table, chart],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Standalone local entry point  (python demo.py)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧹 Data Cleaning Agent Arena  —  standalone mode")
    print("UI  →  http://localhost:7861")
    gradio_app.launch(server_port=7861, server_name="0.0.0.0",
                      share=False, show_error=True)
