"""
HF Spaces entry point — Gradio SDK.
Imports the Gradio blocks from demo.py and launches them.
The app internals (DatasetRegistry, DataCleaningEnv, grader) are called
directly — no separate FastAPI process needed on HF Spaces.
"""
from demo import gradio_app

gradio_app.launch()
