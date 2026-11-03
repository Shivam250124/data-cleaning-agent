"""
Render entry point.
Runs Gradio Arena UI directly on port 10000.
The app internals (DatasetRegistry, DataCleaningEnv) are imported directly.
"""
import os
from demo import gradio_app

# Render assigns PORT env var; default 10000
port = int(os.environ.get("PORT", 10000))

gradio_app.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    show_error=True,
)
