FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/     ./app/
COPY data/    ./data/
COPY scripts/ ./scripts/
COPY demo.py  ./demo.py
COPY inference.py ./inference.py

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# FastAPI serves on 7860; Gradio is mounted at /demo inside the same process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
