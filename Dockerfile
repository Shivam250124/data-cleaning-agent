FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/     ./app/
COPY data/    ./data/
COPY scripts/ ./scripts/
COPY demo.py  ./demo.py
COPY app.py   ./app.py
COPY inference.py ./inference.py

# Render uses port 10000 by default
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Run FastAPI (which has Gradio mounted at /demo) on Render's port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
