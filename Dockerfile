FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/      ./app/
COPY data/     ./data/
COPY scripts/  ./scripts/
COPY demo.py   ./demo.py
COPY app.py    ./app.py
COPY inference.py ./inference.py

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:10000/ || exit 1

# Run Gradio directly — it serves the full Arena UI on port 10000
CMD ["python", "app.py"]
