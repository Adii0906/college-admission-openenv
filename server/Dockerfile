# Server-only Docker (for local testing without Gradio)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/req.txt

COPY . /app/env
ENV PYTHONPATH="/app/env:$PYTHONPATH"
WORKDIR /app/env

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
