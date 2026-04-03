# College Admission Counselling Environment
# Runs FastAPI (OpenEnv) + Gradio on port 7860
# Architecture: FastAPI is main app, Gradio mounted at /ui

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY server/requirements.txt /tmp/server_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core[core]>=0.2.1" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.24.0" \
        "gradio>=5.0.0,<7.0.0" \
        "openai>=1.0.0" \
        "groq>=0.9.0" \
        "python-dotenv>=1.0.0" \
        "pillow>=10.0.0" \
        "requests>=2.31.0"

# Copy all project files
COPY . /app

ENV PYTHONPATH="/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

# Run the combined FastAPI+Gradio app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
