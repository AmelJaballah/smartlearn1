
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install CPU-only PyTorch stack first to avoid pulling huge GPU/CUDA wheels
# sentence-transformers may pull torchvision; preinstall the CPU wheel from the PyTorch index.
RUN pip install --no-cache-dir --timeout=1000 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchvision==0.17.2

COPY requirements.txt .
# Install other dependencies (without --user to install system-wide)
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

COPY api_server.py .
COPY exercice_generator_rag.py .
COPY index.py .
COPY data.py .
COPY subjects_examples.json .
COPY chroma_exercises ./chroma_exercises

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create runtime directories (model cache) and fix ownership
RUN mkdir -p ./models && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    EXERCISE_PORT=5001 \
    OLLAMA_BASE_URL=http://localhost:11434 \
    MODEL_PATH=/app/models

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["python", "api_server.py"]
