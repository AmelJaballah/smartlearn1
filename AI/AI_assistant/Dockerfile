# AI Assistant (RAG Chatbot) API - Production Dockerfile
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
COPY rag.py .
COPY embeddings.py .
COPY chroma_db ./chroma_db

# Create non-root user
RUN useradd -m -u 1000 appuser

# Fix ownership for persisted data + runtime
RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    RAG_PORT=5002 \
    OLLAMA_BASE_URL=http://localhost:11434 \
    RAG_RETRIEVER_K=3 \
    HF_HUB_OFFLINE=0

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["python", "api_server.py"]
