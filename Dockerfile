# ==============================================================================
# Base image with Python 3.12 and uv pre-installed
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation and optimize uv performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 1. Install dependencies first (layer caching)
# Copy only dependency definition files so rebuilds are cached if code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ==============================================================================
# Production Runtime Stage
# ==============================================================================
FROM python:3.12-slim-bookworm

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install system runtime dependencies (e.g., curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY app/ /app/app/
COPY data/ /app/data/
COPY pyproject.toml README.md ./

# Ensure data storage directories exist
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/data/doc_store

# Expose FastAPI default port
EXPOSE 8000

# Health check to ensure the API is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI app with Uvicorn
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
