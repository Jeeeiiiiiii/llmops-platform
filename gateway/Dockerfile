###############################################################################
# LLM Gateway — Multi-stage Dockerfile
# Python 3.11 slim, non-root, port 8080
###############################################################################

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Runtime OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl tini && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY gateway/ /app/gateway/
COPY guardrails/ /app/guardrails/

# Ensure prompt templates are included
COPY gateway/prompt_templates/ /app/gateway/prompt_templates/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_URL="http://vllm-inference:8000" \
    RETRIEVER_URL="http://retriever:8080" \
    PROMPT_TEMPLATES_DIR="/app/gateway/prompt_templates" \
    LOG_PII_REDACTION="true"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "gateway.app:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", "--timeout-keep-alive", "30", \
     "--access-log"]
