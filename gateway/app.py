"""
LLM Gateway — FastAPI API Gateway (main entry point).

Orchestrates the full RAG pipeline:
  1. Validate input (guardrails)
  2. Retrieve context from retriever service
  3. Format prompt with template + context + user message
  4. Call vLLM /v1/completions API
  5. Filter output (guardrails)
  6. Redact PII before logging
  7. Log request/response and token usage
  8. Return structured response

Also supports:
  - SSE streaming via POST /chat/stream
  - API key authentication (X-API-Key header)
  - Per-key rate limiting (configurable, default 60 req/min)
  - Prometheus metrics at /metrics

Usage:
    uvicorn gateway.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import PlainTextResponse, StreamingResponse
from jinja2 import Template
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field

# Guardrails imports — relative to project root
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from guardrails.input_validator import validate_input
from guardrails.output_filter import filter_output
from guardrails.pii_redactor import redact_pii, redact_before_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8080")
API_KEYS_SECRET = os.getenv("API_KEYS_FILE", "")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
PROMPT_TEMPLATES_DIR = Path(os.getenv(
    "PROMPT_TEMPLATES_DIR",
    str(Path(__file__).parent / "prompt_templates"),
))
DEFAULT_TEMPLATE = os.getenv("DEFAULT_TEMPLATE", "customer_support")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/Llama-3-8B-Instruct")
LOG_PII_REDACTION = os.getenv("LOG_PII_REDACTION", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
CHAT_LATENCY = Histogram(
    "chat_latency_seconds",
    "End-to-end latency of /chat requests",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
TOKEN_USAGE = Counter(
    "token_usage_total",
    "Total token consumption",
    ["direction"],  # "input" or "output"
)
COST_PER_REQUEST = Histogram(
    "cost_per_request_usd",
    "Estimated cost per request in USD",
    buckets=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
)
GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total",
    "Total requests blocked by guardrails",
    ["stage"],  # "input" or "output"
)
REQUEST_ERRORS = Counter(
    "request_errors_total",
    "Total request errors",
    ["error_type"],
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of in-flight requests",
)
RETRIEVAL_CALLS = Counter(
    "retrieval_calls_total",
    "Total calls to retriever service",
    ["status"],
)

# ---------------------------------------------------------------------------
# Rate limiter (sliding window in-memory)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory sliding window rate limiter keyed by API key."""

    def __init__(self, default_rpm: int = 60):
        self.default_rpm = default_rpm
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, api_key: str, rpm_override: int | None = None) -> bool:
        limit = rpm_override or self.default_rpm
        now = time.time()
        window_start = now - 60.0

        # Prune old entries
        entries = self._requests[api_key]
        self._requests[api_key] = [t for t in entries if t > window_start]

        if len(self._requests[api_key]) >= limit:
            return False

        self._requests[api_key].append(now)
        return True


# ---------------------------------------------------------------------------
# Prompt template loader
# ---------------------------------------------------------------------------

_templates_cache: dict[str, dict] = {}


def load_prompt_template(name: str) -> dict:
    """Load a prompt template YAML by name. Caches results."""
    if name in _templates_cache:
        return _templates_cache[name]

    path = PROMPT_TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        template_data = yaml.safe_load(f)

    _templates_cache[name] = template_data
    return template_data


def render_prompt(template_data: dict, context: str, user_message: str) -> str:
    """Render a Jinja2 prompt template with context and user message."""
    tmpl = Template(template_data.get("template", "{{ user_message }}"))
    return tmpl.render(
        context=context,
        user_message=user_message,
    )


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

_valid_api_keys: set[str] = set()


def _load_api_keys() -> None:
    """Load API keys from file or environment."""
    global _valid_api_keys
    # From file (mounted secret)
    if API_KEYS_SECRET and Path(API_KEYS_SECRET).exists():
        with open(API_KEYS_SECRET, "r") as f:
            data = json.load(f)
            _valid_api_keys = set(data.get("keys", []))
    # From environment variable (comma-separated)
    env_keys = os.getenv("API_KEYS", "")
    if env_keys:
        _valid_api_keys.update(k.strip() for k in env_keys.split(",") if k.strip())

    if _valid_api_keys:
        logger.info("Loaded %d API keys", len(_valid_api_keys))
    else:
        logger.warning("No API keys configured — authentication disabled")


async def verify_api_key(request: Request) -> str:
    """Dependency: extract and validate X-API-Key header."""
    if not _valid_api_keys:
        return "anonymous"

    api_key = request.headers.get("X-API-Key", "")
    if not api_key or api_key not in _valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize HTTP clients and load config at startup."""
    _state["http_client"] = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    _state["rate_limiter"] = RateLimiter(default_rpm=RATE_LIMIT_RPM)
    _load_api_keys()

    logger.info("Gateway started — vLLM: %s, Retriever: %s", VLLM_URL, RETRIEVER_URL)
    yield

    await _state["http_client"].aclose()
    logger.info("Gateway shutdown complete")


app = FastAPI(
    title="LLMOps Gateway",
    description="API Gateway for the RAG-augmented LLM platform",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    conversation_id: str | None = Field(None, description="Optional conversation ID")
    model: str | None = Field(None, description="Model override")
    template: str | None = Field(None, description="Prompt template name")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)


class ChatSource(BaseModel):
    text: str
    score: float
    metadata: dict = {}


class ChatResponse(BaseModel):
    response: str
    sources: list[ChatSource]
    tokens_used: dict  # {"input": N, "output": M, "total": N+M}
    latency_ms: float
    conversation_id: str
    model: str


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def _retrieve_context(
    query: str, top_k: int, client: httpx.AsyncClient
) -> list[dict]:
    """Call the retriever service to get relevant context chunks."""
    try:
        resp = await client.post(
            f"{RETRIEVER_URL}/retrieve",
            json={"query": query, "top_k": top_k, "threshold": 0.5},
        )
        resp.raise_for_status()
        data = resp.json()
        RETRIEVAL_CALLS.labels(status="success").inc()
        return data.get("results", [])
    except httpx.HTTPStatusError as exc:
        RETRIEVAL_CALLS.labels(status="error").inc()
        logger.error("Retriever returned %d: %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        RETRIEVAL_CALLS.labels(status="error").inc()
        logger.error("Retriever call failed: %s", exc)
        return []


async def _call_vllm(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: httpx.AsyncClient,
    stream: bool = False,
) -> dict | AsyncGenerator[str, None]:
    """Call vLLM /v1/chat/completions endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stream:
        return _stream_vllm(payload, client)

    resp = await client.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


async def _stream_vllm(
    payload: dict, client: httpx.AsyncClient
) -> AsyncGenerator[str, None]:
    """Stream SSE events from vLLM."""
    async with client.stream(
        "POST", f"{VLLM_URL}/v1/chat/completions", json=payload
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Full RAG pipeline: validate -> retrieve -> prompt -> generate -> filter."""
    start = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    request_id = str(uuid.uuid4())[:8]

    try:
        # Rate limit
        limiter: RateLimiter = _state["rate_limiter"]
        if not limiter.is_allowed(api_key):
            REQUEST_ERRORS.labels(error_type="rate_limit").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # 1. Validate input
        validation = validate_input(req.message)
        if not validation.is_valid:
            GUARDRAIL_BLOCKS.labels(stage="input").inc()
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {validation.reason}",
            )
        sanitized_message = validation.sanitized_text

        # 2. Retrieve context
        client: httpx.AsyncClient = _state["http_client"]
        context_results = await _retrieve_context(sanitized_message, req.top_k, client)

        context_text = "\n\n---\n\n".join(
            r.get("text", "") for r in context_results
        ) if context_results else "No relevant context found."

        sources = [
            ChatSource(
                text=r.get("text", "")[:200],
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in context_results
        ]

        # 3. Format prompt
        template_name = req.template or DEFAULT_TEMPLATE
        try:
            template_data = load_prompt_template(template_name)
        except FileNotFoundError:
            template_data = load_prompt_template(DEFAULT_TEMPLATE)

        system_prompt = template_data.get("system_prompt", "You are a helpful assistant.")
        rendered_prompt = render_prompt(template_data, context_text, sanitized_message)
        temperature = req.temperature or template_data.get("temperature", 0.3)
        max_tokens = req.max_tokens or template_data.get("max_tokens", 1024)
        model = req.model or DEFAULT_MODEL

        # 4. Call vLLM
        try:
            vllm_response = await _call_vllm(
                prompt=rendered_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
            )
        except httpx.HTTPStatusError as exc:
            REQUEST_ERRORS.labels(error_type="vllm_error").inc()
            logger.error("[%s] vLLM error: %s", request_id, exc)
            raise HTTPException(status_code=502, detail="LLM service error")
        except httpx.TimeoutException:
            REQUEST_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(status_code=504, detail="LLM request timed out")

        # Extract response text and token counts
        choice = vllm_response.get("choices", [{}])[0]
        response_text = choice.get("message", {}).get("content", "")
        usage = vllm_response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # 5. Filter output
        filter_result = filter_output(response_text)
        if not filter_result.is_safe:
            GUARDRAIL_BLOCKS.labels(stage="output").inc()
            response_text = filter_result.filtered_text
        else:
            response_text = filter_result.filtered_text

        # 6. Redact PII before logging
        if LOG_PII_REDACTION:
            log_message = redact_before_logging(sanitized_message)
            log_response = redact_before_logging(response_text)
        else:
            log_message = sanitized_message
            log_response = response_text

        # 7. Record metrics and log
        elapsed = time.perf_counter() - start
        CHAT_LATENCY.observe(elapsed)
        TOKEN_USAGE.labels(direction="input").inc(input_tokens)
        TOKEN_USAGE.labels(direction="output").inc(output_tokens)

        # Estimate cost ($0.15 per 1M input tokens, $0.60 per 1M output tokens)
        estimated_cost = (input_tokens * 0.15 / 1_000_000) + (
            output_tokens * 0.60 / 1_000_000
        )
        COST_PER_REQUEST.observe(estimated_cost)

        logger.info(
            "[%s] api_key=%s model=%s input_tokens=%d output_tokens=%d "
            "latency_ms=%.1f cost=$%.6f message=%s",
            request_id,
            api_key[:8] + "..." if len(api_key) > 8 else api_key,
            model,
            input_tokens,
            output_tokens,
            elapsed * 1000,
            estimated_cost,
            log_message[:80],
        )

        # 8. Return response
        conversation_id = req.conversation_id or str(uuid.uuid4())

        return ChatResponse(
            response=response_text,
            sources=sources,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            latency_ms=round(elapsed * 1000, 2),
            conversation_id=conversation_id,
            model=model,
        )

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_ERRORS.labels(error_type="internal").inc()
        logger.exception("[%s] Unexpected error", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """SSE streaming version of /chat."""
    # Rate limit
    limiter: RateLimiter = _state["rate_limiter"]
    if not limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Validate input
    validation = validate_input(req.message)
    if not validation.is_valid:
        GUARDRAIL_BLOCKS.labels(stage="input").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Input validation failed: {validation.reason}",
        )

    # Retrieve context
    client: httpx.AsyncClient = _state["http_client"]
    context_results = await _retrieve_context(
        validation.sanitized_text, req.top_k, client
    )
    context_text = "\n\n---\n\n".join(
        r.get("text", "") for r in context_results
    ) if context_results else "No relevant context found."

    # Format prompt
    template_name = req.template or DEFAULT_TEMPLATE
    try:
        template_data = load_prompt_template(template_name)
    except FileNotFoundError:
        template_data = load_prompt_template(DEFAULT_TEMPLATE)

    system_prompt = template_data.get("system_prompt", "You are a helpful assistant.")
    rendered_prompt = render_prompt(template_data, context_text, validation.sanitized_text)
    temperature = req.temperature or template_data.get("temperature", 0.3)
    max_tokens = req.max_tokens or template_data.get("max_tokens", 1024)
    model = req.model or DEFAULT_MODEL

    # Stream from vLLM
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            generator = await _call_vllm(
                prompt=rendered_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
                stream=True,
            )
            async for chunk in generator:
                yield chunk
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint for probes."""
    return {"status": "healthy", "service": "llm-gateway"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "gateway.app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
