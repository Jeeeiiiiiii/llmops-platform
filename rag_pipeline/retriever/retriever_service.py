"""
Retriever Service — FastAPI microservice for semantic similarity search.

Connects to Qdrant and exposes a /retrieve endpoint that accepts a natural-
language query, encodes it via sentence-transformers, and returns the top-k
most relevant document chunks with scores and metadata.

Prometheus metrics are exposed at /metrics for observability.

Usage:
    uvicorn rag_pipeline.retriever.retriever_service:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.7"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Latency of the retrieval operation in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
RETRIEVAL_REQUESTS = Counter(
    "retrieval_requests_total",
    "Total number of retrieval requests",
    ["status"],
)
AVG_RELEVANCE_SCORE = Summary(
    "avg_relevance_score",
    "Average relevance score of returned results",
)
ENCODING_LATENCY = Histogram(
    "query_encoding_latency_seconds",
    "Latency of query embedding generation",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to Qdrant at startup."""
    logger.info("Loading sentence-transformer model: %s", EMBEDDING_MODEL)
    _state["encoder"] = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Connecting to Qdrant at %s", QDRANT_URL)
    _state["qdrant"] = QdrantClient(url=QDRANT_URL, timeout=30)

    # Verify collection exists
    try:
        info = _state["qdrant"].get_collection(COLLECTION_NAME)
        logger.info(
            "Collection '%s' online (%d points)", COLLECTION_NAME, info.points_count
        )
    except Exception as exc:
        logger.error("Collection '%s' not found: %s", COLLECTION_NAME, exc)

    yield

    logger.info("Shutting down retriever service")
    _state.clear()


app = FastAPI(
    title="RAG Retriever Service",
    description="Semantic similarity search over Qdrant vector store",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="Search query")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100, description="Number of results")
    threshold: float = Field(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class RetrieveResult(BaseModel):
    text: str
    score: float
    metadata: dict


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
    query: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Perform semantic similarity search over the document collection."""
    start = time.perf_counter()

    encoder: SentenceTransformer = _state.get("encoder")
    qdrant: QdrantClient = _state.get("qdrant")

    if encoder is None or qdrant is None:
        RETRIEVAL_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Encode query
        encode_start = time.perf_counter()
        query_vector = encoder.encode(
            req.query, normalize_embeddings=True, convert_to_numpy=True
        )
        ENCODING_LATENCY.observe(time.perf_counter() - encode_start)

        # Search Qdrant
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=req.top_k,
            score_threshold=req.threshold,
        )

        # Build response
        results = []
        scores = []
        for hit in search_results:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            results.append(
                RetrieveResult(
                    text=text,
                    score=round(hit.score, 4),
                    metadata=payload,
                )
            )
            scores.append(hit.score)

        # Record metrics
        elapsed = time.perf_counter() - start
        RETRIEVAL_LATENCY.observe(elapsed)
        RETRIEVAL_REQUESTS.labels(status="success").inc()

        if scores:
            AVG_RELEVANCE_SCORE.observe(float(np.mean(scores)))

        return RetrieveResponse(
            results=results,
            query=req.query,
            latency_ms=round(elapsed * 1000, 2),
        )

    except UnexpectedResponse as exc:
        RETRIEVAL_REQUESTS.labels(status="error").inc()
        logger.exception("Qdrant error during retrieval")
        raise HTTPException(status_code=502, detail=f"Vector store error: {exc}")

    except Exception as exc:
        RETRIEVAL_REQUESTS.labels(status="error").inc()
        logger.exception("Unexpected error during retrieval")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Readiness/liveness probe — checks Qdrant connectivity."""
    qdrant: QdrantClient | None = _state.get("qdrant")
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    try:
        qdrant.get_collections()
    except Exception:
        raise HTTPException(status_code=503, detail="Qdrant unreachable")

    return {"status": "healthy", "collection": COLLECTION_NAME}


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
        "rag_pipeline.retriever.retriever_service:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
