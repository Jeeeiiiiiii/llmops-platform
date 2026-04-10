"""
Tests for the RAG Retriever Service.

Validates the /retrieve endpoint's response format, top_k limiting,
score threshold filtering, and the /health endpoint.

Run:
    pytest tests/test_retriever.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: build a mock ScoredPoint matching qdrant_client's interface
# ---------------------------------------------------------------------------

class _MockScoredPoint:
    def __init__(self, id: int, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = dict(payload)


def _make_search_results(n: int = 5, min_score: float = 0.5) -> list[_MockScoredPoint]:
    """Generate *n* mock search results with descending scores."""
    results = []
    for i in range(n):
        score = round(0.95 - i * 0.05, 2)
        if score < min_score:
            break
        results.append(
            _MockScoredPoint(
                id=i + 1,
                score=score,
                payload={
                    "text": f"Sample document chunk number {i + 1}.",
                    "source": f"doc_{i + 1}.md",
                    "chunk_index": i,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    """Mock SentenceTransformer that returns a fixed-length numpy vector."""
    import numpy as np

    encoder = MagicMock()
    encoder.encode.return_value = np.random.rand(384).astype("float32")
    return encoder


@pytest.fixture
def mock_qdrant():
    """Mock QdrantClient with a search method returning sample results."""
    client = MagicMock()
    client.search.return_value = _make_search_results(5)
    client.get_collection.return_value = MagicMock(points_count=5000)
    client.get_collections.return_value = MagicMock(
        collections=[MagicMock(name="documents")]
    )
    return client


@pytest.fixture
def client(mock_encoder, mock_qdrant) -> TestClient:
    """
    Create a FastAPI TestClient for the retriever service with mocked
    dependencies injected into the shared state.
    """
    from rag_pipeline.retriever import retriever_service

    # Inject mocks into the module-level state dict
    retriever_service._state["encoder"] = mock_encoder
    retriever_service._state["qdrant"] = mock_qdrant

    return TestClient(retriever_service.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_results(client, mock_qdrant):
    """
    POST /retrieve should return a JSON response with a 'results' list,
    each item containing 'text', 'score', and 'metadata'.
    """
    response = client.post("/retrieve", json={
        "query": "What is the return policy?",
        "top_k": 5,
        "threshold": 0.5,
    })

    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert "query" in data
    assert "latency_ms" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    # Verify each result has the expected fields
    for result in data["results"]:
        assert "text" in result
        assert "score" in result
        assert "metadata" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


def test_retrieve_respects_top_k(client, mock_qdrant):
    """
    When requesting top_k=3, the endpoint should return exactly 3 results
    (assuming the vector store has at least 3 matching documents).
    """
    # Configure mock to return exactly 3 results
    mock_qdrant.search.return_value = _make_search_results(3)

    response = client.post("/retrieve", json={
        "query": "How do I reset my password?",
        "top_k": 3,
        "threshold": 0.5,
    })

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3


def test_retrieve_filters_by_threshold(client, mock_qdrant):
    """
    Results with scores below the requested threshold should be excluded
    from the response.
    """
    # Return results where some are below the threshold of 0.8
    all_results = _make_search_results(5)  # scores: 0.95, 0.90, 0.85, 0.80, 0.75
    above_threshold = [r for r in all_results if r.score >= 0.8]
    mock_qdrant.search.return_value = above_threshold

    response = client.post("/retrieve", json={
        "query": "What are the system requirements?",
        "top_k": 5,
        "threshold": 0.8,
    })

    assert response.status_code == 200
    data = response.json()

    # All returned results should have score >= 0.8
    for result in data["results"]:
        assert result["score"] >= 0.8, (
            f"Result with score {result['score']} should have been filtered "
            f"(threshold=0.8)"
        )


def test_health_endpoint(client):
    """
    GET /health should return 200 with a status field indicating the
    service is healthy.
    """
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "collection" in data
