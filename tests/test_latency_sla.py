"""
Tests for Latency SLA Compliance.

Validates that critical path operations complete within their SLA
budgets:
  - Retrieval (Qdrant search):    < 100ms
  - Guardrail input validation:   < 10ms
  - PII redaction:                < 5ms

These tests use mocked backends to isolate the code-level performance
from network latency. They verify that application-layer processing
does not add unacceptable overhead.

Run:
    pytest tests/test_latency_sla.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from guardrails.input_validator import validate_input
from guardrails.pii_redactor import redact_pii


# ---------------------------------------------------------------------------
# Helper: mock Qdrant scored point
# ---------------------------------------------------------------------------

class _MockScoredPoint:
    def __init__(self, id: int, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = dict(payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieval_latency_under_100ms(mock_qdrant_client):
    """
    The retrieval operation (encode query + Qdrant search + result
    formatting) should complete in under 100ms when using a mock
    Qdrant client.

    This validates that the application code path (excluding actual
    network I/O and embedding computation) adds minimal overhead.
    """
    # Set up mock encoder with a fast, pre-computed vector
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.random.rand(384).astype("float32")

    # Warm up (first call may incur import overhead)
    mock_encoder.encode("warm up", normalize_embeddings=True, convert_to_numpy=True)

    # Measure the retrieval operation
    iterations = 50
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Simulate the retrieval pipeline
        query_vector = mock_encoder.encode(
            "What is the return policy?",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        results = mock_qdrant_client.search(
            collection_name="documents",
            query_vector=query_vector.tolist(),
            limit=5,
            score_threshold=0.5,
        )
        # Format results
        formatted = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            formatted.append({
                "text": text,
                "score": round(hit.score, 4),
                "metadata": payload,
            })

        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(len(times) * 0.95)]

    assert avg_ms < 100, (
        f"Average retrieval latency {avg_ms:.2f}ms exceeds 100ms SLA"
    )
    assert p95_ms < 100, (
        f"P95 retrieval latency {p95_ms:.2f}ms exceeds 100ms SLA"
    )


def test_guardrail_processing_under_10ms():
    """
    Input validation (injection detection, length check, HTML sanitization)
    should complete in under 10ms for a typical user message.
    """
    test_messages = [
        "What is the return policy for electronics?",
        "How do I reset my password? I forgot it yesterday.",
        "Can you help me understand the pricing differences between plans?",
        "I'm having trouble with my API integration returning 429 errors.",
        "What compliance certifications does your platform have?",
    ]

    # Warm up
    validate_input(test_messages[0])

    iterations = 100
    times = []

    for i in range(iterations):
        message = test_messages[i % len(test_messages)]
        start = time.perf_counter()
        result = validate_input(message)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        assert result.is_valid is True  # Sanity check

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(len(times) * 0.95)]

    assert avg_ms < 10, (
        f"Average guardrail processing {avg_ms:.2f}ms exceeds 10ms SLA"
    )
    assert p95_ms < 10, (
        f"P95 guardrail processing {p95_ms:.2f}ms exceeds 10ms SLA"
    )


def test_pii_redaction_under_5ms():
    """
    PII redaction (email, phone, SSN, credit card, IP patterns) should
    complete in under 5ms for a typical response containing mixed PII.
    """
    test_texts = [
        # Text with multiple PII types
        "Contact john@example.com or call (555) 123-4567. "
        "SSN: 123-45-6789. Card: 4111-1111-1111-1111.",
        # Text with no PII
        "The weather in Seattle is usually rainy in November. "
        "ProductX Pro requires 8GB RAM and a 4-core CPU.",
        # Text with single PII
        "Please reach out to support@company.com for assistance.",
        # Text with phone and email
        "My number is (212) 555-0198 and email is user@test.org.",
    ]

    # Warm up (first call compiles regex patterns)
    redact_pii(test_texts[0])

    iterations = 100
    times = []

    for i in range(iterations):
        text = test_texts[i % len(test_texts)]
        start = time.perf_counter()
        result = redact_pii(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(len(times) * 0.95)]

    assert avg_ms < 5, (
        f"Average PII redaction {avg_ms:.2f}ms exceeds 5ms SLA"
    )
    assert p95_ms < 5, (
        f"P95 PII redaction {p95_ms:.2f}ms exceeds 5ms SLA"
    )
