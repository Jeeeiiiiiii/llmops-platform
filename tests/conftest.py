"""
Shared pytest fixtures for the LLMOps test suite.

Provides mocked clients, sample data, and reusable configuration so
individual test modules remain lean and focused on assertions.

Usage:
    Fixtures are auto-discovered by pytest — just declare the fixture
    name as a test function parameter.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import application code
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixture: mock Qdrant client
# ---------------------------------------------------------------------------

class _MockScoredPoint:
    """Mimics qdrant_client.http.models.ScoredPoint."""

    def __init__(self, id: int, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = dict(payload)  # copy to avoid mutation


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """
    Return a mock QdrantClient pre-loaded with sample vector search
    results.  Calling ``client.search(...)`` returns five scored points
    with realistic payloads and descending scores.
    """
    sample_results = [
        _MockScoredPoint(
            id=1,
            score=0.92,
            payload={
                "text": "Electronics purchased online can be returned within 30 days "
                        "of delivery for a full refund, provided they are in original "
                        "packaging and unused condition.",
                "source": "return_policy.md",
                "chunk_index": 0,
            },
        ),
        _MockScoredPoint(
            id=2,
            score=0.87,
            payload={
                "text": "To reset your password, go to the login page and click "
                        "'Forgot Password'. Enter your registered email address.",
                "source": "account_faq.md",
                "chunk_index": 3,
            },
        ),
        _MockScoredPoint(
            id=3,
            score=0.81,
            payload={
                "text": "ProductX Pro v3.0 requires Windows 10/11 or macOS 12+, "
                        "8GB RAM minimum (16GB recommended), 4-core CPU.",
                "source": "system_requirements.md",
                "chunk_index": 1,
            },
        ),
        _MockScoredPoint(
            id=4,
            score=0.74,
            payload={
                "text": "Yes, you can upgrade at any time. The price difference is "
                        "prorated for the remaining days in your billing cycle.",
                "source": "billing_faq.md",
                "chunk_index": 2,
            },
        ),
        _MockScoredPoint(
            id=5,
            score=0.68,
            payload={
                "text": "We accept Visa, MasterCard, American Express, PayPal, "
                        "wire transfers for annual plans.",
                "source": "payment_methods.md",
                "chunk_index": 0,
            },
        ),
    ]

    client = MagicMock()
    client.search.return_value = sample_results
    client.get_collection.return_value = MagicMock(points_count=5000)
    client.get_collections.return_value = MagicMock(
        collections=[MagicMock(name="documents")]
    )
    return client


# ---------------------------------------------------------------------------
# Fixture: mock vLLM client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vllm_client() -> MagicMock:
    """
    Return a mock HTTP client that simulates vLLM's
    ``/v1/chat/completions`` response.
    """
    mock_response = {
        "id": "cmpl-mock-001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "meta-llama/Llama-3-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Based on the provided context, electronics purchased online "
                        "can be returned within 30 days of delivery for a full refund. "
                        "The items must be in their original packaging and unused "
                        "condition. Please note that opened items may be subject to "
                        "a 15% restocking fee."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 250,
            "completion_tokens": 85,
            "total_tokens": 335,
        },
    }

    client = MagicMock()
    # Synchronous .json() for use in tests that don't need async
    response_mock = MagicMock()
    response_mock.json.return_value = mock_response
    response_mock.status_code = 200
    response_mock.raise_for_status = MagicMock()

    client.post.return_value = response_mock

    # Also support async usage
    async_response = AsyncMock()
    async_response.json.return_value = mock_response
    async_response.status_code = 200
    async_response.raise_for_status = MagicMock()

    client.post = AsyncMock(return_value=async_response)

    return client


# ---------------------------------------------------------------------------
# Fixture: sample documents
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """
    A list of sample document chunks with metadata, representative of
    ingested knowledge base content.
    """
    return [
        {
            "text": "Electronics purchased online can be returned within 30 days "
                    "of delivery for a full refund, provided they are in original "
                    "packaging and unused condition. Opened items may be subject "
                    "to a 15% restocking fee.",
            "metadata": {
                "source": "return_policy.md",
                "chunk_index": 0,
                "doc_id": "doc-001",
                "category": "policy",
            },
        },
        {
            "text": "To reset your password, go to the login page and click "
                    "'Forgot Password'. Enter your registered email address, and "
                    "you will receive a password reset link within 5 minutes.",
            "metadata": {
                "source": "account_faq.md",
                "chunk_index": 3,
                "doc_id": "doc-002",
                "category": "troubleshooting",
            },
        },
        {
            "text": "ProductX Pro v3.0 requires Windows 10/11 or macOS 12+, "
                    "8GB RAM minimum (16GB recommended), 4-core CPU, 2GB free "
                    "disk space, and an internet connection for activation.",
            "metadata": {
                "source": "system_requirements.md",
                "chunk_index": 1,
                "doc_id": "doc-003",
                "category": "product",
            },
        },
        {
            "text": "Yes, you can upgrade from the Basic plan to the Enterprise "
                    "plan at any time. The price difference is prorated for the "
                    "remaining days in your current billing cycle.",
            "metadata": {
                "source": "billing_faq.md",
                "chunk_index": 2,
                "doc_id": "doc-004",
                "category": "policy",
            },
        },
        {
            "text": "A 429 error means you have exceeded the rate limit. The "
                    "Basic plan allows 60 requests per minute, Standard allows "
                    "300, and Enterprise allows 1000.",
            "metadata": {
                "source": "api_troubleshooting.md",
                "chunk_index": 0,
                "doc_id": "doc-005",
                "category": "troubleshooting",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Fixture: sample evaluation dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_eval_dataset() -> list[dict[str, Any]]:
    """
    Load the evaluation dataset from the JSONL file, returning a list of
    dictionaries. Falls back to a minimal inline dataset if the file is
    not available (e.g., running tests in an isolated environment).
    """
    dataset_path = _PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl"
    items: list[dict[str, Any]] = []

    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        # Fallback minimal dataset for isolated test runs
        items = [
            {
                "question": "What is the return policy?",
                "expected_answer": "Electronics can be returned within 30 days.",
                "context_required": True,
                "category": "policy",
            },
            {
                "question": "How do I reset my password?",
                "expected_answer": "Go to the login page and click Forgot Password.",
                "context_required": True,
                "category": "troubleshooting",
            },
        ]

    return items


# ---------------------------------------------------------------------------
# Fixture: guardrail configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def guardrail_config() -> dict[str, Any]:
    """
    Default guardrail configuration for testing. Matches the production
    defaults in guardrails/input_validator.py and guardrails/output_filter.py
    but can be overridden per-test.
    """
    return {
        # Input validator settings
        "max_length": 4096,
        "injection_patterns": [
            r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|directions?)",
            r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
            r"(?i)forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|context)",
            r"(?i)you\s+are\s+now\s+(a|an|the)\s+",
            r"(?i)new\s+system\s+prompt\s*:",
            r"(?i)system\s*:\s*you\s+are",
            r"(?i)jailbreak",
            r"(?i)DAN\s+mode",
        ],
        "forbidden_topics": [
            "how to make a bomb",
            "how to make explosives",
            "how to hack",
        ],
        "sanitize_html": True,
        # Output filter settings
        "max_response_length": 16384,
        "toxic_patterns": [
            r"(?i)\b(kill\s+yourself|kys)\b",
            r"(?i)\bcommit\s+suicide\b",
            r"(?i)\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)\b",
        ],
        "redact_profanity": True,
        "hallucination_threshold": 0.5,
    }
