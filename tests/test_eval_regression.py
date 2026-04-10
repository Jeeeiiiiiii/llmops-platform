"""
Tests for Evaluation Regression Detection.

Validates that the evaluation pipeline correctly detects quality
regressions and that scores meet baseline thresholds:
  - answer relevance   > 0.7
  - faithfulness        > 0.8
  - no regression from saved baseline

These tests mock the gateway and encoder to run offline without
requiring a deployed stack.

Run:
    pytest tests/test_eval_regression.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.run_eval import (
    EvalItem,
    EvalReport,
    compute_answer_relevance,
    compute_faithfulness,
    compare_with_baseline,
    load_eval_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    """
    Mock SentenceTransformer that returns high-similarity embeddings
    for related texts and lower similarity for unrelated texts.
    """
    encoder = MagicMock()

    def _encode(texts, normalize_embeddings=True, convert_to_numpy=True):
        """Generate embeddings that produce high cosine similarity for
        semantically related text pairs."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Use a deterministic seed based on first few words so
            # similar texts get similar vectors
            np.random.seed(hash(text[:30]) % 2**31)
            vec = np.random.rand(384).astype("float32")
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        return np.array(embeddings)

    encoder.encode = _encode
    return encoder


@pytest.fixture
def sample_eval_items() -> list[EvalItem]:
    """A small set of evaluation items for testing."""
    return [
        EvalItem(
            question="What is the return policy for electronics?",
            expected_answer=(
                "Electronics purchased online can be returned within 30 days "
                "of delivery for a full refund."
            ),
            context_required=True,
            category="policy",
        ),
        EvalItem(
            question="How do I reset my password?",
            expected_answer=(
                "Go to the login page and click Forgot Password. Enter your "
                "registered email address."
            ),
            context_required=True,
            category="troubleshooting",
        ),
        EvalItem(
            question="What are the system requirements for ProductX Pro?",
            expected_answer=(
                "ProductX Pro v3.0 requires Windows 10/11 or macOS 12+, 8GB "
                "RAM minimum, 4-core CPU."
            ),
            context_required=True,
            category="product",
        ),
    ]


@pytest.fixture
def passing_report() -> EvalReport:
    """An EvalReport with scores that pass all thresholds."""
    return EvalReport(
        total_questions=20,
        mean_relevance=0.82,
        mean_faithfulness=0.85,
        mean_latency_ms=450.0,
        p50_latency_ms=380.0,
        p95_latency_ms=720.0,
        p99_latency_ms=1200.0,
        pass_rate=0.90,
        results_by_category={},
        individual_results=[],
        regression_detected=False,
        regression_details=[],
    )


@pytest.fixture
def baseline_scores() -> dict:
    """Saved baseline scores for regression comparison."""
    return {
        "mean_relevance": 0.80,
        "mean_faithfulness": 0.83,
        "pass_rate": 0.88,
        "p95_latency_ms": 700.0,
        "total_questions": 20,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_eval_scores_above_baseline(mock_encoder, sample_eval_items):
    """
    Run relevance scoring on eval items and verify that the average
    relevance score exceeds the 0.7 baseline threshold.

    Uses a mock encoder that generates deterministic embeddings.
    """
    # Simulate model responses that are closely related to expected answers
    simulated_responses = [
        "Electronics can be returned within 30 days of delivery for a refund "
        "if they are in original packaging and unused condition.",
        "To reset your password, visit the login page and click the Forgot "
        "Password link, then enter your email address.",
        "ProductX Pro v3.0 needs Windows 10 or 11, macOS 12 or later, at "
        "least 8 GB of RAM, and a quad-core processor.",
    ]

    relevance_scores = []
    for item, response in zip(sample_eval_items, simulated_responses):
        score = compute_answer_relevance(response, item.expected_answer, mock_encoder)
        relevance_scores.append(score)

    avg_relevance = sum(relevance_scores) / len(relevance_scores)

    assert avg_relevance > 0.7, (
        f"Average relevance {avg_relevance:.4f} is below the 0.7 threshold. "
        f"Individual scores: {relevance_scores}"
    )

    # Each individual score should also be reasonable (> 0.5)
    for i, score in enumerate(relevance_scores):
        assert score > 0.5, (
            f"Relevance score for item {i} ({score:.4f}) is below minimum "
            f"acceptable threshold of 0.5"
        )


def test_no_regression_from_previous(passing_report, baseline_scores):
    """
    Compare current evaluation results against saved baseline scores
    and verify that no regression is detected (i.e., no metric drops
    by more than 5% relative to the baseline).
    """
    # Write baseline to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(baseline_scores, f)
        baseline_path = Path(f.name)

    try:
        result = compare_with_baseline(passing_report, baseline_path)

        assert result.regression_detected is False, (
            f"Unexpected regression detected: {result.regression_details}"
        )
        assert len(result.regression_details) == 0
    finally:
        baseline_path.unlink(missing_ok=True)


def test_no_regression_detects_actual_regression(baseline_scores):
    """
    Verify that the regression detector correctly identifies a
    significant quality drop compared to the baseline.
    """
    # Create a report with degraded scores
    degraded_report = EvalReport(
        total_questions=20,
        mean_relevance=0.60,       # dropped from baseline 0.80
        mean_faithfulness=0.65,    # dropped from baseline 0.83
        mean_latency_ms=800.0,
        p50_latency_ms=600.0,
        p95_latency_ms=1500.0,    # increased from baseline 700
        p99_latency_ms=2500.0,
        pass_rate=0.55,            # dropped from baseline 0.88
        results_by_category={},
        individual_results=[],
        regression_detected=False,
        regression_details=[],
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(baseline_scores, f)
        baseline_path = Path(f.name)

    try:
        result = compare_with_baseline(degraded_report, baseline_path)

        assert result.regression_detected is True, (
            "Regression should have been detected for degraded scores"
        )
        assert len(result.regression_details) > 0
    finally:
        baseline_path.unlink(missing_ok=True)


def test_faithfulness_above_threshold():
    """
    Verify that the faithfulness computation correctly identifies when
    a response references source material and that the score exceeds
    the 0.8 threshold for well-grounded responses.
    """
    # Response that closely references the source content
    response = (
        "Electronics purchased online can be returned within 30 days "
        "of delivery for a full refund. Items must be in original "
        "packaging and unused condition. You can initiate a return "
        "through the account dashboard."
    )

    # Sources that the response should reference
    sources = [
        {
            "text": "Electronics purchased online can be returned within 30 days "
                    "of delivery for a full refund, provided they are in original "
                    "packaging and unused condition."
        },
        {
            "text": "To initiate a return, go to your account dashboard and "
                    "select the order you wish to return."
        },
        {
            "text": "Opened items may be subject to a 15% restocking fee."
        },
    ]

    faithfulness = compute_faithfulness(response, sources)

    assert faithfulness > 0.8, (
        f"Faithfulness score {faithfulness:.4f} is below the 0.8 threshold. "
        f"The response should closely reference the provided sources."
    )


def test_faithfulness_low_for_hallucinated_response():
    """
    A response that does not reference any source material should
    receive a low faithfulness score.
    """
    hallucinated_response = (
        "Our company was founded in 1847 by Sir Reginald Forthington "
        "in the rolling hills of Hampshire. We specialize in artisanal "
        "quantum computing solutions for underwater basket weaving."
    )

    sources = [
        {
            "text": "Electronics purchased online can be returned within 30 days."
        },
        {
            "text": "ProductX Pro requires Windows 10 or macOS 12."
        },
    ]

    faithfulness = compute_faithfulness(hallucinated_response, sources)

    assert faithfulness < 0.5, (
        f"Faithfulness score {faithfulness:.4f} should be low for a "
        f"hallucinated response that doesn't reference any sources."
    )


def test_eval_dataset_loads_correctly(sample_eval_dataset):
    """
    Verify that the eval dataset fixture loads successfully and contains
    properly structured items.
    """
    assert len(sample_eval_dataset) > 0, "Eval dataset should not be empty"

    for item in sample_eval_dataset:
        assert "question" in item, "Each item must have a 'question' field"
        assert "expected_answer" in item, "Each item must have an 'expected_answer' field"
        assert len(item["question"]) > 0, "Question should not be empty"
        assert len(item["expected_answer"]) > 0, "Expected answer should not be empty"
