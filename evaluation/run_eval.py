"""
Automated Evaluation — Run golden Q&A evaluation against the LLM Gateway.

Loads an evaluation dataset (JSONL), sends each question to the /chat
endpoint, and computes metrics:
  - answer_relevance: cosine similarity between response and expected answer
  - faithfulness: whether the response references retrieved context
  - latency_ms: end-to-end response time

Aggregates results and optionally compares against a baseline to detect
regressions.

Usage:
    python run_eval.py \
        --dataset evaluation/eval_dataset.jsonl \
        --gateway-url http://localhost:8080 \
        --api-key <key> \
        --baseline evaluation/baseline.json \
        --output evaluation/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "all-MiniLM-L6-v2"
REGRESSION_THRESHOLD = 0.05  # 5% drop triggers failure


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    question: str
    expected_answer: str
    context_required: bool = True
    category: str = "general"


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    actual_response: str
    category: str
    answer_relevance: float
    faithfulness: float
    latency_ms: float
    sources_count: int
    tokens_used: int
    passed: bool


@dataclass
class EvalReport:
    total_questions: int
    mean_relevance: float
    mean_faithfulness: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    pass_rate: float
    results_by_category: dict = field(default_factory=dict)
    individual_results: list[dict] = field(default_factory=list)
    regression_detected: bool = False
    regression_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path) -> list[EvalItem]:
    """Load evaluation items from a JSONL file."""
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(EvalItem(
                    question=data["question"],
                    expected_answer=data["expected_answer"],
                    context_required=data.get("context_required", True),
                    category=data.get("category", "general"),
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping line %d: %s", line_no, exc)
    logger.info("Loaded %d evaluation items from %s", len(items), path)
    return items


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_answer_relevance(
    response: str, expected: str, encoder: SentenceTransformer
) -> float:
    """Cosine similarity between response and expected answer embeddings."""
    if not response.strip() or not expected.strip():
        return 0.0

    embeddings = encoder.encode(
        [response, expected],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, similarity))


def compute_faithfulness(response: str, sources: list[dict]) -> float:
    """
    Estimate faithfulness by checking how many source snippets are
    referenced in the response.

    Returns a ratio between 0 and 1.
    """
    if not sources:
        return 0.0

    referenced = 0
    response_lower = response.lower()

    for source in sources:
        text = source.get("text", "")
        if not text:
            continue
        # Check if key phrases from the source appear in the response
        words = text.lower().split()
        # Use trigrams for more robust matching
        trigrams = [
            " ".join(words[i : i + 3]) for i in range(len(words) - 2)
        ]
        for trigram in trigrams:
            if trigram in response_lower:
                referenced += 1
                break

    return referenced / len(sources)


# ---------------------------------------------------------------------------
# Gateway interaction
# ---------------------------------------------------------------------------

def call_gateway(
    question: str,
    gateway_url: str,
    api_key: str,
    timeout: float = 120.0,
) -> dict:
    """Send a question to the /chat endpoint and return the response."""
    headers = {"X-API-Key": api_key} if api_key else {}

    start = time.perf_counter()
    resp = httpx.post(
        f"{gateway_url}/chat",
        json={"message": question, "top_k": 5},
        headers=headers,
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()
    data["_latency_ms"] = latency_ms
    return data


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset: list[EvalItem],
    gateway_url: str,
    api_key: str = "",
    encoder_model: str = DEFAULT_MODEL,
) -> EvalReport:
    """Run the full evaluation suite and return an EvalReport."""
    logger.info("Loading encoder model: %s", encoder_model)
    encoder = SentenceTransformer(encoder_model)

    results: list[EvalResult] = []

    for i, item in enumerate(dataset):
        logger.info(
            "[%d/%d] Evaluating: %s", i + 1, len(dataset), item.question[:60]
        )

        try:
            response = call_gateway(item.question, gateway_url, api_key)
            actual = response.get("response", "")
            sources = response.get("sources", [])
            tokens = response.get("tokens_used", {}).get("total", 0)
            latency = response.get("_latency_ms", 0.0)

            relevance = compute_answer_relevance(actual, item.expected_answer, encoder)
            faithfulness = compute_faithfulness(
                actual, [{"text": s.get("text", "")} for s in sources]
            )

            result = EvalResult(
                question=item.question,
                expected_answer=item.expected_answer,
                actual_response=actual,
                category=item.category,
                answer_relevance=round(relevance, 4),
                faithfulness=round(faithfulness, 4),
                latency_ms=round(latency, 2),
                sources_count=len(sources),
                tokens_used=tokens,
                passed=relevance >= 0.5,
            )
        except Exception as exc:
            logger.error("Failed to evaluate question %d: %s", i + 1, exc)
            result = EvalResult(
                question=item.question,
                expected_answer=item.expected_answer,
                actual_response=f"ERROR: {exc}",
                category=item.category,
                answer_relevance=0.0,
                faithfulness=0.0,
                latency_ms=0.0,
                sources_count=0,
                tokens_used=0,
                passed=False,
            )

        results.append(result)

    # Aggregate metrics
    relevances = [r.answer_relevance for r in results]
    faithfulness_scores = [r.faithfulness for r in results]
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    pass_count = sum(1 for r in results if r.passed)

    # Per-category breakdown
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_stats = {}
    for cat, cat_results in categories.items():
        cat_relevances = [r.answer_relevance for r in cat_results]
        cat_latencies = [r.latency_ms for r in cat_results if r.latency_ms > 0]
        category_stats[cat] = {
            "count": len(cat_results),
            "mean_relevance": round(statistics.mean(cat_relevances), 4) if cat_relevances else 0,
            "mean_latency_ms": round(statistics.mean(cat_latencies), 2) if cat_latencies else 0,
            "pass_rate": round(sum(1 for r in cat_results if r.passed) / len(cat_results), 4),
        }

    sorted_latencies = sorted(latencies) if latencies else [0.0]

    report = EvalReport(
        total_questions=len(results),
        mean_relevance=round(statistics.mean(relevances), 4) if relevances else 0,
        mean_faithfulness=round(statistics.mean(faithfulness_scores), 4) if faithfulness_scores else 0,
        mean_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0,
        p50_latency_ms=round(sorted_latencies[len(sorted_latencies) // 2], 2),
        p95_latency_ms=round(sorted_latencies[int(len(sorted_latencies) * 0.95)], 2),
        p99_latency_ms=round(sorted_latencies[int(len(sorted_latencies) * 0.99)], 2),
        pass_rate=round(pass_count / len(results), 4) if results else 0,
        results_by_category=category_stats,
        individual_results=[asdict(r) for r in results],
    )

    return report


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_with_baseline(report: EvalReport, baseline_path: Path) -> EvalReport:
    """Compare current results against a saved baseline. Flag regressions."""
    if not baseline_path.exists():
        logger.info("No baseline found at %s — skipping comparison", baseline_path)
        return report

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    checks = [
        ("mean_relevance", report.mean_relevance, baseline.get("mean_relevance", 0)),
        ("mean_faithfulness", report.mean_faithfulness, baseline.get("mean_faithfulness", 0)),
        ("pass_rate", report.pass_rate, baseline.get("pass_rate", 0)),
    ]

    regressions = []
    for metric_name, current, base in checks:
        if base > 0 and (base - current) / base > REGRESSION_THRESHOLD:
            drop_pct = ((base - current) / base) * 100
            msg = (
                f"{metric_name}: {current:.4f} vs baseline {base:.4f} "
                f"({drop_pct:.1f}% drop, threshold={REGRESSION_THRESHOLD*100}%)"
            )
            regressions.append(msg)
            logger.warning("REGRESSION: %s", msg)

    # Check latency increase
    base_p95 = baseline.get("p95_latency_ms", 0)
    if base_p95 > 0 and (report.p95_latency_ms - base_p95) / base_p95 > REGRESSION_THRESHOLD:
        increase_pct = ((report.p95_latency_ms - base_p95) / base_p95) * 100
        msg = (
            f"p95_latency_ms: {report.p95_latency_ms:.1f}ms vs baseline "
            f"{base_p95:.1f}ms ({increase_pct:.1f}% increase)"
        )
        regressions.append(msg)

    report.regression_detected = len(regressions) > 0
    report.regression_details = regressions
    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(report: EvalReport) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("  LLMOps Evaluation Report")
    print("=" * 70)
    print(f"  Total questions:     {report.total_questions}")
    print(f"  Pass rate:           {report.pass_rate:.1%}")
    print(f"  Mean relevance:      {report.mean_relevance:.4f}")
    print(f"  Mean faithfulness:   {report.mean_faithfulness:.4f}")
    print(f"  Mean latency:        {report.mean_latency_ms:.1f} ms")
    print(f"  P50 latency:         {report.p50_latency_ms:.1f} ms")
    print(f"  P95 latency:         {report.p95_latency_ms:.1f} ms")
    print(f"  P99 latency:         {report.p99_latency_ms:.1f} ms")
    print("-" * 70)

    if report.results_by_category:
        print("  Category breakdown:")
        for cat, stats in report.results_by_category.items():
            print(
                f"    {cat:20s}  n={stats['count']:3d}  "
                f"rel={stats['mean_relevance']:.3f}  "
                f"lat={stats['mean_latency_ms']:.0f}ms  "
                f"pass={stats['pass_rate']:.1%}"
            )
        print("-" * 70)

    if report.regression_detected:
        print("  *** REGRESSION DETECTED ***")
        for detail in report.regression_details:
            print(f"    - {detail}")
    else:
        print("  No regressions detected.")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLMOps evaluation suite against the gateway."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to eval_dataset.jsonl",
    )
    parser.add_argument(
        "--gateway-url", type=str, default="http://localhost:8080",
        help="Gateway base URL",
    )
    parser.add_argument(
        "--api-key", type=str, default="",
        help="API key for gateway authentication",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to baseline JSON for regression comparison",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/eval_report.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save current results as the new baseline",
    )
    parser.add_argument(
        "--encoder-model", type=str, default=DEFAULT_MODEL,
        help=f"Sentence-transformer model for relevance scoring (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load dataset
    dataset = load_eval_dataset(args.dataset)
    if not dataset:
        logger.error("No evaluation items loaded. Exiting.")
        sys.exit(1)

    # Run evaluation
    report = run_evaluation(
        dataset=dataset,
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        encoder_model=args.encoder_model,
    )

    # Compare with baseline
    if args.baseline:
        report = compare_with_baseline(report, args.baseline)

    # Print summary
    print_summary(report)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", args.output)

    # Optionally save as new baseline
    if args.save_baseline:
        baseline_path = args.baseline or Path("evaluation/baseline.json")
        baseline_data = {
            "mean_relevance": report.mean_relevance,
            "mean_faithfulness": report.mean_faithfulness,
            "pass_rate": report.pass_rate,
            "p95_latency_ms": report.p95_latency_ms,
            "total_questions": report.total_questions,
        }
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        logger.info("Baseline saved to %s", baseline_path)

    # Exit with failure if regression detected
    if report.regression_detected:
        logger.error("Evaluation FAILED — regression detected")
        sys.exit(1)

    logger.info("Evaluation PASSED")


if __name__ == "__main__":
    main()
