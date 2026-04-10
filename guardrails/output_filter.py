"""
Output Filter — Post-processing guardrail for LLM responses.

Inspects generated text for:
  - Toxic / harmful content keywords
  - Response length limits
  - Hallucination indicators (hedging phrases at high confidence)

Usage:
    from guardrails.output_filter import filter_output

    result = filter_output(response_text, config={})
    if not result.is_safe:
        print(f"Flags: {result.flags}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of output filtering."""

    is_safe: bool
    filtered_text: str
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # Maximum response length in characters
    "max_response_length": 16384,

    # Toxic / harmful content patterns — these trigger a block
    "toxic_patterns": [
        r"(?i)\b(kill\s+yourself|kys)\b",
        r"(?i)\bcommit\s+suicide\b",
        r"(?i)\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)\b",
        r"(?i)\bchild\s+(porn|exploitation|abuse)\b",
        r"(?i)\bracial\s+slur\b",
        r"(?i)\bhate\s+speech\b",
        r"(?i)\bterrorist\s+(attack|plot|plan)\b",
    ],

    # Patterns to redact (replace with [REDACTED]) rather than block entirely
    "redact_patterns": [
        r"(?i)\b(fuck|shit|damn|ass|bitch)\b",
    ],

    # Hallucination indicators — hedging phrases that may indicate the model
    # is confabulating. These raise a warning flag but do not block.
    "hallucination_indicators": [
        r"(?i)I\s+(think|believe|guess|suppose)\s+(that\s+)?(the|this|it)",
        r"(?i)I\'?m\s+not\s+(entirely\s+)?sure\s+(but|however|if)",
        r"(?i)to\s+the\s+best\s+of\s+my\s+knowledge",
        r"(?i)I\s+don\'?t\s+have\s+(access\s+to|information\s+about|data\s+on)",
        r"(?i)as\s+of\s+my\s+(last\s+)?(training|knowledge)\s+(cutoff|update)",
        r"(?i)I\s+cannot\s+verify",
        r"(?i)this\s+(may|might|could)\s+not\s+be\s+(accurate|correct|up.to.date)",
    ],

    # Threshold: if more than this fraction of sentences contain hedging,
    # flag as potential hallucination
    "hallucination_threshold": 0.5,

    # Enable profanity redaction (vs blocking)
    "redact_profanity": True,
}


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_compiled_toxic: list[re.Pattern] | None = None
_compiled_redact: list[re.Pattern] | None = None
_compiled_hallucination: list[re.Pattern] | None = None


def _compile_patterns(
    patterns: list[str], cache: list[re.Pattern] | None
) -> list[re.Pattern]:
    if cache is not None and len(cache) == len(patterns):
        return cache
    return [re.compile(p) for p in patterns]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_toxic(text: str, config: dict) -> tuple[bool, str | None]:
    """Return (is_toxic, matched_text)."""
    global _compiled_toxic
    patterns = config.get("toxic_patterns", DEFAULT_CONFIG["toxic_patterns"])
    _compiled_toxic = _compile_patterns(patterns, _compiled_toxic)

    for pattern in _compiled_toxic:
        match = pattern.search(text)
        if match:
            return True, match.group()
    return False, None


def _redact_profanity(text: str, config: dict) -> tuple[str, bool]:
    """Replace profanity with [REDACTED]. Returns (text, was_modified)."""
    global _compiled_redact
    patterns = config.get("redact_patterns", DEFAULT_CONFIG["redact_patterns"])
    _compiled_redact = _compile_patterns(patterns, _compiled_redact)

    modified = False
    for pattern in _compiled_redact:
        new_text = pattern.sub("[REDACTED]", text)
        if new_text != text:
            modified = True
            text = new_text
    return text, modified


def _check_length(text: str, config: dict) -> bool:
    """Return True if text exceeds max response length."""
    max_len = config.get("max_response_length", DEFAULT_CONFIG["max_response_length"])
    return len(text) > max_len


def _check_hallucination(text: str, config: dict) -> tuple[bool, float]:
    """
    Detect hallucination indicators.
    Returns (has_hallucination_risk, indicator_ratio).
    """
    global _compiled_hallucination
    patterns = config.get(
        "hallucination_indicators", DEFAULT_CONFIG["hallucination_indicators"]
    )
    _compiled_hallucination = _compile_patterns(patterns, _compiled_hallucination)
    threshold = config.get(
        "hallucination_threshold", DEFAULT_CONFIG["hallucination_threshold"]
    )

    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return False, 0.0

    hedging_count = 0
    for sentence in sentences:
        for pattern in _compiled_hallucination:
            if pattern.search(sentence):
                hedging_count += 1
                break

    ratio = hedging_count / len(sentences)
    return ratio >= threshold, ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_output(text: str, config: dict | None = None) -> FilterResult:
    """
    Filter LLM output text through safety guardrails.

    Parameters
    ----------
    text : str
        Raw LLM response text.
    config : dict, optional
        Override default configuration.

    Returns
    -------
    FilterResult
        Contains ``is_safe``, ``filtered_text``, and ``flags``.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    flags: list[str] = []
    filtered = text

    # Empty output
    if not text or not text.strip():
        return FilterResult(is_safe=True, filtered_text="", flags=["empty_response"])

    # Check for toxic content — hard block
    is_toxic, matched = _check_toxic(filtered, cfg)
    if is_toxic:
        logger.warning("Toxic content detected: '%s'", matched[:40])
        flags.append("toxic_content")
        return FilterResult(
            is_safe=False,
            filtered_text="I'm sorry, but I cannot provide that response.",
            flags=flags,
        )

    # Redact profanity if enabled
    if cfg.get("redact_profanity", True):
        filtered, was_redacted = _redact_profanity(filtered, cfg)
        if was_redacted:
            flags.append("profanity_redacted")

    # Check response length
    if _check_length(filtered, cfg):
        max_len = cfg.get("max_response_length", DEFAULT_CONFIG["max_response_length"])
        filtered = filtered[:max_len] + "\n\n[Response truncated due to length limit]"
        flags.append("response_truncated")
        logger.info("Response truncated to %d characters", max_len)

    # Check for hallucination indicators
    has_hallucination, ratio = _check_hallucination(filtered, cfg)
    if has_hallucination:
        flags.append(f"hallucination_risk(ratio={ratio:.2f})")
        logger.info("Hallucination risk detected: %.1f%% hedging ratio", ratio * 100)

    return FilterResult(
        is_safe=True,
        filtered_text=filtered,
        flags=flags,
    )
