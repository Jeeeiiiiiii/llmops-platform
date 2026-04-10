"""
Input Validator — Pre-processing guardrail for user inputs.

Validates incoming text against configurable rules before it reaches
the LLM, including:
  - Maximum length enforcement
  - Prompt injection detection
  - Forbidden topic blocking
  - HTML/script tag sanitization

Usage:
    from guardrails.input_validator import validate_input

    result = validate_input("Hello, help me with my order", config={})
    if not result.is_valid:
        print(f"Blocked: {result.reason}")
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of input validation."""

    is_valid: bool
    reason: str | None = None
    sanitized_text: str = ""
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "max_length": 4096,

    # Prompt injection patterns — regex patterns that indicate attempts to
    # override system instructions or manipulate the LLM.
    "injection_patterns": [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|directions?)",
        r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"(?i)forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|context)",
        r"(?i)you\s+are\s+now\s+(a|an|the)\s+",
        r"(?i)new\s+system\s+prompt\s*:",
        r"(?i)system\s*:\s*you\s+are",
        r"(?i)\bsystem\s+prompt\s+override\b",
        r"(?i)act\s+as\s+(if\s+)?(you\s+have\s+)?no\s+(restrictions?|limits?|rules?|guardrails?)",
        r"(?i)pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|new|unrestricted)",
        r"(?i)bypass\s+(the\s+)?(safety|content|output)\s+(filter|guard|check)",
        r"(?i)jailbreak",
        r"(?i)DAN\s+mode",
        r"(?i)\[INST\]|\[\/INST\]|<<SYS>>|<\|im_start\|>",
        r"(?i)```\s*system\b",
        r"(?i)override\s+(safety|content)\s+(settings?|filters?|policies?)",
    ],

    # Forbidden topics — block requests that reference these subjects
    "forbidden_topics": [
        "how to make a bomb",
        "how to make explosives",
        "how to hack",
        "illegal drugs synthesis",
        "child exploitation",
        "self-harm instructions",
    ],

    # HTML / script sanitization
    "sanitize_html": True,
}


# ---------------------------------------------------------------------------
# Compiled patterns (cached at module level for performance)
# ---------------------------------------------------------------------------

_compiled_injection: list[re.Pattern] | None = None
_compiled_forbidden: list[re.Pattern] | None = None


def _get_injection_patterns(config: dict) -> list[re.Pattern]:
    """Compile and cache injection regex patterns."""
    global _compiled_injection
    patterns = config.get("injection_patterns", DEFAULT_CONFIG["injection_patterns"])
    if _compiled_injection is None or len(_compiled_injection) != len(patterns):
        _compiled_injection = [re.compile(p) for p in patterns]
    return _compiled_injection


def _get_forbidden_patterns(config: dict) -> list[re.Pattern]:
    """Compile and cache forbidden-topic patterns."""
    global _compiled_forbidden
    topics = config.get("forbidden_topics", DEFAULT_CONFIG["forbidden_topics"])
    if _compiled_forbidden is None or len(_compiled_forbidden) != len(topics):
        _compiled_forbidden = [
            re.compile(re.escape(topic), re.IGNORECASE) for topic in topics
        ]
    return _compiled_forbidden


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_length(text: str, config: dict) -> str | None:
    """Return a reason string if text exceeds max length, else None."""
    max_len = config.get("max_length", DEFAULT_CONFIG["max_length"])
    if len(text) > max_len:
        return f"Input exceeds maximum length ({len(text)} > {max_len} characters)"
    return None


def _check_injection(text: str, config: dict) -> str | None:
    """Detect prompt injection patterns."""
    for pattern in _get_injection_patterns(config):
        match = pattern.search(text)
        if match:
            return f"Prompt injection detected: '{match.group()[:60]}...'"
    return None


def _check_forbidden_topics(text: str, config: dict) -> str | None:
    """Check for forbidden topics."""
    for pattern in _get_forbidden_patterns(config):
        match = pattern.search(text)
        if match:
            return f"Forbidden topic detected: '{match.group()[:60]}'"
    return None


def _sanitize_html(text: str) -> str:
    """Strip HTML tags and decode entities. Removes <script> blocks entirely."""
    # Remove <script>...</script> blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <style>...</style> blocks
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_input(text: str, config: dict | None = None) -> ValidationResult:
    """
    Validate user input text against all configured guardrails.

    Parameters
    ----------
    text : str
        Raw user input.
    config : dict, optional
        Override default config. Keys: max_length, injection_patterns,
        forbidden_topics, sanitize_html.

    Returns
    -------
    ValidationResult
        Contains ``is_valid``, ``reason`` (if blocked), ``sanitized_text``,
        and ``flags`` (list of non-blocking warnings).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    flags: list[str] = []

    # Empty input
    if not text or not text.strip():
        return ValidationResult(
            is_valid=False,
            reason="Input is empty",
            sanitized_text="",
            flags=flags,
        )

    # Sanitize HTML first (before other checks)
    sanitized = text
    if cfg.get("sanitize_html", True):
        sanitized = _sanitize_html(text)
        if sanitized != text.strip():
            flags.append("html_sanitized")

    # Check length
    reason = _check_length(sanitized, cfg)
    if reason:
        return ValidationResult(
            is_valid=False,
            reason=reason,
            sanitized_text=sanitized,
            flags=flags,
        )

    # Check prompt injection
    reason = _check_injection(sanitized, cfg)
    if reason:
        logger.warning("Prompt injection blocked: %s", reason)
        return ValidationResult(
            is_valid=False,
            reason=reason,
            sanitized_text=sanitized,
            flags=flags + ["injection_detected"],
        )

    # Check forbidden topics
    reason = _check_forbidden_topics(sanitized, cfg)
    if reason:
        logger.warning("Forbidden topic blocked: %s", reason)
        return ValidationResult(
            is_valid=False,
            reason=reason,
            sanitized_text=sanitized,
            flags=flags + ["forbidden_topic"],
        )

    return ValidationResult(
        is_valid=True,
        reason=None,
        sanitized_text=sanitized,
        flags=flags,
    )
