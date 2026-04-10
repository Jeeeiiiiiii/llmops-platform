"""
PII Redactor — Detect and redact personally identifiable information.

Identifies and replaces PII patterns in text:
  - Email addresses       -> [EMAIL_REDACTED]
  - Phone numbers         -> [PHONE_REDACTED]
  - Social Security Nums  -> [SSN_REDACTED]
  - Credit card numbers   -> [CREDIT_CARD_REDACTED]
  - IP addresses          -> [IP_REDACTED]

Supports two redaction modes:
  - redact_before_logging:  Redact PII in logs only (preserve original)
  - redact_before_response: Redact PII in user-facing responses

Usage:
    from guardrails.pii_redactor import redact_pii

    result = redact_pii("Contact me at john@acme.com or 555-123-4567")
    print(result.redacted_text)  # "Contact me at [EMAIL_REDACTED] or [PHONE_REDACTED]"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class PIIEntity(NamedTuple):
    """A single detected PII entity."""

    type: str
    original: str
    position: tuple[int, int]  # (start, end) character offsets


@dataclass
class RedactionResult:
    """Outcome of PII redaction."""

    redacted_text: str
    entities_found: list[dict] = field(default_factory=list)
    pii_detected: bool = False


# ---------------------------------------------------------------------------
# PII patterns — order matters (more specific patterns first)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # Email address
    (
        "EMAIL",
        "[EMAIL_REDACTED]",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    # US Social Security Number (XXX-XX-XXXX or XXXXXXXXX)
    (
        "SSN",
        "[SSN_REDACTED]",
        re.compile(
            r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
        ),
    ),
    # Credit card numbers (13-19 digits, with optional separators)
    # Covers Visa, MasterCard, Amex, Discover
    (
        "CREDIT_CARD",
        "[CREDIT_CARD_REDACTED]",
        re.compile(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
            r"6(?:011|5[0-9]{2})[0-9]{12}|"
            r"(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{1}|6(?:011|5[0-9]{1}))"
            r"[-\s][0-9]{4}[-\s][0-9]{4}[-\s][0-9]{4,7})\b"
        ),
    ),
    # US phone numbers — various formats
    (
        "PHONE",
        "[PHONE_REDACTED]",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[-.\s]?)?"
            r"(?:\(?[2-9]\d{2}\)?[-.\s]?)"
            r"[2-9]\d{2}[-.\s]?"
            r"\d{4}"
            r"(?!\d)"
        ),
    ),
    # IPv4 address
    (
        "IP_ADDRESS",
        "[IP_REDACTED]",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
    ),
    # IPv6 address (simplified — catches common formats)
    (
        "IP_ADDRESS",
        "[IP_REDACTED]",
        re.compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_pii(text: str) -> RedactionResult:
    """
    Scan *text* for PII patterns and replace matches with redaction tokens.

    Parameters
    ----------
    text : str
        Input text to scan.

    Returns
    -------
    RedactionResult
        Contains ``redacted_text``, ``entities_found`` (list of dicts with
        type, original, position), and ``pii_detected`` flag.
    """
    if not text:
        return RedactionResult(redacted_text="", entities_found=[], pii_detected=False)

    entities: list[dict] = []
    # We process patterns in order; to avoid double-matching on overlapping
    # regions, we track which character positions have already been redacted.
    redacted = text
    offset_delta = 0  # track cumulative length change from replacements

    # Collect all matches first, then sort by position to handle overlaps
    all_matches: list[tuple[str, str, re.Match]] = []
    for pii_type, replacement, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            all_matches.append((pii_type, replacement, match))

    # Sort by start position, then by longest match first
    all_matches.sort(key=lambda m: (m[2].start(), -(m[2].end() - m[2].start())))

    # Deduplicate overlapping matches (keep the first / longest)
    covered: set[int] = set()
    unique_matches: list[tuple[str, str, re.Match]] = []
    for pii_type, replacement, match in all_matches:
        span = set(range(match.start(), match.end()))
        if span & covered:
            continue  # overlaps with an already-accepted match
        covered |= span
        unique_matches.append((pii_type, replacement, match))

    # Sort by start position for sequential replacement
    unique_matches.sort(key=lambda m: m[2].start())

    # Perform replacements from end to start to preserve positions
    for pii_type, replacement, match in reversed(unique_matches):
        original = match.group()
        start, end = match.start(), match.end()

        entities.append({
            "type": pii_type,
            "original": original,
            "position": (start, end),
        })

        redacted = redacted[:start] + replacement + redacted[end:]

    # Re-sort entities by position (ascending)
    entities.sort(key=lambda e: e["position"][0])

    if entities:
        logger.info(
            "Redacted %d PII entities: %s",
            len(entities),
            ", ".join(e["type"] for e in entities),
        )

    return RedactionResult(
        redacted_text=redacted,
        entities_found=entities,
        pii_detected=len(entities) > 0,
    )


def redact_before_logging(text: str) -> str:
    """
    Convenience function for log sanitization.
    Returns only the redacted text string.
    """
    return redact_pii(text).redacted_text


def redact_before_response(text: str) -> RedactionResult:
    """
    Convenience alias for full redaction with entity tracking.
    Identical to ``redact_pii()`` but named for clarity at the call site.
    """
    return redact_pii(text)
