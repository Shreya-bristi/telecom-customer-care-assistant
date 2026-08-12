"""
Pre-retrieval guardrails applied to the raw customer query.

Three checks run in sequence:
  1. Length gate    — reject excessively long inputs.
  2. PII redaction  — mask phone numbers, emails, account IDs.
  3. Topic filter   — block clearly off-topic queries.

If a check fails, the function raises a `GuardrailError` which the
caller (CLI / Streamlit app) should catch and surface to the user.
"""

import re
from telecom_rag.config import MAX_INPUT_LENGTH


class GuardrailError(ValueError):
    """Raised when input fails a guardrail check."""


# ── PII patterns ─────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(?:ACC|ACCT)[- ]?\d{6,12}\b", re.IGNORECASE), "[ACCOUNT_REDACTED]"),
    (re.compile(r"\b\d{13,19}\b"), "[CARD_REDACTED]"),        # credit-card-length numbers
]

# ── Off-topic keywords (simple blocklist approach) ───────────────────────────
_OFF_TOPIC_SIGNALS = [
    "weather", "stock price", "recipe", "football score",
    "write me a poem", "translate to french", "tell me a joke",
    "who is the president", "play music",
]


def _check_length(query: str) -> None:
    if len(query) > MAX_INPUT_LENGTH:
        raise GuardrailError(
            f"Your message is too long ({len(query)} chars). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )


def _redact_pii(query: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        query = pattern.sub(replacement, query)
    return query


def _check_topic(query: str) -> None:
    lower = query.lower()
    for signal in _OFF_TOPIC_SIGNALS:
        if signal in lower:
            raise GuardrailError(
                "That question doesn't seem related to your telecom service. "
                "I can help with connectivity, billing, SIM, roaming, and device issues."
            )


def validate_input(query: str) -> str:
    """Run all input guardrails and return the sanitised query.

    Raises GuardrailError if the query is rejected.
    """
    _check_length(query)
    _check_topic(query)
    sanitised = _redact_pii(query)
    return sanitised
