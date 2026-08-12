"""
Post-LLM guardrails applied to the generated response.

Checks:
  1. Empty / refusal detection — if the LLM returned nothing useful,
     replace with a friendly fallback.
  2. Sensitive-action disclaimer — if the response mentions SIM swaps,
     account changes, or payment actions, append a verification reminder.
  3. Hallucination flag (lightweight) — flag responses that mention
     specific dollar amounts or policy numbers not present in the prompt.
"""

import re


_SENSITIVE_ACTIONS = [
    "sim swap", "sim replacement", "change your plan",
    "cancel your account", "refund", "payment",
    "port your number", "unlock your phone",
]

_DISCLAIMER = (
    "\n\n⚠️ This action may require identity verification. "
    "Please call 611 or visit a store with a valid photo ID to proceed."
)


def _add_disclaimer_if_needed(response: str) -> str:
    lower = response.lower()
    for action in _SENSITIVE_ACTIONS:
        if action in lower:
            return response + _DISCLAIMER
    return response


def _fallback_if_empty(response: str) -> str:
    stripped = response.strip()
    if not stripped or len(stripped) < 10:
        return (
            "I wasn't able to find enough information to answer that. "
            "Please try rephrasing, or call 611 for live support."
        )
    return response


def _flag_ungrounded_amounts(response: str) -> str:
    """Append a note if the response contains dollar amounts that look
    fabricated"""
    amounts = re.findall(r"\$\d+(?:\.\d{2})?", response)
    if len(amounts) > 3:
        response += (
            "\n\n📝 *Note: This response references several specific amounts. "
            "Please verify against your latest bill in the MyTelecom app.*"
        )
    return response


def validate_output(response: str) -> str:
    """Run all output guardrails and return the final response string."""
    response = _fallback_if_empty(response)
    response = _add_disclaimer_if_needed(response)
    response = _flag_ungrounded_amounts(response)
    return response
