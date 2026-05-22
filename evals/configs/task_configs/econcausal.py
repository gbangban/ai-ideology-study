"""Process-results function for EconCausal tasks.

Used via YAML config:
    process_results: !function econcausal.process_results

The model generates free-text JSON output. This function extracts the
`predicted_sign` field and compares it to the ground truth `answer`.
"""

import re
from typing import Any, Dict, Optional


_SIGN_JSON = re.compile(r'"predicted_sign"\s*:\s*"([^"]+)"', re.IGNORECASE)
# Context-aware: sign must appear after an answer keyword or as a standalone token
# \b doesn't work with + and - (non-word chars), so use lookahead/lookbehind
_SIGN_CONTEXT = re.compile(
    r"(?:sign|answer|prediction|result|conclusion)[:\s]+(?<!\w)(\+|-|None|mixed)(?!\w)",
    re.IGNORECASE,
)
# Standalone sign: avoid matching '-' in hyphenated words or prose dashes.
# For '-' specifically, require it's not surrounded by letters (catches "not-significant").
# '+' is safe (rare in prose). 'None'/'mixed' are word-bounded.
_SIGN_STANDALONE = re.compile(
    r"(?<![a-zA-Z0])(\+|-)(?![a-zA-Z0-])|(?<!\w)(None|mixed)(?!\w)",
    re.IGNORECASE
)


def _extract_sign(text: str) -> Optional[str]:
    """Extract predicted causal sign from model output.

    Strategy:
    1. Try JSON extraction (most reliable).
    2. Try context-aware extraction (sign after answer keyword).
    3. Fallback: take the LAST standalone sign token (model often corrects itself).
    """
    # Try JSON extraction first
    m = _SIGN_JSON.search(text)
    if m:
        return m.group(1)

    # Context-aware: sign after answer keyword — use LAST match (model often
    # corrects itself, e.g. "initial prediction: + ... final answer: -")
    context_matches = list(_SIGN_CONTEXT.finditer(text))
    if context_matches:
        return _normalize(context_matches[-1].group(1))

    # Fallback: take the LAST standalone sign token (model often corrects itself,
    # e.g. "not + but rather -"). Using last match avoids picking up signs in
    # negated or disclaimed contexts earlier in the text.
    # Note: regex has two groups - group(1) for +/-, group(2) for None/mixed.
    matches = list(_SIGN_STANDALONE.finditer(text))
    if matches:
        val = matches[-1].group(1) or matches[-1].group(2)
        if val:
            return _normalize(val)
        return None

    return None


def _normalize(sign: Optional[str]) -> Optional[str]:
    if sign is None:
        return None
    sign = sign.strip()
    if sign.lower() == "none":
        return "None"
    if sign.lower() == "mixed":
        return "mixed"
    return sign


def process_results(doc: Dict[str, Any], results: list) -> Dict[str, float]:
    """Compare predicted sign to ground truth.

    Args:
        doc: Dataset record with 'answer' field (one of "+", "-", "None", "mixed").
        results: List containing the model's generated text as first element.

    Returns:
        Dict with "acc" key (1.0 if correct, 0.0 otherwise).
    """
    generated = results[0] if results else ""
    predicted = _normalize(_extract_sign(generated))
    actual = _normalize(doc.get("answer", ""))

    correct = 1.0 if predicted == actual else 0.0
    return {"acc": correct}
