"""Process-results function for EconCausal tasks.

Used via YAML config:
    process_results: !function econcausal.process_results

The model generates free-text JSON output. This function extracts the
`predicted_sign` field and compares it to the ground truth `answer`.
"""

import re
from typing import Any, Dict, Optional


_SIGN_JSON = re.compile(r'"predicted_sign"\s*:\s*"([^"]+)"', re.IGNORECASE)
# \b doesn't work with + and - (non-word chars), so use lookahead/lookbehind
_SIGN_STANDALONE = re.compile(
    r"(?<!\w)(\+|-|None|mixed)(?!\w)", re.IGNORECASE
)


def _extract_sign(text: str) -> Optional[str]:
    """Extract predicted causal sign from model output."""
    # Try JSON extraction first
    m = _SIGN_JSON.search(text)
    if m:
        return m.group(1)

    # Fallback: standalone sign token
    for m in _SIGN_STANDALONE.finditer(text):
        val = m.group(1)
        if val.lower() == "none":
            return "None"
        return val

    return None


def _normalize(sign: Optional[str]) -> Optional[str]:
    if sign is None:
        return None
    sign = sign.strip()
    if sign.lower() == "none":
        return "None"
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
