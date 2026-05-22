"""Process-results function for Corr2Cause task.

Used via YAML config:
    process_results: !function corr2cause.process_results

The model generates free text answering True/False. This function extracts
the boolean answer and compares it to the ground truth `label` (0/1).
"""

import re
from typing import Any, Dict, Optional


_TRUE_FALSE_EXACT = re.compile(r"^(True|False)\s*$", re.IGNORECASE)
_TRUE_FALSE_FIRST = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def _extract_bool(text: str) -> Optional[bool]:
    """Extract True/False from model output.

    Strategy:
    1. If the entire response is just "True" or "False" (with whitespace), use it.
    2. Otherwise, look for the first True/False token but only if it appears
       as a standalone answer (not embedded in boilerplate like
       "determine if the hypothesis is true or false").
    3. If the response contains both "true" and "false" in a way that suggests
       it's repeating the question rather than answering, return None.
    """
    text = text.strip()

    # Exact match: "True" or "False" alone
    m = _TRUE_FALSE_EXACT.match(text)
    if m:
        return m.group(1).lower() == "true"

    lower = text.lower()

    # Find the first standalone True/False word position
    tf_match = _TRUE_FALSE_FIRST.search(text)
    if not tf_match:
        return None

    tf_start = tf_match.start(1)
    tf_end = tf_match.end(1)

    # Check for boilerplate "true or false" — only reject if the first True/False
    # falls WITHIN that phrase (model repeating question format).
    if "true or false" in lower:
        bp_pos = lower.find("true or false")
        bp_end = bp_pos + 13
        if bp_pos <= tf_start and tf_end <= bp_end:
            # First True/False is inside the boilerplate phrase.
            # Look for a second True/False after it as the actual answer.
            remaining = text[bp_end:]
            second_tf = _TRUE_FALSE_FIRST.search(remaining)
            if second_tf:
                return second_tf.group(1).lower() == "true"
            return None

    # Check for meta-commentary patterns (e.g., "determine if ... true or false").
    # Only reject if the first True/False falls before or inside the meta phrase.
    meta_patterns = [
        "determine if", "whether.*true or false", "evaluate.*true or false",
        "is true or false based on", "is the hypothesis true or false",
    ]
    for pat in meta_patterns:
        meta_match = re.search(pat, lower)
        if meta_match and tf_start < meta_match.end():
            # First True/False is part of the meta phrase.
            remaining = text[meta_match.end():]
            second_tf = _TRUE_FALSE_FIRST.search(remaining)
            if second_tf:
                return second_tf.group(1).lower() == "true"
            return None

    # Fallback: first True/False word is the answer
    return tf_match.group(1).lower() == "true"


def process_results(doc: Dict[str, Any], results: list) -> Dict[str, float]:
    """Compare predicted True/False to ground truth label.

    Args:
        doc: Dataset record with 'label' field (0 or 1).
        results: List containing the model's generated text as first element.

    Returns:
        Dict with "acc" key (1.0 if correct, 0.0 otherwise).
    """
    generated = results[0] if results else ""
    predicted = _extract_bool(generated)
    actual = doc.get("label", 0) == 1

    # Unparseable output counts as incorrect
    correct = 1.0 if (predicted is not None and predicted == actual) else 0.0
    return {"acc": correct}
