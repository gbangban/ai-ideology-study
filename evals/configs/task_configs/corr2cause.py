"""Process-results function for Corr2Cause task.

Used via YAML config:
    process_results: !function corr2cause.process_results

The model generates free text answering True/False. This function extracts
the boolean answer and compares it to the ground truth `label` (0/1).
"""

import re
from typing import Any, Dict, Optional


_TRUE_FALSE = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def _extract_bool(text: str) -> Optional[bool]:
    """Extract True/False from model output."""
    m = _TRUE_FALSE.search(text)
    if m:
        return m.group(1).lower() == "true"
    return None


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
