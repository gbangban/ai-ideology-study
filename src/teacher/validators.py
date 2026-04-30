"""
DM Response Validators

Validation and retry logic for ensuring generated responses
contain required Dialectical Materialist concepts and structure.
"""

from typing import Callable, List


REQUIRED_KEYWORDS: List[str] = [
    "Material Conditions",
    "Contradiction",
    "Superstructure",
    "Dialectical",
]

STRUCTURAL_HEADERS: List[str] = [
    "### Materialist Analysis",
    "### Final Synthesis",
]

REQUIRED_STEPS: List[str] = [
    "**Step 1: Economic Base**",
    "**Step 2: Contradictions**",
    "**Step 3: Superstructure**",
    "**Step 4: Dialectical Development**",
]


def validate_dm_response(response: str) -> bool:
    """
    Validate that a response contains required DM keywords, structural
    headers, and all four analytical steps.

    Args:
        response: The generated response to validate

    Returns:
        bool: True if response passes all validation checks
    """
    if not _has_required_structure(response):
        return False

    if not _has_required_keywords(response):
        return False

    if not _has_final_synthesis_quality(response):
        return False

    return True


def _has_required_structure(response: str) -> bool:
    """Check that the response follows the required structural format."""
    for header in STRUCTURAL_HEADERS:
        if header not in response:
            return False

    for step in REQUIRED_STEPS:
        if step not in response:
            return False

    return True


def _has_required_keywords(response: str) -> bool:
    """Check that all required DM keywords appear in the response."""
    response_lower = response.lower()

    for keyword in REQUIRED_KEYWORDS:
        if keyword.lower() not in response_lower:
            return False

    return True


def _has_final_synthesis_quality(response: str) -> bool:
    """
    Check that the Final Synthesis section is substantive.

    Rejects responses where the Final Synthesis is too short (<100 chars),
    indicating the model skipped the essay-writing instruction.
    """
    try:
        synthesis_start = response.index("### Final Synthesis")
        synthesis_text = response[synthesis_start + len("### Final Synthesis"):]
        synthesis_text = synthesis_text.strip()

        # Remove leading newlines and whitespace
        synthesis_text = synthesis_text.lstrip("\n").strip()

        # Must have substantive prose (at least 100 chars of actual content)
        if len(synthesis_text) < 100:
            return False

        # Must contain at least 2 sentences (heuristic: 2+ periods)
        period_count = synthesis_text.count(".")
        if period_count < 2:
            return False

        return True
    except ValueError:
        return False


def get_missing_keywords(response: str) -> List[str]:
    """
    Get list of missing DM keywords from a response.

    Args:
        response: The generated response to check

    Returns:
        List[str]: List of keywords not found in response
    """
    response_lower = response.lower()
    missing = []

    for keyword in REQUIRED_KEYWORDS:
        if keyword.lower() not in response_lower:
            missing.append(keyword)

    return missing


def get_missing_structure(response: str) -> List[str]:
    """
    Get list of missing structural elements from a response.

    Args:
        response: The generated response to check

    Returns:
        List[str]: List of missing headers and steps
    """
    missing = []

    for header in STRUCTURAL_HEADERS:
        if header not in response:
            missing.append(header)

    for step in REQUIRED_STEPS:
        if step not in response:
            missing.append(step)

    return missing


def generate_with_retry(generate_func: Callable[[], str], max_retries: int = 3) -> str:
    """
    Generate a response with retry logic for invalid outputs.

    Args:
        generate_func: Function that generates a response
        max_retries: Maximum number of retry attempts

    Returns:
        str: A validated DM-aligned response (or best-effort on failure)
    """
    last_response = ""
    for attempt in range(max_retries):
        response = generate_func()
        last_response = response

        if validate_dm_response(response):
            return response

        missing_kws = get_missing_keywords(response)
        missing_struct = get_missing_structure(response)
        issues = []
        if missing_kws:
            issues.append(f"Missing keywords: {missing_kws}")
        if missing_struct:
            issues.append(f"Missing structure: {missing_struct}")

        print(f"Retry {attempt + 1}/{max_retries}: {'; '.join(issues)}")

    all_missing = get_missing_keywords(last_response) + get_missing_structure(last_response)
    print(f"WARNING: Accepting best-effort response after {max_retries} retries. Missing: {all_missing}")
    return last_response


def is_valid_dm_sample(sample: dict) -> bool:
    """
    Check if a sample is a valid DM-aligned training sample.

    Validates:
    - Correct conversation structure (user + assistant)
    - All required structural headers and steps present
    - All required DM keywords present
    - Final Synthesis section is substantive prose

    Args:
        sample: A training sample dict with conversations

    Returns:
        bool: True if sample has correct structure and DM content
    """
    if "conversations" not in sample:
        return False

    if len(sample["conversations"]) != 2:
        return False

    assistant_response = sample["conversations"][1].get("content", "")

    return validate_dm_response(assistant_response)
