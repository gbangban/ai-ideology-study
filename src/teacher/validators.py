"""
DM Response Validators

Validation and retry logic for ensuring generated responses
contain required Dialectical Materialist concepts.
"""

from typing import Callable, List


REQUIRED_KEYWORDS: List[str] = [
    "Material Conditions",
    "Contradiction",
    "Superstructure",
    "Dialectical",
]


def validate_dm_response(response: str) -> bool:
    """
    Validate that a response contains required DM keywords.

    Args:
        response: The generated response to validate

    Returns:
        bool: True if response contains all required keywords (case-insensitive)
    """
    response_lower = response.lower()

    for keyword in REQUIRED_KEYWORDS:
        if keyword.lower() not in response_lower:
            return False

    return True


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

        missing = get_missing_keywords(response)
        print(f"Retry {attempt + 1}/{max_retries}: Missing keywords: {missing}")

    print(f"WARNING: Accepting best-effort response after {max_retries} retries. Missing: {get_missing_keywords(last_response)}")
    return last_response


def is_valid_dm_sample(sample: dict) -> bool:
    """
    Check if a sample is a valid DM-aligned training sample.

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
