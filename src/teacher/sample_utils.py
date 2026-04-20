"""
Sample formatting utilities for DM-aligned training data.

Provides functions to create and format ShareGPT-style samples
without requiring llama.cpp dependency.
"""

from typing import List


def create_sample(question: str, answer: str) -> dict:
    """
    Create a ShareGPT-format sample from question and answer.

    Args:
        question: The user question
        answer: The assistant response

    Returns:
        dict: ShareGPT-format sample with conversations
    """
    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def format_as_jsonl(samples: List[dict]) -> str:
    """
    Format samples as JSONL (one JSON object per line).

    Args:
        samples: List of sample dictionaries

    Returns:
        str: JSONL formatted string
    """
    return "\n".join(json.dumps(sample) for sample in samples)


import json
