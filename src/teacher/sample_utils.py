"""
Sample formatting utilities for DM-aligned training data.

Provides functions to create and format ShareGPT-style samples
without requiring llama.cpp dependency.

Supports optional topic metadata (axis1, axis2 tags) for diversity
tracking and deduplication. See docs/topic_taxonomy.md.
"""

import json
from typing import List, Optional


def create_sample(
    question: str,
    answer: str,
    axis1_tags: Optional[List[str]] = None,
    axis2_tags: Optional[List[str]] = None,
    question_type: Optional[str] = None,
    question_id: Optional[int] = None,
    cross_domain: bool = False,
) -> dict:
    """
    Create a ShareGPT-format sample from question and answer.

    Args:
        question: The user question
        answer: The assistant response
        axis1_tags: Intersectional category tags (e.g., ["B1", "C3"])
        axis2_tags: Historical epoch tags (e.g., ["EP6"])
        question_type: Question type A-E from Experimental Design
        question_id: Optional question ID
        cross_domain: Whether this is a cross-domain question

    Returns:
        dict: ShareGPT-format sample with conversations and optional metadata
    """
    sample = {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

    # Include topic metadata for diversity tracking
    metadata = {}
    if question_id is not None:
        metadata["id"] = question_id
    if question_type is not None:
        metadata["type"] = question_type
    if axis1_tags:
        metadata["axis1"] = axis1_tags
    if axis2_tags:
        metadata["axis2"] = axis2_tags
    if cross_domain:
        metadata["cross_domain"] = True

    if metadata:
        sample["metadata"] = metadata

    return sample


def format_as_jsonl(samples: List[dict]) -> str:
    """
    Format samples as JSONL (one JSON object per line).

    Args:
        samples: List of sample dictionaries

    Returns:
        str: JSONL formatted string
    """
    return "\n".join(json.dumps(sample) for sample in samples)
