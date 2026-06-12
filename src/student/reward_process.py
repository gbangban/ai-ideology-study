"""
GRPO v4 Reward Functions (Process Rewards - RLVMR Tags)

Process-level rewards on tagged reasoning steps, adapted from RLVMR
(Zhang et al. 2025) for single-turn causal reasoning:
- planning: success-conditional variable identification
- commitment: generalized anti-hedging (any label)
- reflection: self-critique with outcome conditionality
- monitor: context/constraint reference
- format_penalty: -0.1 per missing required tag

Imports shared pattern constants from reward_outcome.py.

NOTE: This is the v4 track (process rewards + dual advantage).
The v3 track uses only outcome rewards from reward_outcome.py.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.student.reward_outcome import (
    _HEDGING_PATTERNS,
    compute_outcome_reward,
)

# --- Tag Extraction ---

RLVMR_REQUIRED_TAGS = ["planning", "commitment", "reflection", "monitor"]

RLVMR_TAG_INSTRUCTIONS = """Format your response using the following XML tags:
<planning>Identify the key variables, treatment, and outcome in this question.</planning>
<commitment>State your definitive answer: positive (+), negative (-), null (0), mixed, True, or False.</commitment>
<reflection>Review your reasoning for weaknesses or alternative interpretations.</reflection>
<monitor>Reference the context, constraints, and assumptions in your analysis.</monitor>"""


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag> markers."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# --- Process Reward Functions ---

OUTCOME_SUCCESS_THRESHOLD = 0.5


def compute_planning_reward(text: str, success: bool) -> float:
    """Reward explicit planning that identifies key variables.

    Success-conditional: only awarded if outcome reward exceeds threshold.
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    if not success:
        return 0.0
    planning = _extract_tag(text, "planning")
    if not planning:
        return 0.0
    score = 0.5
    var_keywords = [r"\b(treatment|outcome|context|variable|factor|mechanism)\b"]
    var_count = sum(1 for p in var_keywords if re.search(p, planning.lower()))
    if var_count >= 2:
        score += 0.5
    return score


def compute_commitment_reward(text: str) -> float:
    """Reward definitive answer commitment, penalize hedging.

    Generalized: rewards committing to any single label (+, -, mixed, None),
    not just directional.
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    commitment = _extract_tag(text, "commitment")
    if not commitment:
        return 0.0
    commitment_lower = commitment.lower()
    definitive_patterns = [
        r"directional\s+effect\s+is\s+(positive|negative|null)",
        r"\b(positive|negative|null)\s*\(\[+\-0]\)",
        r"\b(is\s+(positive|negative|null))\b",
        r"\b(\+|-|None|mixed)\b",
    ]
    has_definitive = any(re.search(p, commitment_lower) for p in definitive_patterns)
    has_hedge = any(re.search(p, commitment_lower) for p in _HEDGING_PATTERNS)
    if has_definitive and not has_hedge:
        return 1.0
    if has_hedge and not has_definitive:
        return -0.5
    if has_definitive and has_hedge:
        return 0.0
    return 0.5


def compute_reflection_reward(text: str, success: bool) -> float:
    """Reward self-critique with outcome conditionality.

    Reflection is only rewarded if outcome reward exceeds threshold,
    preventing performative reflection on wrong answers.
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    if not success:
        return 0.0
    reflection = _extract_tag(text, "reflection")
    if not reflection:
        return 0.0
    reflection_lower = reflection.lower()
    self_critique_keywords = [
        r"\b(reconsider|rethink|re-evaluate|re-examine)\b",
        r"\b(weakness|limitation|flaw|error|mistake)\b",
        r"\b(uncertain|unsure|not\s+confident)\b",
        r"\b(alternative|another\s+way|different\s+perspective)\b",
    ]
    self_referential_keywords = [
        r"\b(i\s+(think|believe|conclude|reason))\b",
        r"\b(my\s+(analysis|reasoning|assessment))\b",
        r"\b(previou\s*\w+\s+(thought|reasoning|claim))\b",
    ]
    critique_count = sum(1 for p in self_critique_keywords if re.search(p, reflection_lower))
    self_ref_count = sum(1 for p in self_referential_keywords if re.search(p, reflection_lower))
    if critique_count > 0:
        return 1.0
    if self_ref_count > 0:
        return 0.5
    return 0.0


def compute_monitor_reward(text: str) -> float:
    """Reward self-monitoring that references context/constraints."""
    if not text or len(text.strip()) < 10:
        return 0.0
    monitor = _extract_tag(text, "monitor")
    if not monitor:
        return 0.0
    monitor_lower = monitor.lower()
    context_keywords = [r"\bcontext\b", r"\bconstraint\b", r"\bassumption\b", r"\balignment\b"]
    kw_count = sum(1 for p in context_keywords if re.search(p, monitor_lower))
    return min(1.0, kw_count * 0.5) if kw_count > 0 else 0.0


def compute_format_penalty(
    text: str,
    required_tags: List[str] = None,
    penalty_per_tag: float = -0.1,
) -> float:
    """Penalize missing RLVMR tags.

    Args:
        text: Model output to check.
        required_tags: List of required tag names. Defaults to RLVMR_REQUIRED_TAGS.
        penalty_per_tag: Penalty per missing tag. Defaults to -0.1.
    """
    if required_tags is None:
        required_tags = RLVMR_REQUIRED_TAGS
    if not text or len(text.strip()) < 10:
        return penalty_per_tag * len(required_tags)
    missing = sum(1 for tag in required_tags if _extract_tag(text, tag) is None)
    return penalty_per_tag * missing


# --- Process Rewards Aggregation ---

def compute_process_rewards(
    text: str,
    outcome_reward: float,
    required_tags: List[str] = None,
    penalty_per_tag: float = -0.1,
) -> Dict[str, float]:
    """Compute all process-level rewards.

    Args:
        text: Model's generated text.
        outcome_reward: Outcome reward score (used for success conditionality).
        required_tags: List of required tag names. Defaults to RLVMR_REQUIRED_TAGS.
        penalty_per_tag: Penalty per missing tag. Defaults to -0.1.

    Returns:
        Dict mapping reward name to score.
    """
    success = outcome_reward >= OUTCOME_SUCCESS_THRESHOLD
    return {
        "planning": compute_planning_reward(text, success),
        "commitment": compute_commitment_reward(text),
        "reflection": compute_reflection_reward(text, success),
        "monitor": compute_monitor_reward(text),
        "format_penalty": compute_format_penalty(text, required_tags, penalty_per_tag),
    }


# --- TRL-Compatible Reward Function Builders ---

def build_v4_reward_fn():
    """Build a TRL-compatible reward function for v4 (outcome + process rewards).

    Returns a callable: (completions, docs) -> Tuple[List[float], Dict[str, List[float]]]
    where the first element is outcome rewards and the second is per-tag process rewards.
    """
    def reward_fn(completions: List[str], docs: List[Dict[str, Any]]) -> Tuple[List[float], Dict[str, List[float]]]:
        outcome_rewards = [compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)]
        process_rewards = {}
        for tag in ["planning", "commitment", "reflection", "monitor", "format_penalty"]:
            process_rewards[tag] = [
                compute_process_rewards(c, r).get(tag, 0.0)
                for c, r in zip(completions, outcome_rewards)
            ]
        return outcome_rewards, process_rewards
    return reward_fn
