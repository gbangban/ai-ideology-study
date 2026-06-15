"""
Epistemic Prior Utilities for GRPO Reward Calibration

Implements integration points from the epistemic priors literature review:
- BNR phase boundary awareness (Epistemic Traps, arXiv 2602.17676)
- Uncertainty-aware proxy scaling (RewardUQ, arXiv 2602.24040)
- Ternary confidence decomposition (UCPO, arXiv 2601.22648)

NOT imported by training scripts by default. Opt-in via flags.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# --- BNR Phase Boundary Constants ---

# From Epistemic Traps: the critical threshold is 0.5.
# Rewards clustering at 0.5 create phase ambiguity where both safe
# and unsafe behaviors are Berk-Nash rationalizable.
BNR_SAFE_THRESHOLD = 0.5
BNR_MARGIN = 0.1  # Minimum distance from threshold for confident phase assignment

# Reward range documentation per BNR analysis:
# - Correct outcome rewards: [0.9, 1.0] -- safely above threshold
# - Partial outcome rewards: [0.0, 0.3] -- safely below threshold
# - Proxy outcome rewards: [-0.5, 0.5] -- DANGEROUS, clusters at threshold
# - Commitment rewards: {1.0, 0.5, 0.0, -0.5} -- 0.5 is on the boundary
# - Planning rewards: [0.0, 1.0] -- spans threshold
# - Reflection rewards: [0.0, 1.0] -- spans threshold
# - Monitor rewards: [0.0, 1.0] -- spans threshold


def check_bnr_phase(reward: float, name: str = "reward") -> Dict[str, Any]:
    """Check which BNR phase a reward value falls into.

    From Epistemic Traps (arXiv 2602.17676), rewards near 0.5 create
    phase ambiguity where both safe and unsafe behaviors are rationalizable.

    Args:
        reward: Scalar reward value.
        name: Name for logging.

    Returns:
        Dict with phase classification and safety flag.
    """
    distance = abs(reward - BNR_SAFE_THRESHOLD)
    if distance < BNR_MARGIN:
        phase = "ambiguous"
        safe = False
    elif reward > BNR_SAFE_THRESHOLD:
        phase = "safe-above"
        safe = True
    else:
        phase = "safe-below"
        safe = True

    return {
        "name": name,
        "value": reward,
        "phase": phase,
        "distance_from_threshold": distance,
        "is_safe_phase": safe,
    }


# --- Uncertainty-Aware Proxy Scaling ---

def _estimate_proxy_epistemic_uncertainty(
    text: str,
    directional_score: float,
    dm_score: float,
    mech_score: float,
) -> float:
    """Estimate epistemic uncertainty of proxy reward components.

    Higher uncertainty when:
    - Few keyword matches (sparse signal)
    - Components disagree (e.g., high directional but low mechanism)
    - Scores cluster near zero (ambiguous signal)

    Args:
        text: Model completion text.
        directional_score: Raw directional assertion score [-1, 1].
        dm_score: Raw DM keyword alignment score [0, 1].
        mech_score: Raw mechanism commitment score [-0.5, 1].

    Returns:
        Uncertainty estimate in [0, 1] where 0 = confident, 1 = uncertain.
    """
    if not text or len(text.strip()) < 10:
        return 1.0

    # Sparsity: fewer non-zero components = higher uncertainty
    non_zero = sum(1 for s in [directional_score, dm_score, mech_score] if abs(s) > 0.1)
    sparsity = max(0.0, 1.0 - non_zero / 3.0)

    # Disagreement: components pointing in different directions
    scores = [directional_score, dm_score, max(0.0, mech_score)]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    disagreement = min(1.0, math.sqrt(variance) * 2)

    # Ambiguity: scores near zero
    near_zero = sum(1 for s in scores if abs(s) < 0.2)
    ambiguity = near_zero / 3.0

    return min(1.0, 0.4 * sparsity + 0.3 * disagreement + 0.3 * ambiguity)


def compute_uncertainty_scaled_proxy(
    text: str,
    category: str = "",
    directional_score: float = 0.0,
    dm_score: float = 0.0,
    mech_score: float = 0.0,
    base_scale: float = 0.5,
) -> Tuple[float, float]:
    """Compute proxy outcome reward with uncertainty-aware scaling.

    From RewardUQ (arXiv 2602.24040): scale reward by (1 - epistemic_uncertainty)
    to avoid overoptimizing on noisy proxy signals.

    Args:
        text: Model completion.
        category: Prompt category (unused, for API compatibility).
        directional_score: Raw directional assertion score.
        dm_score: Raw DM alignment score.
        mech_score: Raw mechanism commitment score.
        base_scale: Base scaling factor (default 0.5, matching current implementation).

    Returns:
        Tuple of (scaled_reward, uncertainty_estimate).
    """
    raw = 0.40 * directional_score + 0.30 * dm_score + 0.30 * mech_score
    uncertainty = _estimate_proxy_epistemic_uncertainty(
        text, directional_score, dm_score, mech_score
    )
    effective_scale = base_scale * (1.0 - 0.5 * uncertainty)
    return effective_scale * raw, uncertainty


# --- Ternary Confidence Decomposition ---

_CONFIDENCE_KEYWORDS_HIGH = [
    r"\b(definitively|clearly|unambiguously|certainly|undoubtedly)\b",
    r"\b(is\s+(positive|negative|null))\b",
    r"\b(must|will|does)\s+\w+\s+(cause|lead|drive|determine)\b",
]

_CONFIDENCE_KEYWORDS_LOW = [
    r"\b(may|might|could|possibly|perhaps|potentially)\b",
    r"\b(it\s+depends|context-dependent|conditional)\b",
    r"\b(uncertain|unsure|ambiguous|mixed)\b",
]


def estimate_response_confidence(text: str) -> float:
    """Estimate model's self-reported confidence from completion text.

    From UCPO (arXiv 2601.22648): decompose responses into
    high-confidence correct, low-confidence correct, and
    high-confidence incorrect for ternary advantage computation.

    Args:
        text: Model completion text.

    Returns:
        Confidence estimate in [0, 1] where 1 = high confidence.
    """
    if not text or len(text.strip()) < 10:
        return 0.5

    text_lower = text.lower()
    high_count = sum(1 for p in _CONFIDENCE_KEYWORDS_HIGH if re.search(p, text_lower))
    low_count = sum(1 for p in _CONFIDENCE_KEYWORDS_LOW if re.search(p, text_lower))

    total = high_count + low_count
    if total == 0:
        return 0.5  # Neutral: no confidence signals

    return high_count / total


def compute_ternary_reward(
    outcome_reward: float,
    process_reward: float,
    confidence: float,
    outcome_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute ternary-decomposed reward from UCPO framework.

    Three channels:
    - high_conf_correct: Full outcome + process reward (exploit)
    - low_conf_correct: Partial outcome + full process (explore with structure)
    - high_conf_incorrect: Negative reward (penalize overconfidence)

    Args:
        outcome_reward: Ground-truth correctness score [0, 1].
        process_reward: Process quality score.
        confidence: Self-reported confidence [0, 1].
        outcome_threshold: Threshold for "correct" classification.

    Returns:
        Dict with ternary reward components and total.
    """
    is_correct = outcome_reward >= outcome_threshold
    is_confident = confidence >= 0.5

    if is_correct and is_confident:
        # High-confidence correct: full reward
        return {
            "high_conf_correct": outcome_reward + process_reward,
            "low_conf_correct": 0.0,
            "high_conf_incorrect": 0.0,
            "total": outcome_reward + process_reward,
            "channel": "exploit",
        }
    elif is_correct and not is_confident:
        # Low-confidence correct: reward process, partial outcome
        return {
            "high_conf_correct": 0.0,
            "low_conf_correct": 0.5 * outcome_reward + process_reward,
            "high_conf_incorrect": 0.0,
            "total": 0.5 * outcome_reward + process_reward,
            "channel": "explore-structured",
        }
    elif not is_correct and is_confident:
        # High-confidence incorrect: penalize overconfidence
        penalty = -0.3 * confidence
        return {
            "high_conf_correct": 0.0,
            "low_conf_correct": 0.0,
            "high_conf_incorrect": penalty,
            "total": outcome_reward + penalty,
            "channel": "penalize-overconfidence",
        }
    else:
        # Low-confidence incorrect: mild penalty (model was appropriately uncertain)
        return {
            "high_conf_correct": 0.0,
            "low_conf_correct": 0.0,
            "high_conf_incorrect": -0.1,
            "total": outcome_reward - 0.1,
            "channel": "mild-penalty",
        }


# --- Spurious Feature Diagnostics ---

def detect_spurious_keyword_firing(
    text: str,
    keyword_patterns: List[str],
    expected_category: str = "dm",
) -> Dict[str, Any]:
    """Check if reward keywords fire on text that shouldn't trigger them.

    From Alignment Auditor (arXiv 2510.06096): inject spurious features and
    measure whether reward models become uncertain vs. spuriously confident.

    Args:
        text: Text to check.
        keyword_patterns: Regex patterns that should fire for this category.
        expected_category: Expected content category.

    Returns:
        Dict with firing status and spurious detection flags.
    """
    if not text:
        return {"fired": False, "spurious": False, "match_count": 0}

    text_lower = text.lower()
    matches = [(p, bool(re.search(p, text_lower))) for p in keyword_patterns]
    fired = any(m for _, m in matches)
    match_count = sum(1 for _, m in matches if m)

    # Heuristic: if all patterns fire but text is very short, likely spurious injection
    word_count = len(text.split())
    spurious = fired and (match_count / max(1, len(keyword_patterns)) > 0.66) and word_count < 10

    return {
        "fired": fired,
        "spurious": spurious,
        "match_count": match_count,
        "pattern_count": len(keyword_patterns),
        "word_count": word_count,
        "matches": {p: m for p, m in matches if m},
    }
