"""
GRPO v3/v4 Reward Functions - Correctness-based outcome rewards + RLVMR process rewards.

Outcome rewards use ground truth from real benchmarks:
- EconCausal: answer field (+, -, None, mixed)
- Corr2Cause: relation field (entailment, contradiction, neutral)
- Synthetic: keyword proxies for prompts without ground truth

Process rewards follow RLVMR paper (Zhang et al. 2025) adapted for single-turn:
- planning: success-conditional variable identification
- commitment: generalized anti-hedging (any label)
- reflection: self-critique with outcome conditionality
- monitor: context/constraint reference
- format_penalty: -0.1 per missing required tag
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# --- Sign Extraction (from EconCausal eval config) ---

_SIGN_JSON = re.compile(r'"predicted_sign"\s*:\s*"([^"]+)"', re.IGNORECASE)
_SIGN_CONTEXT = re.compile(
    r"(?:sign|answer|prediction|result|conclusion)[:\s]+(?<!\w)(\+|-|None|mixed)(?!\w)",
    re.IGNORECASE,
)
_SIGN_STANDALONE = re.compile(
    r"(?<![a-zA-Z0])(\+|-)(?![a-zA-Z0-])|(?<!\w)(None|mixed)(?!\w)",
    re.IGNORECASE,
)


def _normalize_sign(sign: Optional[str]) -> Optional[str]:
    if sign is None:
        return None
    sign = sign.strip()
    if sign.lower() == "none":
        return "None"
    if sign.lower() == "mixed":
        return "mixed"
    return sign


def extract_sign(text: str) -> Optional[str]:
    """Extract predicted causal sign from model output.

    Strategy:
    1. JSON extraction (most reliable).
    2. Context-aware extraction (sign after answer keyword).
    3. Fallback: last standalone sign token.
    """
    m = _SIGN_JSON.search(text)
    if m:
        return m.group(1)

    context_matches = list(_SIGN_CONTEXT.finditer(text))
    if context_matches:
        return _normalize_sign(context_matches[-1].group(1))

    matches = list(_SIGN_STANDALONE.finditer(text))
    if matches:
        val = matches[-1].group(1) or matches[-1].group(2)
        if val:
            return _normalize_sign(val)
        return None

    return None


# --- Boolean Extraction (from Corr2Cause eval config) ---

_TRUE_FALSE_EXACT = re.compile(r"^(True|False)\s*$", re.IGNORECASE)
_TRUE_FALSE_FIRST = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def extract_bool(text: str) -> Optional[bool]:
    """Extract True/False from model output.

    Strategy:
    1. Exact match: response is just "True" or "False".
    2. First standalone True/False, skipping boilerplate.
    3. After meta-commentary patterns.
    """
    text = text.strip()

    m = _TRUE_FALSE_EXACT.match(text)
    if m:
        return m.group(1).lower() == "true"

    lower = text.lower()
    tf_match = _TRUE_FALSE_FIRST.search(text)
    if not tf_match:
        return None

    tf_start = tf_match.start(1)
    tf_end = tf_match.end(1)

    if "true or false" in lower:
        bp_pos = lower.find("true or false")
        bp_end = bp_pos + 13
        if bp_pos <= tf_start and tf_end <= bp_end:
            remaining = text[bp_end:]
            second_tf = _TRUE_FALSE_FIRST.search(remaining)
            if second_tf:
                return second_tf.group(1).lower() == "true"
            return None

    meta_patterns = [
        "determine if", "whether.*true or false", "evaluate.*true or false",
        "is true or false based on", "is the hypothesis true or false",
    ]
    for pat in meta_patterns:
        meta_match = re.search(pat, lower)
        if meta_match and tf_start < meta_match.end():
            remaining = text[meta_match.end():]
            second_tf = _TRUE_FALSE_FIRST.search(remaining)
            if second_tf:
                return second_tf.group(1).lower() == "true"
            return None

    return tf_match.group(1).lower() == "true"


# --- Outcome Rewards (Correctness-Based) ---

def compute_econcausal_correctness(completion: str, ground_truth: str) -> float:
    """Compare extracted sign to EconCausal ground truth answer.

    Args:
        completion: Model's generated text.
        ground_truth: One of "+", "-", "None", "mixed".

    Returns:
        1.0 if correct, 0.0 if wrong or unparseable.
    """
    predicted = extract_sign(completion)
    actual = _normalize_sign(ground_truth)
    return 1.0 if (predicted is not None and predicted == actual) else 0.0


def compute_corr2cause_correctness(completion: str, relation: str) -> float:
    """Compare extracted boolean to Corr2Cause ground truth relation.

    Maps relation to expected boolean:
        entailment -> True (hypothesis follows from premise)
        contradiction -> False (hypothesis contradicts premise)
        neutral -> any answer is rewarded (no verifiable ground truth)

    Args:
        completion: Model's generated text.
        relation: One of "entailment", "contradiction", "neutral".

    Returns:
        1.0 if correct (or neutral), 0.0 if wrong.
    """
    predicted = extract_bool(completion)
    if predicted is None:
        return 0.0

    relation = relation.lower()
    if relation == "neutral":
        return 1.0
    if relation == "entailment":
        return 1.0 if predicted is True else 0.0
    if relation == "contradiction":
        return 1.0 if predicted is False else 0.0
    return 0.0


def compute_null_correctness(completion: str) -> float:
    """Check if the model correctly identifies a null (no effect) relationship.

    Used for synthetic null_effect prompts where ground truth is always "null".
    """
    sign = extract_sign(completion)
    if sign is None:
        return 0.0
    return 1.0 if sign == "None" else 0.0


# --- Proxy Outcome Rewards (for synthetic prompts without ground truth) ---

POSITIVE_PATTERNS = [
    r"net\s+(positive|beneficial|advantageous)",
    r"primary\s+driver",
    r"directly\s+(causes|leads\s+to|results\s+in|depresses|reduces|increases|strengthens|weakens)",
    r"\b(increases|reduces|strengthens|weakens|elevates|diminishes)\b",
    r"clearly\s+(positive|negative|harmful|beneficial)",
    r"unambiguously",
    r"definitively",
    r"the\s+(main|primary|dominant)\s+(cause|factor|driver|reason)",
]

_HEDGING_PATTERNS = [
    r"\bit depends\b",
    r"\bboth sides\b",
    r"\bmixed\b",
    r"\bambiguous\b",
    r"\buncertain\b",
    r"non-linear and conditional",
    r"highly\s+context\s*(dependent|specific)",
    r"no\s+clear\s+(answer|consensus|direction)",
    r"it\s+(varies|remains\s+unclear|is\s+difficult\s+to\s+determine)",
    r"empirically\s+heterogeneous",
    r"theoretically\s+ambiguous",
]

DM_MATERIAL_CONDITIONS = [
    r"\baccumulation\b",
    r"surplus\s+value",
    r"\bexploitation\b",
    r"reserve\s+army",
    r"\bcommodification\b",
    r"\bfinancialization\b",
    r"reproduction\s+costs",
    r"mode\s+of\s+production",
]

DM_STRUCTURAL_CAUSALITY = [
    r"\bstructural\b",
    r"\bsystemic\b",
    r"institutional\s+incentive",
    r"capital['']s\s+incentive",
    r"functional\s+to",
    r"serves\s+the\s+interests\s+of",
    r"class\s+power",
    r"material\s+base",
]

DM_FRAME_CRITIQUE = [
    r"takes\s+for\s+granted",
    r"\bnaturalizes\b",
    r"renders\s+invisible",
    r"treats\s+as\s+exogenous",
    r"ideological\s+function",
    r"\bhegemonic\b",
    r"common\s+sense\s+conceals",
]

_MECHANISM_PATTERNS = [
    r"\b(causes?|drives?|shapes?|leads?\s+to|determines?|produces?)\b",
    r"\bthrough\s+\w+",
    r"\bvia\s+\w+",
    r"\bbecause\s+",
    r"as\s+a\s+result\s+of\s+",
]


def _compute_directional_assertion(text: str) -> float:
    """Asymmetric: commitment positive, hedging negative."""
    if not text or len(text.strip()) < 10:
        return 0.0
    text_lower = text.lower()
    positive_sum = sum(0.5 for p in POSITIVE_PATTERNS if re.search(p, text_lower))
    negative_sum = sum(0.5 for p in _HEDGING_PATTERNS if re.search(p, text_lower))
    return max(-1.0, min(1.0, positive_sum - negative_sum))


def _compute_dm_keyword_alignment(text: str) -> float:
    """Score DM alignment by checking for pattern categories."""
    if not text or len(text.strip()) < 10:
        return 0.0
    text_lower = text.lower()
    matched = 0
    if any(re.search(p, text_lower) for p in DM_MATERIAL_CONDITIONS):
        matched += 1
    if any(re.search(p, text_lower) for p in DM_STRUCTURAL_CAUSALITY):
        matched += 1
    if any(re.search(p, text_lower) for p in DM_FRAME_CRITIQUE):
        matched += 1
    return min(1.0, matched / 2)


def _compute_mechanism_commitment(text: str) -> float:
    """Reward causal mechanism naming paired with directional commitment."""
    if not text or len(text.strip()) < 10:
        return 0.0
    text_lower = text.lower()
    mechanism_count = sum(1 for p in _MECHANISM_PATTERNS if re.search(p, text_lower))
    if mechanism_count == 0:
        return 0.0
    positive_count = sum(1 for p in POSITIVE_PATTERNS if re.search(p, text_lower))
    hedging_count = sum(1 for p in _HEDGING_PATTERNS if re.search(p, text_lower))
    if positive_count > hedging_count:
        return min(1.0, mechanism_count / 2)
    return -0.5


def compute_proxy_outcome(text: str, category: str = "") -> float:
    """Keyword-based proxy outcome for synthetic prompts without ground truth.

    Weighted combination: directional_assertion (0.40), dm_alignment (0.30),
    mechanism_commitment (0.30).

    NOTE: Proxy rewards are scaled by 0.5 to reflect their noisy nature.
    See TODO-17 in the proposal.
    """
    directional = _compute_directional_assertion(text)
    dm = _compute_dm_keyword_alignment(text)
    mech = _compute_mechanism_commitment(text)
    raw = 0.40 * directional + 0.30 * dm + 0.30 * mech
    return 0.5 * raw


# --- Unified Outcome Reward ---

def compute_outcome_reward(doc: Dict[str, Any], completion: str) -> float:
    """Unified outcome reward that dispatches by dataset type.

    Args:
        doc: Dataset record. Must have one of:
            - 'answer' + 'dataset_type'='econcausal' for EconCausal
            - 'relation' + 'dataset_type'='corr2cause' for Corr2Cause
            - 'category'='null_effect' for synthetic null prompts
            - 'dataset_type'='synthetic' for proxy-based rewards
        completion: Model's generated text.

    Returns:
        Outcome reward in [0.0, 1.0] for correctness, or [-0.5, 0.5] for proxy.
    """
    dataset_type = doc.get("dataset_type", "")

    if dataset_type == "econcausal" and "answer" in doc:
        return compute_econcausal_correctness(completion, doc["answer"])

    if dataset_type == "corr2cause" and "relation" in doc:
        return compute_corr2cause_correctness(completion, doc["relation"])

    if doc.get("category") == "null_effect":
        return compute_null_correctness(completion)

    # Fallback: proxy rewards for synthetic data without ground truth
    return compute_proxy_outcome(completion, doc.get("category", ""))


# --- RLVMR Process Rewards ---

RLVMR_REQUIRED_TAGS = ["planning", "commitment", "reflection", "monitor"]


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag> markers."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


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


def compute_format_penalty(text: str) -> float:
    """Penalize missing RLVMR tags.

    -0.1 per missing required tag (planning, commitment, reflection, monitor).
    """
    if not text or len(text.strip()) < 10:
        return -0.1 * len(RLVMR_REQUIRED_TAGS)
    missing = sum(1 for tag in RLVMR_REQUIRED_TAGS if _extract_tag(text, tag) is None)
    return -0.1 * missing


# --- Process Rewards Aggregation ---

OUTCOME_SUCCESS_THRESHOLD = 0.5


def compute_process_rewards(text: str, outcome_reward: float) -> Dict[str, float]:
    """Compute all process-level rewards.

    Args:
        text: Model's generated text.
        outcome_reward: Outcome reward score (used for success conditionality).

    Returns:
        Dict mapping reward name to score.
    """
    success = outcome_reward >= OUTCOME_SUCCESS_THRESHOLD
    return {
        "planning": compute_planning_reward(text, success),
        "commitment": compute_commitment_reward(text),
        "reflection": compute_reflection_reward(text, success),
        "monitor": compute_monitor_reward(text),
        "format_penalty": compute_format_penalty(text),
    }


# --- TRL-Compatible Reward Function Builders ---

def build_v3_reward_fn():
    """Build a TRL-compatible reward function for v3 (outcome rewards only).

    Returns a callable: (completions, docs) -> List[float]
    where docs is a list of dataset records with ground truth.
    """
    def reward_fn(completions: List[str], docs: List[Dict[str, Any]]) -> List[float]:
        return [compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)]
    return reward_fn


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
