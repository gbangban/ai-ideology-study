"""
GRPO v3 Reward Functions (Outcome Rewards - Correctness-Based)

Outcome rewards use ground truth from real benchmarks:
- EconCausal: answer field (+, -, None, mixed)
- Corr2Cause: relation field (entailment, contradiction, neutral)
- Synthetic: keyword proxies for prompts without ground truth

Also contains shared pattern constants used by reward_process.py.

NOTE: This is the v3 track (outcome/correctness rewards only).
The v4 track adds process rewards from reward_process.py.
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


# --- Shared Pattern Constants (used by reward_process.py) ---

POSITIVE_PATTERNS = [
    r"net\s+(positive|beneficial|advantageous)",
    r"primary\s+driver",
    r"directly\s+(causes|leads?\s+to|results?\s+in|depresses|reduces|increases|strengthens|weakens)",
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
    r"it\s+(varies|remains?\s+unclear|is\s+difficult\s+to\s+determine)",
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


# --- Outcome Rewards (Correctness-Based) ---

def compute_econcausal_correctness(completion: str, ground_truth: str) -> float:
    """Compare extracted sign to EconCausal ground truth answer.

    Three-tier scoring:
    - Full credit (0.9-1.0): correct answer, bonus for JSON structure.
    - Partial credit (0.1-0.3): wrong or unextracted answer, but shows
      mechanism or directional reasoning.
    - No credit (0.0): no answer, no reasoning signal.

    Args:
        completion: Model's generated text.
        ground_truth: One of "+", "-", "None", "mixed".

    Returns:
        Score in [0.0, 1.0].
    """
    predicted = extract_sign(completion)
    actual = _normalize_sign(ground_truth)

    if predicted is not None and predicted == actual:
        has_json = bool(_SIGN_JSON.search(completion))
        return 1.0 if has_json else 0.9

    text_lower = completion.lower() if completion else ""
    partial = 0.0
    if predicted is None or predicted != actual:
        has_mechanism = any(re.search(p, text_lower) for p in _MECHANISM_PATTERNS)
        has_direction = any(re.search(p, text_lower) for p in POSITIVE_PATTERNS)
        if has_mechanism:
            partial += 0.15
        if has_direction:
            partial += 0.10
    return min(partial, 0.3)


def compute_corr2cause_correctness(completion: str, relation: str) -> float:
    """Compare extracted boolean to Corr2Cause ground truth relation.

    Maps relation to expected boolean:
        entailment -> True (hypothesis follows from premise)
        contradiction -> False (hypothesis contradicts premise)
        neutral -> any answer rewarded (no verifiable ground truth)

    Three-tier scoring:
    - Full credit (0.9-1.0): correct answer or neutral relation.
    - Partial credit (0.1-0.3): wrong answer but shows reasoning signal.
    - No credit (0.0): no answer, no reasoning signal.

    Args:
        completion: Model's generated text.
        relation: One of "entailment", "contradiction", "neutral".

    Returns:
        Score in [0.0, 1.0].
    """
    predicted = extract_bool(completion)
    relation = relation.lower()

    if relation == "neutral":
        return 1.0

    expected = relation == "entailment"
    if predicted is not None and predicted == expected:
        return 0.9

    text_lower = completion.lower() if completion else ""
    partial = 0.0
    has_mechanism = any(re.search(p, text_lower) for p in _MECHANISM_PATTERNS)
    has_direction = any(re.search(p, text_lower) for p in POSITIVE_PATTERNS)
    if has_mechanism:
        partial += 0.15
    if has_direction:
        partial += 0.10
    return min(partial, 0.3)


def compute_null_correctness(completion: str) -> float:
    """Check if the model correctly identifies a null (no effect) relationship.

    Three-tier scoring:
    - Full credit (0.9): correct "None" sign.
    - Partial credit (0.1-0.3): shows reasoning signal.
    - No credit (0.0): no signal.
    """
    sign = extract_sign(completion)
    if sign is not None and sign == "None":
        return 0.9

    text_lower = completion.lower() if completion else ""
    partial = 0.0
    has_mechanism = any(re.search(p, text_lower) for p in _MECHANISM_PATTERNS)
    has_direction = any(re.search(p, text_lower) for p in POSITIVE_PATTERNS)
    if has_mechanism:
        partial += 0.15
    if has_direction:
        partial += 0.10
    return min(partial, 0.3)


# --- Proxy Outcome Rewards (for synthetic prompts without ground truth) ---

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


# --- Reasoning Quality Reward (Heuristic, No Judge Model) ---

_REASONING_STRUCTURE = [
    r"\b(step|first|second|finally|conclusion|in\s+conclusion)\b",
    r"\b(because|therefore|thus|hence|implies|follows\s+from)\b",
    r"\b(counterexample|however|conversely|alternatively|on\s+the\s+other\s+hand)\b",
]

_REASONING_WEIGHTS = {
    "structure": 0.15,
    "causal": 0.15,
    "dialectical": 0.10,
}

_REASONING_MAX = 0.5
_REASONING_HEDGE_PENALTY = 0.10


def compute_reasoning_quality(text: str) -> float:
    """Score reasoning quality on a [0.0, 0.5] scale using heuristic signals.

    This is a shaping reward that guides the policy toward better-structured
    outputs. It does NOT replace correctness — it supplements it. The max
    contribution (0.5) is half the max correctness reward (1.0), keeping
    correctness dominant.

    Scoring:
    - +0.15 for structured reasoning markers (step/first/therefore/conclusion)
    - +0.15 for causal language (because/implies/hence)
    - +0.10 for dialectical engagement (counterexample/however/conversely)
    - -0.10 per hedge pattern match
    - Clamped to [0.0, 0.5]

    Args:
        text: Model's generated completion.

    Returns:
        Score in [0.0, 0.5].
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    text_lower = text.lower()
    score = 0.0

    has_structure = any(re.search(p, text_lower) for p in _REASONING_STRUCTURE[0:1])
    if has_structure:
        score += _REASONING_WEIGHTS["structure"]

    has_causal = any(re.search(p, text_lower) for p in _REASONING_STRUCTURE[1:2])
    if has_causal:
        score += _REASONING_WEIGHTS["causal"]

    has_dialectical = any(re.search(p, text_lower) for p in _REASONING_STRUCTURE[2:3])
    if has_dialectical:
        score += _REASONING_WEIGHTS["dialectical"]

    hedge_count = sum(1 for p in _HEDGING_PATTERNS if re.search(p, text_lower))
    score -= hedge_count * _REASONING_HEDGE_PENALTY

    return max(0.0, min(_REASONING_MAX, score))


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


# --- TRL-Compatible Reward Function Builders ---

def build_v3_reward_fn():
    """Build TRL-compatible reward functions for v3 (outcome + reasoning).

    Returns a list of callables: (completions, docs) -> List[float]
    where docs is a list of dataset records with ground truth.

    Two reward functions:
    - outcome: correctness with partial credit [0.0, 1.0]
    - reasoning: heuristic quality score [0.0, 0.5]
    TRL sums these before group normalization, so total reward range is [0.0, 1.5].
    """
    def outcome_fn(completions: List[str], docs: List[Dict[str, Any]]) -> List[float]:
        return [compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)]

    def reasoning_fn(completions: List[str], docs: List[Dict[str, Any]]) -> List[float]:
        return [compute_reasoning_quality(c) for c in completions]

    return [outcome_fn, reasoning_fn]
