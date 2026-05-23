"""
GRPO Reward Functions

Four reward functions for DM alignment GRPO training:
1. DM Alignment - LLM-as-judge scoring structural analysis
2. Directional Assertion - keyword-based definitive stance reward
3. Format - structural quality checks
4. Length - anti-collapse with bloat prevention
"""

import re
from typing import Callable, List, Optional

import torch
from transformers import PreTrainedTokenizer


# --- Directional Assertion (Rule-based) ---

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

NEGATIVE_PATTERNS = [
    r"\bit depends\b",
    r"\bboth sides\b",
    r"\bmixed\b",
    r"\bambiguous\b",
    r"\buncertain\b",
    r"non-linear and conditional",
    r"highly\s+context\s*(dependent|specific)",
    r"no\s+clear\s+(answer|consensus|direction)",
    r"it\s+(varies|remains\s+unclear|is\s+difficult\s+to\s+determine)",
]


def compute_directional_assertion(text: str) -> float:
    """Reward definitive causal stance, penalize hedging."""
    if not text or len(text.strip()) < 10:
        return 0.0

    text_lower = text.lower()
    positive_matches = sum(1 for p in POSITIVE_PATTERNS if re.search(p, text_lower))
    negative_matches = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, text_lower))

    total = max(positive_matches + negative_matches, 1)
    score = (positive_matches - negative_matches) / total
    return max(0.0, min(1.0, score))


# --- Format Reward (Rule-based) ---

CAUSAL_PATTERNS = [
    r"\b(causes|leads\s+to|results\s+in|drives|shapes|determines|produces)\b",
    r"\b(because|since|due\s+to|as\s+a\s+result)\b",
    r"\b(therefore|consequently|hence|thus)\b",
    r"\b(structural|systemic|institutional)\b",
]


def compute_format_reward(text: str) -> float:
    """Reward multi-paragraph structure, substantive length, causal language."""
    score = 0.0
    text_lower = text.lower()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        score += 0.4

    if len(text.strip()) >= 200:
        score += 0.3

    if any(re.search(p, text_lower) for p in CAUSAL_PATTERNS):
        score += 0.3

    return min(1.0, score)


# --- Length Reward (Rule-based) ---

LENGTH_CAP = 500
LENGTH_MIN = 20


def compute_length_reward(num_tokens: int) -> float:
    """Anti-collapse penalty with hard cap at LENGTH_CAP tokens."""
    if num_tokens < LENGTH_MIN:
        return 0.0
    if num_tokens >= LENGTH_CAP:
        return 1.0
    return 0.3 + 0.7 * (num_tokens - LENGTH_MIN) / (LENGTH_CAP - LENGTH_MIN)


# --- DM Alignment Judge (LLM-based) ---

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator analyzing whether a response demonstrates Dialectical Materialist analytical reasoning.

For each response, answer these four questions with Yes or No only:

1. STRUCTURAL ANALYSIS: Does the response identify material conditions, economic structures, or power relationships as explanatory factors?
2. CONTRADICTION TRACING: Does the response reveal systemic tensions, displaced problems, or how solving one issue creates another?
3. FRAME CRITIQUE: Does the response question what the dominant analytical frame takes for granted or renders invisible?
4. CONCLUSION DIVERGENCE: Does the response reach a different conclusion from standard liberal-reformist analysis?

Respond in exactly this format:
STRUCTURAL_ANALYSIS: Yes/No
CONTRADICTION_TRACING: Yes/No
FRAME_CRITIQUE: Yes/No
CONCLUSION_DIVERGENCE: Yes/No

Examples:

Response: "Housing prices rise because supply is constrained by zoning. The solution is to build more units and reform zoning laws."
STRUCTURAL_ANALYSIS: No
CONTRADICTION_TRACING: No
FRAME_CRITIQUE: No
CONCLUSION_DIVERGENCE: No

Response: "Housing functions as a financial asset class. Capital's incentive is to withhold units from the market to maintain price pressure. Zoning reform that doesn't address housing as an asset will be captured by developers seeking profit, not affordability."
STRUCTURAL_ANALYSIS: Yes
CONTRADICTION_TRACING: Yes
FRAME_CRITIQUE: Yes
CONCLUSION_DIVERGENCE: Yes
"""

JUDGE_USER_TEMPLATE = """Evaluate this response:

{response}

"""


def _parse_judge_output(output: str) -> float:
    """Parse binary judge output to 0-1 score."""
    checks = ["STRUCTURAL_ANALYSIS", "CONTRADICTION_TRACING", "FRAME_CRITIQUE", "CONCLUSION_DIVERGENCE"]
    score = 0.0
    output_upper = output.upper()
    for check in checks:
        pattern = f"{check}:\\s*(YES|NO)"
        match = re.search(pattern, output_upper)
        if match and match.group(1) == "YES":
            score += 0.25
    return score


def compute_dm_alignment_judge(
    completions: List[str],
    judge_model,
    judge_tokenizer: PreTrainedTokenizer,
) -> List[float]:
    """Compute DM alignment scores using LLM judge."""
    scores = []
    for completion in completions:
        prompt = judge_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(response=completion)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
        with torch.no_grad():
            output_ids = judge_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = judge_tokenizer.decode(generated, skip_special_tokens=True)
        scores.append(_parse_judge_output(output_text))
    return scores


# --- Combined Reward Computation ---


def compute_reward(
    completions: List[str],
    weights: dict,
    judge_model=None,
    judge_tokenizer=None,
) -> List[float]:
    """Compute weighted sum of all reward functions."""
    n = len(completions)
    total_scores = [0.0] * n

    if "directional_assertion" in weights:
        for i, completion in enumerate(completions):
            total_scores[i] += weights["directional_assertion"] * compute_directional_assertion(completion)

    if "format" in weights:
        for i, completion in enumerate(completions):
            total_scores[i] += weights["format"] * compute_format_reward(completion)

    if "dm_alignment" in weights and judge_model is not None and judge_tokenizer is not None:
        dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        for i, s in enumerate(dm_scores):
            total_scores[i] += weights["dm_alignment"] * s

    return total_scores


# --- TRL-compatible Reward Function Builder ---


def build_reward_fn(
    weights: dict,
    judge_model,
    judge_tokenizer: Optional[PreTrainedTokenizer],
) -> Callable:
    """Build a TRL-compatible reward function.

    TRL GRPOTrainer expects a callable:
        reward_fn(completions: List[str]) -> List[float]

    The length reward is handled inside train_grpo.py since it needs token counts.
    This builder returns a function that computes the three text-based rewards.
    """
    def reward_fn(completions: List[str]) -> List[float]:
        return compute_reward(completions, weights, judge_model, judge_tokenizer)
    return reward_fn
