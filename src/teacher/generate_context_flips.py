#!/usr/bin/env python3
"""
Context-Flipping Prompt Generator

Generates paired economic prompts with different institutional contexts.
Training signal is structural: model must give two distinct directional answers.
"""

import random
from typing import Dict, List, Optional


TREATMENT_OUTCOME_PAIRS = [
    {"treatment": "central bank interest rate increase", "outcome": "working-class household debt burden"},
    {"treatment": "minimum wage increase", "outcome": "small business employment levels"},
    {"treatment": "trade tariff imposition", "outcome": "domestic manufacturing output"},
    {"treatment": "financial deregulation", "outcome": "systemic banking risk"},
    {"treatment": "corporate tax reduction", "outcome": "worker wage growth"},
    {"treatment": "public infrastructure investment", "outcome": "private sector investment levels"},
    {"treatment": "labor union legalization", "outcome": "corporate profit margins"},
    {"treatment": "capital gains tax increase", "outcome": "venture capital investment"},
    {"treatment": "universal basic income program", "outcome": "labor force participation rate"},
    {"treatment": "housing price controls", "outcome": "housing market liquidity"},
    {"treatment": "bank lending restrictions", "outcome": "economic growth rate"},
    {"treatment": "pension fund privatization", "outcome": "retirement income security"},
    {"treatment": "agricultural subsidy removal", "outcome": "rural employment levels"},
    {"treatment": "healthcare privatization", "outcome": "population health outcomes"},
    {"treatment": "education voucher system", "outcome": "educational inequality"},
]

CONTEXT_TEMPLATES = [
    {"label": "financialized_market", "context": "In a highly financialized economy with weak labor protections, deregulated capital markets, and concentrated ownership of productive assets"},
    {"label": "state_directed", "context": "Under a state-directed economic system with strict capital controls, public ownership of strategic sectors, and strong labor institutions"},
    {"label": "mixed_economy", "context": "In a mixed economy with moderate regulation, competitive markets in most sectors, and a strong social safety net"},
    {"label": "developing_economy", "context": "In a developing economy with informal labor markets, limited financial infrastructure, and high income inequality"},
    {"label": "post_crisis", "context": "In a post-financial-crisis economy with heightened regulatory scrutiny, weakened consumer confidence, and elevated public debt"},
    {"label": "high_automation", "context": "In an economy with high automation levels where routine labor has been largely displaced by machines and algorithms"},
]

PROMPT_TEMPLATE = """{context}, what is the directional effect of {treatment} on {outcome}?

Answer with a single directional claim: positive (+), negative (-), null (0), or mixed. Provide brief structural reasoning for your answer."""


def generate_context_flip_pair(
    treatment_outcome: Optional[Dict] = None,
    context_a: Optional[Dict] = None,
    context_b: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    if seed is not None:
        random.seed(seed)
    if treatment_outcome is None:
        treatment_outcome = random.choice(TREATMENT_OUTCOME_PAIRS)
    if context_a is None or context_b is None:
        contexts = random.sample(CONTEXT_TEMPLATES, 2)
        if context_a is None:
            context_a = contexts[0]
        if context_b is None:
            context_b = contexts[1]
    if context_a["label"] == context_b["label"]:
        return None

    return {
        "prompt_a": PROMPT_TEMPLATE.format(context=context_a["context"], treatment=treatment_outcome["treatment"], outcome=treatment_outcome["outcome"]),
        "prompt_b": PROMPT_TEMPLATE.format(context=context_b["context"], treatment=treatment_outcome["treatment"], outcome=treatment_outcome["outcome"]),
        "context_a": context_a["label"],
        "context_b": context_b["label"],
        "treatment": treatment_outcome["treatment"],
        "outcome": treatment_outcome["outcome"],
        "category": "context_flip",
    }


def generate_context_flip_pairs(n_pairs: int = 200, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    pairs = []
    for i in range(n_pairs):
        to = TREATMENT_OUTCOME_PAIRS[i % len(TREATMENT_OUTCOME_PAIRS)]
        pair = generate_context_flip_pair(treatment_outcome=to, seed=seed + i)
        if pair:
            pairs.append(pair)
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]


def main():
    import argparse, json, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = generate_context_flip_pairs(n_pairs=args.n_pairs, seed=args.seed)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Generated {len(pairs)} context-flip pairs -> {args.output}")


if __name__ == "__main__":
    main()
