#!/usr/bin/env python3
"""
Contradiction Pair Generator

Generates prompts asking the model to produce both supporting and opposing
reasoning for the same causal claim. Reward is structural symmetry.
"""

import random
from typing import Dict, List, Optional


ECONOMIC_CLAIMS = [
    "Free trade agreements reduce income inequality within participating countries.",
    "Central bank independence improves long-term economic stability.",
    "Foreign direct investment always benefits the host country's economy.",
    "Deregulation of financial markets leads to higher economic growth.",
    "Progressive taxation reduces overall economic efficiency.",
    "Government deficit spending stimulates private sector investment.",
    "Labor market flexibility reduces long-term unemployment.",
    "Trade protectionism benefits domestic manufacturing employment.",
    "Privatization of public services improves service quality.",
    "Currency devaluation improves a country's trade balance.",
    "Capital controls prevent financial crises.",
    "Corporate welfare policies create sustainable jobs.",
    "Monetary policy is more effective than fiscal policy during recessions.",
    "Economic sanctions achieve their intended political objectives.",
    "Housing market deregulation improves housing affordability.",
]

GENERAL_CLAIMS = [
    "Increasing education spending always improves student outcomes.",
    "Technology adoption always increases labor productivity.",
    "Urbanization leads to higher average income levels.",
    "Population growth drives economic development.",
    "Market competition always reduces consumer prices.",
    "Government intervention in markets reduces economic welfare.",
    "International cooperation reduces the likelihood of conflict.",
    "Social media increases political engagement.",
    "Automation always leads to net job creation.",
    "Free markets naturally reduce poverty over time.",
]


def generate_contradiction_prompt(mode: str = "economic", seed: Optional[int] = None) -> Dict:
    if seed is not None:
        random.seed(seed)
    claims = ECONOMIC_CLAIMS if mode == "economic" else GENERAL_CLAIMS
    claim = random.choice(claims)
    return {
        "prompt": f'Consider the following causal claim: "{claim}"\n\nFirst, provide reasoning that SUPPORTS this claim. What mechanisms would make this true?\n\nThen, provide reasoning that OPPOSES this claim. What mechanisms would make this false?\n\nFinally, give your assessed directional effect: positive (+), negative (-), null (0), or mixed. Commit to a single answer.',
        "claim": claim,
        "category": "contradiction_pair",
        "subcategory": mode,
    }


def generate_contradiction_dataset(n_samples: int = 100, mode: str = "economic", seed: int = 42) -> List[Dict]:
    random.seed(seed)
    return [generate_contradiction_prompt(mode=mode, seed=seed + i) for i in range(n_samples)]


def main():
    import argparse, json, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--mode", choices=["economic", "general"], default="economic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = generate_contradiction_dataset(n_samples=args.n_samples, mode=args.mode, seed=args.seed)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(dataset)} contradiction pair prompts -> {args.output}")


if __name__ == "__main__":
    main()
