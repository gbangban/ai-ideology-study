#!/usr/bin/env python3
"""
Null-Effect (Orthogonality) Prompt Generator

Generates prompts with genuinely decoupled variables.
Model should learn to explicitly declare null causal relationships.
"""

import random
from typing import Dict, List, Optional


ECONOMIC_NULL_PAIRS = [
    ("consumer preference for organic coffee in Colombia", "sovereign debt default risk of Kazakhstan"),
    ("chocolate prices in Switzerland", "unemployment rate in Nigerian manufacturing"),
    ("wine export volume from Bordeaux", "tech startup funding in Bangalore"),
    ("local tourism revenue in Venice", "corporate bond yields in Tokyo"),
    ("fishery catch in Newfoundland", "housing prices in Mumbai"),
    ("textile import tariffs in Bangladesh", "venture capital in Silicon Valley"),
    ("copper mining output in Chile", "university enrollment rates in Germany"),
    ("agricultural subsidies in France", "stock market volatility in Seoul"),
    ("fishing regulations in Alaska", "pension fund returns in Brazil"),
    ("cotton prices in Texas", "infrastructure spending in South Korea"),
]

GENERAL_NULL_PAIRS = [
    ("the number of books published about swallows", "global average temperature"),
    ("ice cream sales in Norway", "earthquake frequency in Japan"),
    ("shoe sizes of professional musicians", "stock market returns in emerging markets"),
    ("rainfall in the Sahara desert", "literacy rates in Finland"),
    ("popularity of jazz music", "ocean current speeds in the Pacific"),
    ("number of libraries per capita", "volcanic activity in Iceland"),
    ("average height of basketball players", "inflation rate in Argentina"),
    ("coffee consumption in Brazil", "traffic accidents in London"),
    ("number of museums per city", "average commute time in rural areas"),
    ("adoption rate of electric vehicles", "birth rates in Scandinavia"),
]


def generate_null_effect_prompt(mode: str = "economic", seed: Optional[int] = None) -> Dict:
    if seed is not None:
        random.seed(seed)
    pairs = ECONOMIC_NULL_PAIRS if mode == "economic" else GENERAL_NULL_PAIRS
    var_a, var_b = random.choice(pairs)
    return {
        "prompt": f"Analyze the causal impact of a change in {var_a} on {var_b}. What is the directional effect? Answer with positive (+), negative (-), null (0), or mixed. Provide structural reasoning for your answer.",
        "answer": "null",
        "var_a": var_a,
        "var_b": var_b,
        "category": "null_effect",
        "subcategory": mode,
    }


def generate_null_effect_dataset(n_samples: int = 100, mode: str = "economic", seed: int = 42) -> List[Dict]:
    random.seed(seed)
    return [generate_null_effect_prompt(mode=mode, seed=seed + i) for i in range(n_samples)]


def main():
    import argparse, json, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--mode", choices=["economic", "general"], default="economic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = generate_null_effect_dataset(n_samples=args.n_samples, mode=args.mode, seed=args.seed)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(dataset)} null-effect prompts -> {args.output}")


if __name__ == "__main__":
    main()
