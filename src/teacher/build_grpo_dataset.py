#!/usr/bin/env python3
"""
GRPO Dataset Builder

Assembles all synthetic causal reasoning data into a single GRPO-compatible dataset.
"""

import json
import random
from pathlib import Path
from typing import List

from src.teacher.generate_causal_graphs import generate_causal_graph_dataset
from src.teacher.generate_context_flips import generate_context_flip_pairs
from src.teacher.generate_null_effects import generate_null_effect_dataset
from src.teacher.generate_contradiction_pairs import generate_contradiction_dataset


def assemble_grpo_dataset(
    causal_graph_general: int = 100,
    causal_graph_economic: int = 100,
    context_flips: int = 140,
    null_effect_economic: int = 60,
    null_effect_general: int = 40,
    contradiction_economic: int = 80,
    seed: int = 42,
) -> List[dict]:
    random.seed(seed)
    all_items = []

    all_items.extend(generate_causal_graph_dataset(n_samples=causal_graph_general, mode="general", seed=seed))
    all_items.extend(generate_causal_graph_dataset(n_samples=causal_graph_economic, mode="economic", seed=seed + 1000))

    flips = generate_context_flip_pairs(n_pairs=context_flips, seed=seed + 2000)
    for flip in flips:
        all_items.append({"prompt": flip["prompt_a"], "category": "context_flip", "subcategory": "economic", "context": flip["context_a"], "pair_key": f"{flip['treatment']}_{flip['outcome']}"})
        all_items.append({"prompt": flip["prompt_b"], "category": "context_flip", "subcategory": "economic", "context": flip["context_b"], "pair_key": f"{flip['treatment']}_{flip['outcome']}"})

    all_items.extend(generate_null_effect_dataset(n_samples=null_effect_economic, mode="economic", seed=seed + 3000))
    all_items.extend(generate_null_effect_dataset(n_samples=null_effect_general, mode="general", seed=seed + 4000))
    all_items.extend(generate_contradiction_dataset(n_samples=contradiction_economic, mode="economic", seed=seed + 5000))

    random.shuffle(all_items)
    return all_items


def save_grpo_dataset(dataset: List[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(dataset)} prompts to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/grpo_causal_dataset.jsonl")
    parser.add_argument("--causal-graph-general", type=int, default=100)
    parser.add_argument("--causal-graph-economic", type=int, default=100)
    parser.add_argument("--context-flips", type=int, default=140)
    parser.add_argument("--null-effect-economic", type=int, default=60)
    parser.add_argument("--null-effect-general", type=int, default=40)
    parser.add_argument("--contradiction-economic", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = assemble_grpo_dataset(
        causal_graph_general=args.causal_graph_general,
        causal_graph_economic=args.causal_graph_economic,
        context_flips=args.context_flips,
        null_effect_economic=args.null_effect_economic,
        null_effect_general=args.null_effect_general,
        contradiction_economic=args.contradiction_economic,
        seed=args.seed,
    )
    save_grpo_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
