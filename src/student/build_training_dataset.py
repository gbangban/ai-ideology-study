"""
Dataset loading pipeline for GRPO v3/v4 training.

Loads EconCausal (all tasks), Corr2Cause (sampled), and synthetic data.
Produces a unified JSONL file with standardized fields.

Usage:
    python3 -m src.student.build_training_dataset \
        --output data/processed/grpo_train_merged.jsonl \
        --corr2cause-sample 5000
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def load_econcausal(config: str) -> list:
    """Load an EconCausal task config."""
    ds = load_dataset("qwqw3535/econcausal-benchmark", config, trust_remote_code=True)
    records = []
    for row in ds["train"]:
        records.append({
            "prompt": row["question"],
            "answer": row["answer"],
            "dataset_type": "econcausal",
            "source": f"econcausal/{config}",
            "treatment": row.get("treatment", ""),
            "outcome": row.get("outcome", ""),
            "context": row.get("context", ""),
        })
    return records


def load_corr2cause(sample_size: int = 5000, seed: int = 42) -> list:
    """Load Corr2Cause with stratified sampling."""
    ds = load_dataset("tasksource/corr2cause", trust_remote_code=True)
    train = ds["train"]

    # Stratified sampling by relation
    by_relation = {"entailment": [], "contradiction": [], "neutral": []}
    for row in train:
        rel = row["relation"]
        if rel in by_relation:
            by_relation[rel].append(row)

    random.seed(seed)
    sampled = []
    for rel, rows in by_relation.items():
        count = int(sample_size * len(rows) / len(train))
        sampled.extend(random.sample(rows, min(count, len(rows))))

    records = []
    for row in sampled:
        records.append({
            "prompt": _build_corr2cause_prompt(row["premise"], row["hypothesis"]),
            "relation": row["relation"],
            "dataset_type": "corr2cause",
            "source": "corr2cause",
            "id": row.get("id", ""),
        })
    return records


def _build_corr2cause_prompt(premise: str, hypothesis: str) -> str:
    """Build a Corr2Cause prompt for the model."""
    return (
        f"# Task\nGiven a premise describing statistical relations among variables, "
        f"determine whether the hypothesis is entailed (True), contradicted (False), "
        f"or undetermined (answer either True or False based on your best judgment).\n\n"
        f"# Premise\n{premise}\n\n"
        f"# Hypothesis\n{hypothesis}\n\n"
        f"Answer True or False."
    )


def load_synthetic(path: str) -> list:
    """Load synthetic dataset, excluding DAG questions."""
    records = []
    with open(path, "r") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("category") == "causal_graph":
                continue
            records.append({
                "prompt": doc["prompt"],
                "answer": doc.get("answer", ""),
                "dataset_type": "synthetic",
                "source": "synthetic",
                "category": doc.get("category", ""),
                "subcategory": doc.get("subcategory", ""),
            })
    return records


def build_merged_dataset(
    output_path: str,
    corr2cause_sample: int = 5000,
    synthetic_path: str = "data/processed/grpo_causal_dataset.jsonl",
    seed: int = 42,
) -> int:
    """Build the merged training dataset.

    Returns total number of records.
    """
    random.seed(seed)

    all_records = []

    # EconCausal (all tasks)
    for config in ["task1_econ", "task1_finance", "task2", "task3"]:
        records = load_econcausal(config)
        all_records.extend(records)
        print(f"  EconCausal [{config}]: {len(records)} prompts")

    # Corr2Cause (sampled)
    records = load_corr2cause(corr2cause_sample, seed)
    all_records.extend(records)
    print(f"  Corr2Cause (sampled): {len(records)} prompts")

    # Synthetic (non-DAG)
    records = load_synthetic(synthetic_path)
    all_records.extend(records)
    print(f"  Synthetic (non-DAG): {len(records)} prompts")

    # Shuffle
    random.shuffle(all_records)

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"  Total: {len(all_records)} prompts -> {output_path}")

    # Distribution summary
    from collections import Counter
    dist = Counter(r["dataset_type"] for r in all_records)
    print(f"  Distribution: {dict(dist)}")

    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description="Build merged GRPO training dataset")
    parser.add_argument(
        "--output",
        default="data/processed/grpo_train_merged.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--corr2cause-sample",
        type=int,
        default=5000,
        help="Number of Corr2Cause samples to include",
    )
    parser.add_argument(
        "--synthetic-path",
        default="data/processed/grpo_causal_dataset.jsonl",
        help="Path to synthetic dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = build_merged_dataset(
        args.output,
        args.corr2cause_sample,
        args.synthetic_path,
        args.seed,
    )
    print(f"Done. {count} records written.")


if __name__ == "__main__":
    main()
