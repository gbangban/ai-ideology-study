#!/usr/bin/env python3
"""
DPO Pair Generation

Generate preference pairs for DPO training from SFT dataset and rejected responses.
Each question produces 3 pairs (one per rejection type), interleaved.

Usage:
    python3 -m src.teacher.generate_dpo_pairs \
        --dpo-data data/processed/dpo_dataset.json \
        --rejections data/processed/rejected_responses.jsonl \
        --output data/processed/dpo_pairs.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_rejections(path: str) -> dict[int, list[dict]]:
    """Load rejected responses indexed by question ID."""
    rejections_by_id: dict[int, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                qid = record["id"]
                rejections_by_id[qid] = record["rejections"]
    return rejections_by_id


def generate_interleaved_pairs(records: list[dict], rejections: list[dict]) -> list[dict]:
    """
    Generate DPO pairs: one per rejection type per question.

    rejections is a list of {"id": ..., "rejections": [...]} records.

    Returns list of dicts with keys: prompt, chosen, rejected, rejection_type.
    Pairs are shuffled to interleave rejection types.
    """
    rejections_by_id: dict[int, list[dict]] = {}
    for rec in rejections:
        rejections_by_id[rec["id"]] = rec["rejections"]

    pairs = []
    for record in records:
        qid = record["id"]
        question = record["question"]
        chosen = record["answer"]
        rejections_for_q = rejections_by_id.get(qid, [])

        for rej in rejections_for_q:
            pairs.append({
                "prompt": question,
                "chosen": chosen,
                "rejected": rej["content"],
                "rejection_type": rej["type"],
            })

    # Shuffle to interleave rejection types
    random.seed(42)
    random.shuffle(pairs)
    return pairs


def save_dpo_pairs(pairs: list[dict], output_path: str):
    """Save DPO pairs to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} DPO pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DPO pairs from dataset and rejections")
    parser.add_argument("--dpo-data", default="data/processed/dpo_dataset.json", help="DPO dataset JSON")
    parser.add_argument("--rejections", default="data/processed/rejected_responses.jsonl", help="Rejected responses JSONL")
    parser.add_argument("--output", default="data/processed/dpo_pairs.jsonl", help="Output DPO pairs JSONL")
    args = parser.parse_args()

    with open(args.dpo_data, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} DPO questions")

    rejections_by_id = load_rejections(args.rejections)
    print(f"Loaded rejections for {len(rejections_by_id)} questions")

    # Build list format for generate_interleaved_pairs
    rejections_list = [{"id": qid, "rejections": rejs} for qid, rejs in rejections_by_id.items()]
    pairs = generate_interleaved_pairs(records, rejections_list)
    save_dpo_pairs(pairs, args.output)


if __name__ == "__main__":
    main()
