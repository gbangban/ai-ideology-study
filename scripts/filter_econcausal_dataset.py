#!/usr/bin/env python3
"""Filter grpo_train_merged.jsonl to EconCausal sources only.

Usage:
    python3 scripts/filter_econcausal_dataset.py
"""
import json
from pathlib import Path


def source_task(doc):
    """Return task identifier for breakdown."""
    source = doc.get("source", "unknown")
    if "/" in source:
        return source
    return source


def main():
    root = Path(__file__).resolve().parent.parent
    input_path = root / "data/processed/grpo_train_merged.jsonl"
    output_path = root / "data/processed/grpo_train_econcausal.jsonl"

    econcausal = []
    skipped = {"corr2cause": 0, "synthetic": 0, "other": 0}

    with open(input_path) as f:
        for line in f:
            doc = json.loads(line)
            source = doc.get("source", "unknown")
            if source.startswith("econcausal/"):
                econcausal.append(doc)
            elif source == "corr2cause":
                skipped["corr2cause"] += 1
            elif source == "synthetic":
                skipped["synthetic"] += 1
            else:
                skipped["other"] += 1

    with open(output_path, "w") as f:
        for doc in econcausal:
            f.write(json.dumps(doc) + "\n")

    print(f"Filtered: {len(econcausal)} EconCausal samples written to {output_path}")
    print(f"Skipped: corr2cause={skipped['corr2cause']}, synthetic={skipped['synthetic']}, other={skipped['other']}")

    task_counts = {}
    for doc in econcausal:
        task = source_task(doc)
        task_counts[task] = task_counts.get(task, 0) + 1
    print("Breakdown:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
