#!/usr/bin/env python3
"""
Convert full parquet dataset to JSON, clean reasoning traces, and split SFT/DPO.

Usage:
    python3 -m src.teacher.convert_full_dataset \
        --parquet-path /path/to/batch_00000.parquet \
        --output-dir data/processed/
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.teacher.clean_reasoning_traces import clean_reasoning_trace


PARQUET_PATH_DEFAULT = (
    "/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/"
    "recipe_ml-1500-v1/parquet-files/batch_00000.parquet"
)


def parquet_to_records(parquet_path: str) -> list[dict]:
    """Read parquet and return list of dicts with native Python types."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if hasattr(val, "item"):
                val = val.item()
            record[col] = val if pd.notna(val) else ""
        records.append(record)
    return records


def split_sft_dpo(records: list[dict], sft_count: int = 1250, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Split records into SFT and DPO sets with balanced type distribution.

    Strategy: stratified random selection — pick DPO samples proportionally
    from each type, then remaining go to SFT.

    Returns (sft_records, dpo_records).
    """
    import random
    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for r in records:
        t = r.get("type", "A")
        by_type.setdefault(t, []).append(r)

    # Calculate DPO quota per type (proportional)
    total = len(records)
    dpo_quota: dict[str, int] = {}
    dpo_assigned = 0
    types_sorted = sorted(by_type.keys())

    for t in types_sorted:
        proportion = len(by_type[t]) / total
        quota = round(proportion * (total - sft_count))
        dpo_quota[t] = quota
        dpo_assigned += quota

    # Adjust to hit exact target
    target_dpo = total - sft_count
    while dpo_assigned > target_dpo:
        largest = max(dpo_quota, key=dpo_quota.get)
        dpo_quota[largest] -= 1
        dpo_assigned -= 1
    while dpo_assigned < target_dpo:
        remaining = {t: len(by_type[t]) - dpo_quota.get(t, 0) for t in types_sorted}
        largest_remaining = max(remaining, key=remaining.get)
        dpo_quota[largest_remaining] = dpo_quota.get(largest_remaining, 0) + 1
        dpo_assigned += 1

    # Select DPO samples
    dpo_records = []
    for t in types_sorted:
        pool = by_type[t][:]
        rng.shuffle(pool)
        dpo_records.extend(pool[:dpo_quota.get(t, 0)])

    # Remaining go to SFT
    dpo_ids = {r["id"] for r in dpo_records}
    sft_records = [r for r in records if r["id"] not in dpo_ids]

    return sft_records, dpo_records


def clean_and_save(records: list[dict], output_path: str) -> list[dict]:
    """Clean reasoning traces in-place and save to JSON."""
    cleaned = 0
    for r in records:
        rc = r.get("answer__reasoning_content", "")
        if rc:
            r["answer__reasoning_content"] = clean_reasoning_trace(rc)
            cleaned += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Cleaned {cleaned}/{len(records)} reasoning traces")
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert full parquet dataset, clean traces, split SFT/DPO")
    parser.add_argument("--parquet-path", default=PARQUET_PATH_DEFAULT, help="Path to parquet file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--sft-count", type=int, default=1250, help="Number of SFT samples")
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet_path}")
    records = parquet_to_records(args.parquet_path)
    print(f"  Loaded {len(records)} records")

    print("Cleaning reasoning traces...")
    records = clean_and_save(records, str(Path(args.output_dir) / "full_dataset.json"))

    print(f"Splitting {len(records)} records: {args.sft_count} SFT + {len(records) - args.sft_count} DPO")
    sft, dpo = split_sft_dpo(records, sft_count=args.sft_count)

    sft_path = str(Path(args.output_dir) / "sft_dataset.json")
    dpo_path = str(Path(args.output_dir) / "dpo_dataset.json")

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft, f, indent=2, ensure_ascii=False)
    print(f"  SFT: {len(sft)} samples -> {sft_path}")

    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo, f, indent=2, ensure_ascii=False)
    print(f"  DPO: {len(dpo)} samples -> {dpo_path}")

    # Type distribution report
    sft_types = Counter(r["type"] for r in sft)
    dpo_types = Counter(r["type"] for r in dpo)
    print(f"\n  SFT type distribution: {dict(sorted(sft_types.items()))}")
    print(f"  DPO type distribution: {dict(sorted(dpo_types.items()))}")


if __name__ == "__main__":
    main()
