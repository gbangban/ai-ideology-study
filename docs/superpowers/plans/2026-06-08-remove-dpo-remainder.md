# Remove DPO Remainder Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead `split_sft_remainder` function and all DPO-remainder references from active code, simplifying `convert_full_dataset.py` to a single-purpose parquet-to-SFT converter.

**Architecture:** The remainder split was exclusively for feeding DPO training data. With DPO deprecated, the function and its tests are unused dead code. Simplify `convert_full_dataset.py` to: parquet -> clean traces -> save full dataset -> save SFT dataset (all records, no split). Update the data prep script and AGENTS.md accordingly.

**Tech Stack:** Python, pytest

---

### Task 1: Remove `split_sft_remainder` tests

**Files:**
- Modify: `src/tests/test_data_prep.py:7-41`

- [ ] **Step 1: Delete the `TestConvertFullDataset` class**

Remove lines 7-41 entirely (the entire class including `test_split_counts`, `test_no_id_overlap`, `test_type_balance`). The remaining classes (`TestBuildSFTDataset`, `TestRejectedResponses`) stay.

The file should start with:
```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBuildSFTDataset:
```

- [ ] **Step 2: Run remaining tests to confirm nothing breaks**

Run: `python3 -m pytest src/tests/test_data_prep.py -v`
Expected: 7 passed (4 build_sft + 3 rejected_responses)

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_data_prep.py
git commit -m "remove dead split_sft_remainder tests (DPO deprecated)"
```

### Task 2: Simplify `convert_full_dataset.py`

**Files:**
- Modify: `src/teacher/convert_full_dataset.py`

- [ ] **Step 1: Remove `split_sft_remainder` function and update docstring**

Replace lines 1-97 with:
```python
#!/usr/bin/env python3
"""
Convert full parquet dataset to JSON and clean reasoning traces.

Usage:
    python3 -m src.teacher.convert_full_dataset \
        --parquet-path /path/to/batch_00000.parquet \
        --output-dir data/processed/
"""

import argparse
import json
import sys
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
```

- [ ] **Step 2: Simplify `main()` to remove split logic**

Replace the `main()` function (lines 117-148) with:
```python
def main():
    parser = argparse.ArgumentParser(description="Convert full parquet dataset and clean reasoning traces")
    parser.add_argument("--parquet-path", default=PARQUET_PATH_DEFAULT, help="Path to parquet file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet_path}")
    records = parquet_to_records(args.parquet_path)
    print(f"  Loaded {len(records)} records")

    print("Cleaning reasoning traces...")
    records = clean_and_save(records, str(Path(args.output_dir) / "full_dataset.json"))

    sft_path = str(Path(args.output_dir) / "sft_dataset.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  SFT: {len(records)} samples -> {sft_path}")
```

Also remove the unused `Counter` import (line 14).

- [ ] **Step 3: Verify the module still imports cleanly**

Run: `python3 -c "from src.teacher.convert_full_dataset import parquet_to_records, clean_and_save; print('OK')"`
Expected: `OK`

Run: `python3 -c "from src.teacher.convert_full_dataset import split_sft_remainder" 2>&1`
Expected: `ImportError: cannot import name 'split_sft_remainder'`

- [ ] **Step 4: Commit**

```bash
git add src/teacher/convert_full_dataset.py
git commit -m "simplify convert_full_dataset: remove dead split_sft_remainder (DPO deprecated)"
```

### Task 3: Update `run_data_prep.sh`

**Files:**
- Modify: `scripts/run_data_prep.sh`

- [ ] **Step 1: Remove remainder-related comment and simplify**

Replace file contents with:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Full data prep pipeline:
# 1. Convert parquet to JSON, clean traces
# 2. Build trace-aligned SFT dataset

PARQUET_PATH="${PARQUET_PATH:-/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/recipe_ml-1500-v1/parquet-files/batch_00000.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"

echo "=== Step 1: Convert parquet, clean traces ==="
python3 -m src.teacher.convert_full_dataset \
    --parquet-path "$PARQUET_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 2: Build trace-aligned SFT dataset ==="
python3 -m src.teacher.build_sft_dataset \
    --input "$OUTPUT_DIR/sft_dataset.json" \
    --output "$OUTPUT_DIR/sft_dataset.jsonl"

echo ""
echo "=== Data prep complete ==="
echo "  SFT dataset: $OUTPUT_DIR/sft_dataset.jsonl"
```

Removed `SFT_COUNT` variable and `--sft-count` flag since they're no longer used.

- [ ] **Step 2: Verify script syntax**

Run: `bash -n scripts/run_data_prep.sh`
Expected: no output (valid syntax)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_data_prep.sh
git commit -m "update run_data_prep.sh: remove sft-count param (no more split)"
```

### Task 4: Update `AGENTS.md`

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `convert_full_dataset.py` description**

Replace:
```
  - `convert_full_dataset.py` - Parquet -> JSON, clean traces, split SFT/remainder
```
With:
```
  - `convert_full_dataset.py` - Parquet -> JSON, clean traces, save SFT dataset
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "update AGENTS.md: convert_full_dataset no longer splits"
```

### Task 5: Verify full test suite

**Files:** none

- [ ] **Step 1: Run full E2E test suite**

Run: `./scripts/run_e2e_tests.sh`
Expected: All tests pass (139 passed, 1 skipped — 4 fewer tests than before since we removed the 3 split tests, but the host numpy skip remains). Actually: 139 passed total = 49 teacher + 19 sft + 6 sglang + 11 grpo + 11 rewards + 35 rlvmr + 3 e2e + 7 data_prep + 3 grpo_config + 6 data_prep = wait, let me just run it and confirm.

- [ ] **Step 2: Commit if needed**

```bash
git add -A
git commit -m "verify: test suite passes after DPO remainder removal"
```

## Self-Review

**Spec coverage:**
- Remove `split_sft_remainder` function? Task 2 covers it.
- Remove tests for the function? Task 1 covers it.
- Update data prep script? Task 3 covers it.
- Update AGENTS.md? Task 4 covers it.
- Verify tests pass? Task 5 covers it.

**Placeholder scan:** No TBDs, no "add validation later", no "similar to Task N". Every step has exact code or exact deletion instructions.

**Type consistency:** `clean_and_save` and `parquet_to_records` remain unchanged — only `split_sft_remainder` is removed. No signature changes to existing functions.
