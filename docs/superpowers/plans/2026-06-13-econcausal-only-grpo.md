# EconCausal-Only GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter GRPO training data to EconCausal-only (2943 samples), fix v4 `num_generations=2` -> `4`, and update both v3/v4 configs to use the new dataset.

**Architecture:** One-shot data filter script to create `grpo_train_econcausal.jsonl`, two config file edits (dataset paths + G=4 fix), one test update for the G=2 -> G=4 change. No training script changes needed — they read `dataset_path` from config defaults.

**Tech Stack:** Python, JSONL data format, TRL `GRPOConfig`.

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/filter_econcausal_dataset.py` | Create | One-shot script to filter `grpo_train_merged.jsonl` to EconCausal sources |
| `data/processed/grpo_train_econcausal.jsonl` | Create | Filtered dataset (2943 EconCausal samples) |
| `src/student/grpo_config_outcome.py` | Modify | Update `dataset_path` to EconCausal-only |
| `src/student/grpo_config_process.py` | Modify | Update `dataset_path` + fix `num_generations=2` -> `4` |
| `src/tests/test_grpo_config.py` | Modify | Update `test_grpo_config_process_defaults_unchanged` G=2 -> G=4 |

---

### Task 1: Create EconCausal dataset filter script

**Files:**
- Create: `scripts/filter_econcausal_dataset.py`

- [ ] **Step 1: Write the filter script**

```python
#!/usr/bin/env python3
"""Filter grpo_train_merged.jsonl to EconCausal sources only.

Usage:
    python3 scripts/filter_econcausal_dataset.py
"""
import json
from pathlib import Path

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


def source_task(doc):
    """Return task identifier for breakdown."""
    source = doc.get("source", "unknown")
    if "/" in source:
        return source
    return source


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the filter script**

Run: `python3 scripts/filter_econcausal_dataset.py`
Expected: `Filtered: 2943 EconCausal samples written to ...`
Expected breakdown: task1_econ=947, task1_finance=860, task2=284, task3=852

- [ ] **Step 3: Verify output**

Run: `wc -l data/processed/grpo_train_econcausal.jsonl`
Expected: `2943 data/processed/grpo_train_econcausal.jsonl`

Run: `python3 -c "import json; sources=set(); [sources.add(json.loads(l)['source']) for l in open('data/processed/grpo_train_econcausal.jsonl')]; print(sources)"`
Expected: `{'econcausal/task1_econ', 'econcausal/task1_finance', 'econcausal/task2', 'econcausal/task3'}`

- [ ] **Step 4: Commit**

```bash
git add scripts/filter_econcausal_dataset.py data/processed/grpo_train_econcausal.jsonl
git commit -m "data: filter EconCausal-only training dataset (2943 samples)

- scripts/filter_econcausal_dataset.py: one-shot filter from grpo_train_merged.jsonl
- data/processed/grpo_train_econcausal.jsonl: 2943 EconCausal samples
  (task1_econ=947, task1_finance=860, task2=284, task3=852)
- Drops 4999 Corr2Cause (already solved at 74.6% via SFT)
  and 460 synthetic (no ground truth)"
```

---

### Task 2: Update v3 outcome config for EconCausal-only data

**Files:**
- Modify: `src/student/grpo_config_outcome.py:31`

- [ ] **Step 1: Update dataset_path in DEFAULT_CONFIG**

Change line 31 from:
```python
    "dataset_path": str(_project_root() / "data/processed/grpo_train_merged.jsonl"),
```
to:
```python
    "dataset_path": str(_project_root() / "data/processed/grpo_train_econcausal.jsonl"),
```

- [ ] **Step 2: Update module docstring**

Change lines 3-4 from:
```
Outcome-only GRPO training on EconCausal + Corr2Cause + synthetic data.
Flat advantage: single group-relative normalization of outcome rewards.
```
to:
```
Outcome-only GRPO training on EconCausal data only (2943 samples).
Corr2Cause removed (solved via SFT at 74.6%), synthetic removed (no ground truth).
```

- [ ] **Step 3: Verify with existing test**

Run: `python3 -m pytest src/tests/test_grpo_config.py::test_grpo_config_outcome_defaults_unchanged -v`
Expected: PASS (test checks max_steps, save_steps, logging_steps, max_completion_length, num_generations — none changed for v3)

- [ ] **Step 4: Commit**

```bash
git add src/student/grpo_config_outcome.py
git commit -m "config(v3): switch to EconCausal-only dataset path

- grpo_config_outcome.py: dataset_path -> grpo_train_econcausal.jsonl
- Updated docstring to reflect EconCausal-only training scope"
```

---

### Task 3: Fix v4 process config G=2 -> G=4 and update dataset path

**Files:**
- Modify: `src/student/grpo_config_process.py`

- [ ] **Step 1: Fix num_generations in create_grpo_config**

Change line 101 from:
```python
        num_generations=2,
```
to:
```python
        num_generations=4,
```

- [ ] **Step 2: Update dataset_path in DEFAULT_CONFIG**

Change line 58 from:
```python
    "dataset_path": str(_project_root() / "data/processed/grpo_train_merged.jsonl"),
```
to:
```python
    "dataset_path": str(_project_root() / "data/processed/grpo_train_econcausal.jsonl"),
```

- [ ] **Step 3: Update module docstring**

Change lines 4-5 from:
```
Dual-advantage GRPO training with outcome + process rewards, KL regularization,
and RLVMR tagged output format.
```
to:
```
Dual-advantage GRPO training on EconCausal data only (2943 samples).
Corr2Cause removed (solved via SFT), synthetic removed (no ground truth).
```

- [ ] **Step 4: Commit**

```bash
git add src/student/grpo_config_process.py
git commit -m "config(v4): fix num_generations G=4, switch to EconCausal-only dataset

- grpo_config_process.py: num_generations=2 -> 4 (matches DEFAULT_CONFIG grpo_g=4)
- dataset_path -> grpo_train_econcausal.jsonl
- Updated docstring to reflect EconCausal-only training scope"
```

---

### Task 4: Update test for v4 G=4 default

**Files:**
- Modify: `src/tests/test_grpo_config.py:84`

- [ ] **Step 1: Update test_grpo_config_process_defaults_unchanged**

Change line 84 from:
```python
    assert config.num_generations == 2
```
to:
```python
    assert config.num_generations == 4
```

- [ ] **Step 2: Run all config tests**

Run: `python3 -m pytest src/tests/test_grpo_config.py -v`
Expected: All tests PASS (including updated `test_grpo_config_process_defaults_unchanged`)

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_grpo_config.py
git commit -m "test: update v4 process config test for G=4 default

- test_grpo_config.py: num_generations assertion 2 -> 4 to match
  grpo_config_process.py fix"
```

---

### Task 5: Final verification

**Files:**
- Verify all changes work together

- [ ] **Step 1: Run full test suite**

Run: `./scripts/run_e2e_tests.sh`
Expected: All non-GPU tests pass. Known failures: `test_memory_profiler.py` VRAM assertions when GPU active, `test_evals.py` missing task_configs (pre-existing).

- [ ] **Step 2: Verify dataset is correct**

Run: `python3 -c "
import json
with open('data/processed/grpo_train_econcausal.jsonl') as f:
    docs = [json.loads(l) for l in f]
print(f'Total: {len(docs)}')
sources = {}
for d in docs:
    s = d.get('source', 'unknown')
    sources[s] = sources.get(s, 0) + 1
for s, c in sorted(sources.items()):
    print(f'  {s}: {c}')
# Verify no non-EconCausal samples
non_ec = [d for d in docs if not d.get('source','').startswith('econcausal/')]
assert not non_ec, f'Found {len(non_ec)} non-EconCausal samples'
print('PASS: All samples are EconCausal')
"`

- [ ] **Step 3: Verify configs point to correct dataset**

Run: `python3 -c "
from src.student.grpo_config_outcome import DEFAULT_CONFIG as v3
from src.student.grpo_config_process import DEFAULT_CONFIG as v4
assert 'econcausal' in v3['dataset_path'], f'v3 wrong: {v3[\"dataset_path\"]}'
assert 'econcausal' in v4['dataset_path'], f'v4 wrong: {v4[\"dataset_path\"]}'
print(f'v3 dataset: {v3[\"dataset_path\"]}')
print(f'v4 dataset: {v4[\"dataset_path\"]}')
"`

Run: `python3 -c "
from src.student.grpo_config_process import create_grpo_config
cfg = create_grpo_config(output_dir='/tmp/test')
assert cfg.num_generations == 4, f'G={cfg.num_generations}'
print(f'v4 G={cfg.num_generations} (correct)')
"`

- [ ] **Step 4: Final commit (if any verification fixes were needed)**

If all checks pass, no additional commit needed. If any fixes were made:
```bash
git add -A
git commit -m "fix: verification corrections from final checks"
```
