# DPO References Inventory

Generated: 2026-06-08
Purpose: Complete audit before DPO removal

## Code Files (Must Remove)

### `src/teacher/convert_full_dataset.py`
- Line 3: docstring "split SFT/remainder"
- Line 44-97: `split_sft_remainder()` function — entire function, only exists to feed DPO training
- Line 118: argparse description "split SFT/remainder"
- Line 131-144: main() calls split, writes remainder report

### `src/tests/test_data_prep.py`
- Lines 10-41: `TestConvertFullDataset` class — 3 tests for `split_sft_remainder`
  - `test_split_counts`
  - `test_no_id_overlap`
  - `test_type_balance`

### `scripts/run_data_prep.sh`
- Line 5: comment "split SFT/remainder"

## Code Files (Harmless Text, No Action)

### `evals/scripts/compare_results.py`
- Line 316: `"Consider DPO remediation if regressions exceed acceptable levels."` — string literal in output

### `scripts/test_wandb_endpoints.py` / `test_wandb_auth2.py`
- "endpoint" matches, not DPO

## Documentation (Historical, Superseded — No Action)

### `AGENTS.md`
- Line 40: "DPO is deprecated" (correct, intentional)
- Line 55: `convert_full_dataset.py` description mentions "split SFT/remainder"

### `README.md`
- Line 3: "custom DPO/GRPO training"
- Line 12: `train_dpo.py` reference
- Line 46: "GRPO/DPO training container"
- Line 74-77: DPO Training section
- Line 151: "teacher, SFT config, DPO, SG-Lang client, GRPO training, E2E integration"
- Line 168: `dpo_pairs.jsonl` reference

### `revisions.md`
- Lines 65, 112-124: Historical DPO section + paper references

### `evals/results/README.md`
- Line 328: "Any post-SFT alignment method (GRPO, DPO, rejection sampling)..."

### `docs/superpowers/plans/` (superseded plans)
- `2026-06-03-sglang-integration.md` — DPO test references
- `2026-06-04-grpo-v3-verifiable-causal-reasoning.md` — SFT+DPO mentions
- `2026-06-08-grpo-unsloth-migration.md` — DPO test reference

### `docs/superpowers/specs/` (superseded specs)
- `2026-06-04-grpo-replacement-design.md` — DPO approach design
- `2026-06-04-post-sft-alignment-design.md` — DPO approach design

### `docs/grpo-experimental-design-oom-debug-synopsis.md`
- Line 281: "SFT+DPO on DM-aligned data..."

### `docs/grpo-v3-proposal-summary.md`
- Line 92: "SFT+DPO pipeline improved..."

### `docs/paper_synthesis_hedging_rlvmr.md`
- Line 51: "SFT > DPO for ideological shift"

### `papers/` directory
- Paper notes and full papers — historical references, not our code

### `.kilo/plans/1780248379670-kind-forest.md`
- Option D: "Abandon GRPO, return to DPO" — superseded plan

## Key Decision

The `split_sft_remainder` function and its tests exist solely to partition data for DPO training. With DPO removed, the function is dead code. The `convert_full_dataset.py` script should be simplified to just: parquet -> JSON, clean traces, save SFT dataset (no split).
