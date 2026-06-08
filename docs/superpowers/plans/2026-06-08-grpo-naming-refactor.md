# Plan: GRPO File Naming Refactor (v1/v2/v3/v4 Semantic Naming)

**Date**: 2026-06-08
**Status**: Ready for execution

## Goal
Rename all GRPO training files to use semantic version labels (v1/v2=DM keyword, v3=outcome, v4=process) with 1:1:1 mapping of rewards:config:training per track.

## Version Mapping

| Version | Track | Rewards | Config | Training |
|---------|-------|---------|--------|----------|
| v1/v2 | DM keyword | `reward_dm.py` | `grpo_config_dm.py` | `train_grpo_dm.py` |
| v3 | Outcome | `reward_outcome.py` | `grpo_config_outcome.py` | `train_grpo_outcome.py` |
| v4 | Process | `reward_process.py` | `grpo_config_process.py` | `train_grpo_process.py` |

## Current -> New File Mapping

| Current File | New File | Action |
|-------------|----------|--------|
| `src/student/rewards.py` | `src/student/reward_dm.py` | Rename + update imports |
| `src/student/rewards_v3v4.py` (outcome part) | `src/student/reward_outcome.py` | Split + new file |
| `src/student/rewards_v3v4.py` (process part) | `src/student/reward_process.py` | Split + new file |
| `src/student/grpo_config.py` | `src/student/grpo_config_dm.py` | Rename |
| `src/student/grpo_config_v4.py` (GRPO_CONFIG_V3) | `src/student/grpo_config_outcome.py` | Split |
| `src/student/grpo_config_v4.py` (GRPO_CONFIG_V4) | `src/student/grpo_config_process.py` | Split |
| `src/student/train_grpo.py` | `src/student/train_grpo_dm.py` | Rename + update imports |
| `src/student/train_grpo_v3.py` | `src/student/legacy/train_grpo_outcome_custom.py` | Move to legacy |
| `src/student/train_grpo_v4.py` | `src/student/legacy/train_grpo_process_custom.py` | Move to legacy |

## Tasks

### Task 1: Create reward_dm.py from rewards.py
- Copy `rewards.py` to `reward_dm.py`
- Update TYPE_CHECKING import to legacy path (already done)
- Update docstring to reference v1/v2
- Delete `rewards.py`

### Task 2: Split rewards_v3v4.py into reward_outcome.py and reward_process.py
- `reward_outcome.py`: sign/bool extraction, correctness functions, `compute_outcome_reward`, `build_v3_reward_fn`, shared patterns (POSITIVE_PATTERNS, _HEDGING_PATTERNS, DM_*, _MECHANISM_PATTERNS, compute_proxy_outcome)
- `reward_process.py`: RLVMR tag extraction, planning/commitment/reflection/monitor/format_penalty, `compute_process_rewards`, `build_v4_reward_fn`
- `reward_process.py` imports shared patterns from `reward_outcome.py`
- Delete `rewards_v3v4.py`

### Task 3: Rename grpo_config.py to grpo_config_dm.py
- Rename file
- Update `create_grpo_config` to reference `reward_dm` reward functions
- Update docstring to reference v1/v2

### Task 4: Split grpo_config_v4.py into grpo_config_outcome.py and grpo_config_process.py
- `grpo_config_outcome.py`: GRPO_CONFIG_V3 renamed to `DEFAULT_CONFIG`, add `create_grpo_config()` for outcome track
- `grpo_config_process.py`: GRPO_CONFIG_V4 renamed to `DEFAULT_CONFIG`, add `create_grpo_config()` for process track
- Delete `grpo_config_v4.py`

### Task 5: Rename train_grpo.py to train_grpo_dm.py
- Rename file
- Update imports from `reward_dm` and `grpo_config_dm`
- Update docstring to reference v1/v2

### Task 6: Move custom loops to legacy
- Move `train_grpo_v3.py` to `legacy/train_grpo_outcome_custom.py`
- Move `train_grpo_v4.py` to `legacy/train_grpo_process_custom.py`
- Update their imports to reference new reward/config file names

### Task 7: Update all import references
- `src/tests/test_grpo_training.py`: `reward_dm`, `grpo_config_dm`
- `src/tests/test_rewards.py`: `reward_dm`
- `src/tests/test_grpo_config.py`: `grpo_config_dm`
- Any other files that import from old paths

### Task 8: Update revisions.md execution guide
- Update all training commands with new module names
- Add v1/v2/v3/v4 labels throughout
- Align with grpo-v3-proposal experimental conditions

### Task 9: Verify
- Run test suite: `./scripts/run_e2e_tests.sh`
- Verify no broken imports
- Verify all module paths resolve

## Constraints
- Don't run training
- Host numpy/pandas incompatibility means some imports can't be tested on host
- Keep function signatures identical within reward files (only moving code, not changing APIs)
