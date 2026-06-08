# Implementation Plan: GRPO Container Smoke Test

## Prerequisites
- `src/student/train_grpo_outcome.py` — reference for v3 code path
- `src/student/train_grpo_process.py` — reference for v4 code path
- `src/student/train_grpo_base.py` — shared utilities
- `src/student/grpo_config_outcome.py` — v3 config factory
- `src/student/grpo_config_process.py` — v4 config factory
- `src/student/reward_outcome.py` — outcome rewards
- `src/student/reward_process.py` — process rewards

## Task 1: Modify `create_grpo_config` to accept smoke test overrides

**Files:** `grpo_config_outcome.py`, `grpo_config_process.py`

Both `create_grpo_config` functions hardcode `max_steps=1000` and `save_steps=100`. For smoke testing, we need `max_steps=1` and `save_steps=99999`. Rather than mutating the config after creation, add optional keyword arguments.

In `grpo_config_outcome.py`:
```python
def create_grpo_config(
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    logging_steps: Optional[int] = None,
) -> GRPOConfig:
```
Use provided values or fall back to defaults.

Same pattern for `grpo_config_process.py`.

**Tests:** Add to `test_grpo_config.py`:
- `test_create_grpo_config_outcome_smoke_override` — verify max_steps=1 is respected
- `test_create_grpo_config_process_smoke_override` — same for v4

## Task 2: Create `src/student/smoke_test.py`

New module. Follows the same code path as training scripts but with smoke test config.

Structure:
- `smoke_test(track, base_model, dataset_path, num_prompts)` — main function
- `main()` — CLI entry point with argparse
- `--track outcome|process` (required)
- `--base-model`, `--dataset-path`, `--num-prompts` (optional, with defaults)

The function:
1. Validates `--track` is "outcome" or "process"
2. Calls `strip_vision_config(base_model)`
3. Loads model via `FastLanguageModel.from_pretrained` (same params as training scripts)
4. Extracts tokenizer, applies `fix_mistral_tokenizer`
5. Applies LoRA via `get_peft_model` + `for_training`
6. Builds dataset via `build_outcome_dataset`, subsamples to `num_prompts`
7. Builds `doc_index`
8. For outcome track: builds single reward fn via `_build_trl_reward_fn` pattern
9. For process track: builds two reward fns via `_build_trl_reward_fns` pattern
10. Creates GRPOConfig with `max_steps=1`, `save_steps=99999`, `logging_steps=1`
11. Instantiates `GRPOTrainer`
12. Calls `trainer.train()`
13. Validates training result — checks loss is finite
14. Prints structured output, exits 0

**Tests:** `src/tests/test_smoke_test.py` — host-only tests that don't need GPU:
- `test_smoke_test_module_imports` — module imports without error
- `test_cli_help` — `--help` returns 0
- `test_track_validation` — invalid track raises error
- `test_train_function_exists` — `smoke_test` is callable
- `test_main_function_exists` — `main` is callable

## Task 3: Create `scripts/smoke_test_training.sh`

Shell wrapper:
- Checks container is running via `ddk ps`
- Executes `python3 -m src.student.smoke_test` inside container
- Passes `--track` and `--num-prompts` through
- When no args, runs both tracks sequentially
- Returns exit code from Python process

## Task 4: Update `scripts/run_e2e_tests.sh`

Add Step 6 after GRPO training tests:
```
Step 6: Running Container Smoke Tests (optional)
```
Skips if `ml-training` container is not running. Prints warning if skipped.

## Task 5: Update `AGENTS.md`

Add under "Workflow Commands":
```
### Smoke Test (One Training Step)
docker exec ml-training python3 -m src.student.smoke_test --track outcome
docker exec ml-training python3 -m src.student.smoke_test --track process
```

## Execution Order

1. Task 1 — config overrides (tests first, then implementation)
2. Task 2 — smoke_test.py module (tests first, then implementation)
3. Task 3 — shell script
4. Task 4 — e2e test runner update
5. Task 5 — AGENTS.md update
6. Run full test suite: `./scripts/run_e2e_tests.sh`
