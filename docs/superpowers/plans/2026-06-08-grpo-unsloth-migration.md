# Unsloth GRPOTrainer Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom 760-line GRPO training loop with Unsloth's `GRPOTrainer` + `GRPOConfig`, preserving the CLI interface and reward functions.

**Architecture:** Full rewrite following Unsloth notebook patterns. `train_grpo.py` becomes a ~150-line script that loads the model via `FastLanguageModel.from_pretrained(fast_inference=False)`, prepares a HuggingFace Dataset, builds reward functions, creates `GRPOConfig`, instantiates `GRPOTrainer`, and calls `.train()`. The custom per-sample backward loop, advantage computation, PPO clipping, and manual W&B logging are all eliminated.

**Tech Stack:** Unsloth GRPOTrainer, GRPOConfig, HuggingFace Datasets, Qwen3.5-9B NF4 QLoRA, RTX 5090 32GB

---

### Task 1: Create legacy directory and archive custom implementation

**Files:**
- Create: `src/student/legacy/__init__.py`
- Move: `src/student/train_grpo.py` -> `src/student/legacy/train_grpo_custom.py`
- Move: `src/student/sglang_client.py` -> `src/student/legacy/sglang_client.py`

- [ ] **Step 1: Create legacy directory with __init__.py**

Create `src/student/legacy/__init__.py` with contents:

```python
# Legacy GRPO implementation - archived during Unsloth GRPOTrainer migration
```

- [ ] **Step 2: Move custom train_grpo.py to legacy**

```bash
mv src/student/train_grpo.py src/student/legacy/train_grpo_custom.py
mv src/student/sglang_client.py src/student/legacy/sglang_client.py
```

- [ ] **Step 3: Run tests to confirm they fail (imports broken)**

Run: `python3 -m pytest src/tests/test_grpo_training.py -v --tb=short`
Expected: FAIL - imports reference moved files

- [ ] **Step 4: Commit**

```bash
git add src/student/legacy/ src/student/train_grpo.py src/student/sglang_client.py
git commit -m "refactor: archive custom GRPO implementation to legacy/"
```

### Task 2: Write new grpo_config.py with GRPOConfig factory

**Files:**
- Create: `src/student/grpo_config.py` (overwrites existing)

- [ ] **Step 1: Write the failing test for config factory**

Add to `src/tests/test_grpo_training.py` (we'll update the full file in Task 6, but verify the config first):

Create a temporary test file `src/tests/test_grpo_config.py`:

```python
def test_grpo_config_factory():
    from src.student.grpo_config import create_grpo_config
    from trl import GRPOConfig

    config = create_grpo_config(output_dir="/tmp/test-grpo")
    assert isinstance(config, GRPOConfig)
    assert config.num_generations == 8
    assert config.beta == 0.1
    assert config.learning_rate == 5e-7
    assert config.max_steps == 500
    assert config.warmup_steps == 50
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert config.max_completion_length == 512
    assert config.epsilon == 0.2
    assert config.loss_type == "dapo"
    assert config.scale_rewards == "group"
    assert config.logging_steps == 25
    assert config.save_steps == 50
    assert config.lr_scheduler_type == "cosine"
    assert config.max_seq_length == 2048
    assert config.report_to == "wandb"
    assert config.wandb_project == "dm-align-grpo"
    assert config.disable_timeout is True


def test_reward_weights_sum_to_one():
    from src.student.grpo_config import REWARD_WEIGHTS
    assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-6


def test_reward_weights_has_expected_keys():
    from src.student.grpo_config import REWARD_WEIGHTS
    assert set(REWARD_WEIGHTS.keys()) == {"dm_alignment", "directional_assertion", "mechanism_commitment"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_grpo_config.py -v --tb=short`
Expected: FAIL - module not found or function not defined

- [ ] **Step 3: Write the new grpo_config.py**

Replace `src/student/grpo_config.py` with:

```python
"""
GRPO Training Configuration

Hyperparameters for Unsloth GRPOTrainer-based GRPO training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime via Unsloth.
full rewrite: uses GRPOConfig instead of flat dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from trl import GRPOConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


REWARD_WEIGHTS: dict[str, float] = {
    "dm_alignment": 0.45,
    "directional_assertion": 0.30,
    "mechanism_commitment": 0.25,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "questions_path": str(_project_root() / "data/raw/questions.json"),
    "output_dir": str(_project_root() / "checkpoints/lora_adapters/grpo_adapter_v2"),
}


def create_grpo_config(output_dir: str | None = None) -> GRPOConfig:
    """Build a GRPOConfig for Unsloth's GRPOTrainer.

    Args:
        output_dir: Override default output directory.
    """
    return GRPOConfig(
        learning_rate=5e-7,
        max_steps=500,
        warmup_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=512,
        beta=0.1,
        epsilon=0.2,
        loss_type="dapo",
        scale_rewards="group",
        logging_steps=25,
        save_steps=50,
        lr_scheduler_type="cosine",
        max_seq_length=2048,
        output_dir=output_dir or DEFAULT_CONFIG["output_dir"],
        report_to="wandb",
        wandb_project="dm-align-grpo",
        disable_timeout=True,
        remove_unused_columns=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tests/test_grpo_config.py -v --tb=short`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/student/grpo_config.py src/tests/test_grpo_config.py
git commit -m "feat: GRPOConfig factory replacing flat dict config"
```

### Task 3: Write new train_grpo.py with GRPOTrainer

**Files:**
- Create: `src/student/train_grpo.py` (new, replaces legacy)

- [ ] **Step 1: Write the new train_grpo.py**

Create `src/student/train_grpo.py`:

```python
#!/usr/bin/env python3
"""
GRPO Training via Unsloth GRPOTrainer

Full rewrite using Unsloth's GRPOTrainer and GRPOConfig.
Qwen3.5 is not vLLM-compatible, so fast_inference=False.
All three reward functions are rule-based (regex).

Usage:
    python3 -m src.student.train_grpo \\
        --base-model /path/to/sft/checkpoint \\
        --output-dir checkpoints/lora_adapters/grpo_adapter_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.grpo_config import DEFAULT_CONFIG, REWARD_WEIGHTS, create_grpo_config
from src.student.rewards import (
    compute_dm_keyword_alignment,
    compute_directional_assertion,
    compute_mechanism_commitment,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _strip_vision_config(model_path: str) -> None:
    """Remove vision_config from model config.json to avoid image processor init errors."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    stripped = False
    for key in list(config.keys()):
        if "vision" in key.lower():
            del config[key]
            stripped = True
    if stripped:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Stripped vision config from {config_path}")


def _build_reward_funcs() -> list:
    """Build TRL-compatible reward functions with weights from config."""
    w = REWARD_WEIGHTS
    return [
        lambda c, w=w["dm_alignment"]: [compute_dm_keyword_alignment(x) * w for x in c],
        lambda c, w=w["directional_assertion"]: [compute_directional_assertion(x) * w for x in c],
        lambda c, w=w["mechanism_commitment"]: [compute_mechanism_commitment(x) * w for x in c],
    ]


def _build_dataset(
    questions_path: str,
    tokenizer,
) -> Dataset:
    """Load questions.json and build a HF Dataset with 'prompt' column."""
    with open(questions_path, "r") as f:
        data = json.load(f)
    questions = [q["question"] for q in data]
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")

    prompts: List[str] = []
    for q in questions:
        chat = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    dataset = Dataset.from_dict({"prompt": prompts})
    return dataset


def _find_latest_checkpoint(output_dir: str) -> tuple[int, str]:
    """Find the latest checkpoint directory in output_dir."""
    if not Path(output_dir).exists():
        return 0, ""
    checkpoints = []
    for d in Path(output_dir).iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step_num = int(d.name.split("-")[1])
                checkpoints.append((step_num, str(d)))
            except (ValueError, IndexError):
                continue
    if not checkpoints:
        return 0, ""
    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    return latest_step, latest_path


def train(
    base_model_path: str,
    output_dir: str,
    questions_path: str,
    resume_step: int = 0,
) -> None:
    """Run GRPO training via Unsloth's GRPOTrainer."""
    # Check for latest checkpoint
    if resume_step == 0:
        latest_step, latest_path = _find_latest_checkpoint(output_dir)
        if latest_step > 0:
            logger.info(
                f"Found checkpoint at step {latest_step}: {latest_path}\n"
                f"To resume, pass --resume-step {latest_step} --base-model {latest_path}"
            )
    else:
        logger.info(f"Resuming from step {resume_step}")

    # Strip vision config from Qwen3.5 checkpoint
    _strip_vision_config(base_model_path)

    # Load model with NF4 quantization
    logger.info(f"Loading model from {base_model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.95,
    )

    # Extract text tokenizer from VLProcessor
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        logger.info("Extracted text tokenizer from VLProcessor (text-only mode)")

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)
    logger.info("Applied Mistral tokenizer regex fix")

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=DEFAULT_CONFIG["lora_rank"],
        lora_alpha=DEFAULT_CONFIG["lora_alpha"],
        lora_dropout=DEFAULT_CONFIG["lora_dropout"],
        target_modules=DEFAULT_CONFIG["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    logger.info(
        f"LoRA applied: rank={DEFAULT_CONFIG['lora_rank']}, "
        f"alpha={DEFAULT_CONFIG['lora_alpha']}"
    )

    # Build dataset
    dataset = _build_dataset(questions_path, tokenizer)

    # Build reward functions
    reward_funcs = _build_reward_funcs()

    # Build GRPOConfig
    grpo_config = create_grpo_config(output_dir=output_dir)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    # Train
    logger.info("Starting GRPO training...")
    logger.info(
        f"  Steps: {grpo_config.max_steps}, "
        f"G: {grpo_config.num_generations}, "
        f"LR: {grpo_config.learning_rate}, "
        f"Beta: {grpo_config.beta}"
    )
    trainer.train()

    # Save final model
    logger.info(f"Training complete. Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO Training for DM Alignment (Unsloth GRPOTrainer)")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_CONFIG["base_model"],
        help="Path to SFT merged checkpoint or GRPO checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CONFIG["output_dir"],
        help="Output directory for GRPO adapter",
    )
    parser.add_argument(
        "--questions-path",
        default=DEFAULT_CONFIG["questions_path"],
        help="Path to questions.json",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Resume from checkpoint at this step",
    )
    parser.add_argument(
        "--find-checkpoint",
        action="store_true",
        help="List available checkpoints and exit",
    )
    args = parser.parse_args()

    if args.find_checkpoint:
        step, path = _find_latest_checkpoint(args.output_dir)
        if step > 0:
            print(f"Latest checkpoint: step {step} at {path}")
            for d in sorted(Path(args.output_dir).iterdir()):
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    print(f"  {d.name}")
        else:
            print(f"No checkpoints found in {args.output_dir}")
        return

    resume_step = args.resume_step
    base_model = args.base_model
    if resume_step > 0:
        ckpt_path = f"{args.output_dir}/checkpoint-{resume_step}"
        if Path(ckpt_path).exists():
            base_model = ckpt_path
            logger.info(f"Auto-resuming from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found, using base-model as-is")

    try:
        train(base_model, args.output_dir, args.questions_path, resume_step=resume_step)
    except Exception:
        logger.error("Training failed, flushing VRAM...")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python3 -m src.student.train_grpo --help`
Expected: Output shows `--base-model`, `--output-dir`, `--questions-path`, `--resume-step`, `--find-checkpoint`

- [ ] **Step 3: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "feat: rewrite train_grpo.py with Unsloth GRPOTrainer"
```

### Task 4: Update test_grpo_training.py for new implementation

**Files:**
- Modify: `src/tests/test_grpo_training.py`

- [ ] **Step 1: Write updated test_grpo_training.py**

Replace `src/tests/test_grpo_training.py` with:

```python
import pytest


class TestGRPOTraining:
    def test_train_function_exists(self):
        from src.student.train_grpo import train
        assert callable(train)

    def test_main_function_exists(self):
        from src.student.train_grpo import main
        assert callable(main)

    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.student.train_grpo", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "base-model" in result.stdout or "base_model" in result.stdout


class TestGRPOIntegration:
    def test_full_pipeline_imports(self):
        """Test that all GRPO components can be imported together."""
        from src.student.grpo_config import create_grpo_config, REWARD_WEIGHTS, DEFAULT_CONFIG
        from src.student.rewards import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.train_grpo import train, _build_reward_funcs, _build_dataset
        from trl import GRPOConfig, GRPOTrainer
        assert True

    def test_reward_pipeline(self):
        """Test reward computation pipeline end-to-end with v2 rewards."""
        from src.student.rewards import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.grpo_config import REWARD_WEIGHTS

        text = "Capital accumulation drives exploitation through reserve army expansion. This directly increases class inequality and is the primary driver of wage suppression."
        da = compute_directional_assertion(text)
        dm = compute_dm_keyword_alignment(text)
        mc = compute_mechanism_commitment(text)

        w = REWARD_WEIGHTS
        total = w["directional_assertion"] * da + w["dm_alignment"] * dm + w["mechanism_commitment"] * mc
        assert total > 0.1
        assert da > 0
        assert dm > 0
        assert mc > 0

    def test_reward_funcs_callable(self):
        """Test that reward functions accept List[str] and return List[float]."""
        from src.student.train_grpo import _build_reward_funcs

        reward_funcs = _build_reward_funcs()
        assert len(reward_funcs) == 3

        completions = [
            "Capital drives exploitation through structural power. This directly increases inequality.",
            "The market is efficient and prices reflect supply and demand.",
        ]
        for func in reward_funcs:
            results = func(completions)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, (int, float)) for r in results)

    def test_find_latest_checkpoint(self):
        """Test checkpoint discovery."""
        import tempfile, os
        from pathlib import Path
        from src.student.train_grpo import _find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            step, path = _find_latest_checkpoint(tmpdir)
            assert step == 0 and path == ""

            for s in [100, 200, 300]:
                os.makedirs(f"{tmpdir}/checkpoint-{s}")
            step, path = _find_latest_checkpoint(tmpdir)
            assert step == 300
            assert "checkpoint-300" in path


class TestDMKeywordAlignment:
    def test_full_score_three_categories(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation of surplus value drives exploitation. The structural power relations takes for granted the commodification of labor."
        score = compute_dm_keyword_alignment(text)
        assert score == 1.0

    def test_partial_score_one_category(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation drives the economic system forward. The market responds to supply and demand signals."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.5

    def test_zero_score_no_dm_patterns(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "The market is efficient and prices reflect supply and demand. Consumers make rational choices."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.0

    def test_frame_critique_category(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Mainstream analysis naturalizes market outcomes and renders invisible the ideological function of hegemonic discourse."
        score = compute_dm_keyword_alignment(text)
        assert score >= 0.5


class TestAsymmetricDirectionalAssertion:
    def test_committed_response_scores_positive(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy directly causes increases in wages. This is the primary driver of inequality."
        score = compute_directional_assertion(text)
        assert score > 0.0

    def test_hedging_response_scores_negative(self):
        from src.student.rewards import compute_directional_assertion
        text = "The effect is mixed and theoretically ambiguous. It depends on context and empirically heterogeneous outcomes."
        score = compute_directional_assertion(text)
        assert score < 0.0

    def test_balanced_response_near_zero(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy increases wages but the effect is mixed across sectors."
        score = compute_directional_assertion(text)
        assert -0.5 <= score <= 0.5

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_directional_assertion
        assert compute_directional_assertion("") == 0.0
        assert compute_directional_assertion("hi") == 0.0

    def test_score_clipped_to_range(self):
        from src.student.rewards import compute_directional_assertion
        text = "mixed ambiguous uncertain depends both sides heterogeneous"
        score = compute_directional_assertion(text)
        assert -1.0 <= score <= 1.0


class TestMechanismCommitment:
    def test_mechanism_with_commitment_scores_positive(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital accumulation drives wage suppression through reserve army expansion. This directly increases exploitation."
        score = compute_mechanism_commitment(text)
        assert score > 0.0

    def test_mechanism_with_hedging_scores_negative(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital drives outcomes through market mechanisms, but the effect is mixed and depends on structural conditions."
        score = compute_mechanism_commitment(text)
        assert score < 0.0

    def test_no_mechanism_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "The situation is complex and multifaceted. There are many factors at play."
        score = compute_mechanism_commitment(text)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        assert compute_mechanism_commitment("") == 0.0
```

- [ ] **Step 2: Run updated GRPO tests**

Run: `python3 -m pytest src/tests/test_grpo_training.py -v --tb=short`
Expected: PASS (all tests - no GPU needed, tests are unit-level only)

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_grpo_training.py
git commit -m "test: update GRPO tests for Unsloth GRPOTrainer migration"
```

### Task 5: Update SG-Lang client tests and test runner

**Files:**
- Modify: `src/tests/test_sglang_client.py` (update imports to legacy path)
- Modify: `scripts/run_e2e_tests.sh` (update SG-Lang test import path)

- [ ] **Step 1: Update SG-Lang client test imports**

In `src/tests/test_sglang_client.py`, replace all occurrences of `from src.student.sglang_client import SglangClient` with `from src.student.legacy.sglang_client import SglangClient`.

- [ ] **Step 2: Run SG-Lang tests**

Run: `python3 -m pytest src/tests/test_sglang_client.py -v --tb=short`
Expected: PASS (6/6)

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_sglang_client.py
git commit -m "test: update sglang_client imports to legacy path"
```

### Task 6: Run full test suite

**Files:**
- No file changes

- [ ] **Step 1: Run full test suite**

Run: `./scripts/run_e2e_tests.sh`
Expected: All steps pass (teacher, SFT config, DPO, SG-Lang client, GRPO training). E2E step may show WARNING which is expected without GPU.

- [ ] **Step 2: Verify CLI help for both scripts**

Run: `python3 -m src.student.train_grpo --help`
Expected: Shows all expected arguments

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "refactor: complete Unsloth GRPOTrainer migration

- Replaced custom 760-line GRPO loop with Unsloth GRPOTrainer
- New grpo_config.py with GRPOConfig factory
- Rewards unchanged (rule-based, TRL-compatible)
- Custom implementation archived to legacy/
- SG-Lang client moved to legacy/ (deprecated for training)
- All tests pass"
```

## Self-Review

**Spec coverage check:**
- File structure (legacy/, new train_grpo.py, new grpo_config.py, rewards.py unchanged): Task 1, 2, 3
- Data flow (questions.json -> HF Dataset -> GRPOTrainer): Task 3 `_build_dataset`
- Model loading (FastLanguageModel, fast_inference=False, gpu_memory_utilization=0.95): Task 3
- Reward functions (3 separate lambdas, weights from config): Task 3 `_build_reward_funcs`
- GRPOConfig parameters (all 17 parameters from table): Task 2 `create_grpo_config`
- CLI interface (preserved args): Task 3 `main()`
- Vision config stripping: Task 3 `_strip_vision_config`
- Testing strategy (removed tests, new tests, unchanged tests): Task 4
- SG-Lang deprecation: Task 1, Task 5

**Placeholder scan:** No TBDs, no TODOs, no "implement later". Every step has exact code or commands.

**Type consistency:** `REWARD_WEIGHTS` dict used consistently across grpo_config.py, train_grpo.py, and tests. `_find_latest_checkpoint` returns `tuple[int, str]` in both train_grpo.py and test. `create_grpo_config` returns `GRPOConfig` type.

**Gap check:** The `_build_dataset` function uses `tokenizer.apply_chat_template` which requires the tokenizer to be loaded first - this is correct since we build the dataset after loading the model. The `remove_unused_columns=False` in GRPOConfig is needed because our dataset only has `prompt` column and GRPOTrainer needs it.
