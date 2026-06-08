#!/usr/bin/env python3
"""
GRPO v4 Training via Unsloth GRPOTrainer (Process Rewards - RLVMR Dual Advantage)

Dual-advantage GRPO training with outcome + process rewards, KL regularization,
and RLVMR tagged output format.

Usage:
    python3 -m src.student.train_grpo_process \\
        --base-model checkpoints/merged/cold_start_merged \\
        --dataset-path data/processed/grpo_train_merged.jsonl \\
        --output-dir checkpoints/lora_adapters/grpo_v4_process
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# CRITICAL: Unsloth must be imported before trl/transformers/peft for optimizations
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

from src.student.grpo_config_process import DEFAULT_CONFIG, REWARD_WEIGHTS, create_grpo_config
from src.student.reward_outcome import compute_outcome_reward
from src.student.reward_process import compute_process_rewards, RLVMR_REQUIRED_TAGS
from src.student.train_grpo_base import (
    build_outcome_dataset,
    build_reward_fn_with_docs,
    find_latest_checkpoint,
    strip_vision_config,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _compute_combined_process_reward(completion: str, doc: Dict[str, Any]) -> float:
    """Compute combined process reward for a single completion.

    Sums individual process rewards (planning, commitment, reflection, monitor,
    format_penalty) weighted by their RLVMR coefficients.
    """
    outcome = compute_outcome_reward(doc, completion)
    process = compute_process_rewards(
        completion,
        outcome,
        required_tags=RLVMR_REQUIRED_TAGS,
        penalty_per_tag=REWARD_WEIGHTS["lambda_format"],
    )
    return sum(process.values())


def _build_reward_funcs() -> list:
    """Build TRL-compatible reward functions for v4 (outcome + process).

    Returns two reward functions:
    1. Outcome reward (correctness-based)
    2. Process reward (RLVMR tags + format penalty)
    """
    def outcome_fn(completions, docs):
        return [compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)]

    def process_fn(completions, docs):
        return [_compute_combined_process_reward(c, doc) for c, doc in zip(completions, docs)]

    return [outcome_fn, process_fn]


def _build_trl_reward_fns(doc_index: Dict[str, Dict[str, Any]]) -> tuple:
    """Build TRL-compatible reward functions that look up docs via prompt.

    Args:
        doc_index: Dict mapping prompt text to original doc record.

    Returns:
        Tuple of (outcome_fn, process_fn) TRL-compatible reward functions.
    """
    outcome_fn = build_reward_fn_with_docs(
        lambda completions, docs: [
            compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
        ],
        doc_index,
    )

    process_fn = build_reward_fn_with_docs(
        lambda completions, docs: [
            _compute_combined_process_reward(c, doc) for c, doc in zip(completions, docs)
        ],
        doc_index,
    )

    return outcome_fn, process_fn


def _find_latest_checkpoint(output_dir: str) -> tuple:
    """Find the latest checkpoint directory in output_dir."""
    return find_latest_checkpoint(output_dir)


def _strip_vision_config(model_path: str) -> None:
    """Remove vision_config from model config.json."""
    strip_vision_config(model_path)


def train(
    base_model_path: str,
    output_dir: str,
    dataset_path: str,
    resume_step: int = 0,
) -> None:
    """Run GRPO v4 training via Unsloth's GRPOTrainer."""
    if resume_step == 0:
        latest_step, latest_path = _find_latest_checkpoint(output_dir)
        if latest_step > 0:
            logger.info(
                f"Found checkpoint at step {latest_step}: {latest_path}\n"
                f"To resume, pass --resume-step {latest_step} --base-model {latest_path}"
            )
    else:
        logger.info(f"Resuming from step {resume_step}")

    _strip_vision_config(base_model_path)

    from unsloth import FastLanguageModel

    logger.info(f"Loading model from {base_model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.95,
    )

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        logger.info("Extracted text tokenizer from VLProcessor (text-only mode)")

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)
    logger.info("Applied Mistral tokenizer regex fix")

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

    dataset = build_outcome_dataset(dataset_path, tokenizer)

    doc_index = {row["prompt"]: row["doc"] for row in dataset}
    outcome_fn, process_fn = _build_trl_reward_fns(doc_index)
    reward_funcs = [outcome_fn, process_fn]

    grpo_config = create_grpo_config(output_dir=output_dir)

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO v4 training (outcome + process rewards)...")
    logger.info(
        f"  Steps: {grpo_config.max_steps}, "
        f"G: {grpo_config.num_generations}, "
        f"LR: {grpo_config.learning_rate}, "
        f"Beta: {grpo_config.beta}"
    )
    resume_from = f"checkpoint-{resume_step}" if resume_step > 0 else None
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"Training complete. Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO v4 Training for Process Rewards (Unsloth GRPOTrainer)"
    )
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
        "--dataset-path",
        default=DEFAULT_CONFIG["dataset_path"],
        help="Path to merged training dataset JSONL",
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
        train(base_model, args.output_dir, args.dataset_path, resume_step=resume_step)
    except Exception:
        logger.error("Training failed, flushing VRAM...")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
