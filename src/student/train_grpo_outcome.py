#!/usr/bin/env python3
"""
GRPO v3 Training via Unsloth GRPOTrainer (Outcome Rewards - Correctness-Based)

Outcome-only GRPO training on EconCausal + Corr2Cause + synthetic data.
Flat advantage: single group-relative normalization of outcome rewards.
Serves as the control condition for v4.

Usage:
    python3 -m src.student.train_grpo_outcome \\
        --base-model checkpoints/merged/cold_start_merged \\
        --dataset-path data/processed/grpo_train_merged.jsonl \\
        --output-dir checkpoints/lora_adapters/grpo_v3_outcome
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# CRITICAL: Unsloth must be imported before trl/transformers/peft for optimizations
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

from src.student.grpo_config_outcome import DEFAULT_CONFIG, create_grpo_config
from src.student.reward_outcome import compute_outcome_reward
from src.student.train_grpo_base import (
    TrackingCallback,
    TrackingManager,
    build_outcome_dataset,
    build_reward_fn_with_docs,
    find_latest_checkpoint,
    patch_unsloth_chunked_log_softmax,
    strip_vision_config,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _build_reward_funcs() -> list:
    """Build TRL-compatible reward function for outcome rewards."""
    return [
        lambda completions, docs=docs: [
            compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
        ]
        for docs in [{}]
    ]


def _build_trl_reward_fn(doc_index: Dict[str, Dict[str, Any]]) -> Any:
    """Build a TRL-compatible reward function that looks up docs via prompt.

    Args:
        doc_index: Dict mapping prompt text to original doc record.

    Returns:
        TRL-compatible reward function: (completions, prompts, *args) -> List[float].
    """
    return build_reward_fn_with_docs(
        lambda completions, docs: [
            compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
        ],
        doc_index,
    )


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
    enable_profile: bool = False,
    enable_compile: bool = False,
) -> None:
    """Run GRPO v3 training via Unsloth's GRPOTrainer."""
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

    # Must patch before creating GRPOTrainer (fixes unsloth #5121)
    patch_unsloth_chunked_log_softmax()

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

    if enable_compile:
        from src.student.train_grpo_base import maybe_compile_model
        model, was_compiled = maybe_compile_model(model, enable=True)
        if was_compiled:
            logger.info("Model successfully compiled with torch.compile")
        else:
            logger.warning("torch.compile requested but fell back to uncompiled model")
    else:
        logger.info("torch.compile disabled (use --compile to enable)")

    tracker = None
    if enable_profile:
        from src.utils.memory_profiler import TrainingMemoryTracker, force_memory_cleanup
        tracker = TrainingMemoryTracker()
        cleanup = force_memory_cleanup()
        logger.info(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB")
        tracker.record(0, "after_model_setup")

    dataset = build_outcome_dataset(dataset_path, tokenizer)

    doc_index = {row["prompt"]: row["doc"] for row in dataset}
    reward_funcs = [_build_trl_reward_fn(doc_index)]

    grpo_config = create_grpo_config(output_dir=output_dir, torch_compile=enable_compile)

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    tracker = TrackingManager()
    tracker.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", f"grpo-v3-outcome-{Path(output_dir).name}"),
        config={
            "training_method": "GRPO",
            "track": "outcome",
            "version": "v3",
            "group_size": grpo_config.num_generations,
            "beta": grpo_config.beta,
            "learning_rate": grpo_config.learning_rate,
            "lora_rank": DEFAULT_CONFIG["lora_rank"],
            "lora_alpha": DEFAULT_CONFIG["lora_alpha"],
            "max_completion_length": grpo_config.max_completion_length,
            "max_steps": grpo_config.max_steps,
        },
        track="outcome",
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
    trainer.add_callback(TrackingCallback(tracker))
    if tracker._active:
        logger.info("TrackingManager initialized (step-by-step logging active)")

    logger.info("Starting GRPO v3 training (outcome rewards)...")
    logger.info(
        f"  Steps: {grpo_config.max_steps}, "
        f"G: {grpo_config.num_generations}, "
        f"LR: {grpo_config.learning_rate}, "
        f"Beta: {grpo_config.beta}"
    )
    resume_from = f"checkpoint-{resume_step}" if resume_step > 0 else None
    trainer.train(resume_from_checkpoint=resume_from)

    tracker.finish()

    if tracker:
        from src.utils.memory_profiler import get_vram_peak_gb
        tracker.record(grpo_config.max_steps, "after_training")
        logger.info(f"Peak VRAM: {get_vram_peak_gb():.2f} GB")
        summary = tracker.summary()
        logger.info(f"Memory summary: {summary}")

    logger.info(f"Training complete. Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO v3 Training for Outcome Rewards (Unsloth GRPOTrainer)"
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable memory profiling during training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model",
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
        train(
        base_model, args.output_dir, args.dataset_path,
        resume_step=resume_step,
        enable_profile=args.profile,
        enable_compile=args.compile,
    )
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
