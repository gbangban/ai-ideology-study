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
import sys
from pathlib import Path
from typing import List

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
):
    """Load questions.json and build a HF Dataset with 'prompt' column."""
    from datasets import Dataset

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
    from trl import GRPOTrainer

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
    resume_from = f"checkpoint-{resume_step}" if resume_step > 0 else None
    trainer.train(resume_from_checkpoint=resume_from)

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
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
