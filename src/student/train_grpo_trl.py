#!/usr/bin/env python3
"""
TRL GRPO Training Script (CUDA 13 + vLLM container only)

Standard TRL GRPOTrainer workflow. Requires vLLM, CUDA 13, torch 2.11+.
Run inside ml-training-grpo container built from docker/Dockerfile.cu13.

Usage:
    python3 -m src.student.train_grpo_trl \
        --base-model /path/to/sft/checkpoint \
        --output-dir checkpoints/lora_adapters/grpo_adapter
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.grpo_config import GRPO_CONFIG
from src.student.rewards import (
    build_reward_fn,
    compute_length_reward,
)


def load_questions(filepath: str) -> List[str]:
    """Load questions from questions.json, returning only prompt text."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [q["question"] for q in data]


def train(config: dict, base_model_path: str, output_dir: str):
    """Run GRPO training using TRL GRPOTrainer."""
    from unsloth import FastLanguageModel

    # Load model with NF4 quantization
    # Strip vision config to avoid image processor init errors
    config_path = Path(base_model_path) / "config.json"
    if config_path.exists():
        cfg = json.load(open(config_path))
        for k in list(cfg.keys()):
            if "vision" in k.lower():
                del cfg[k]
        json.dump(cfg, open(config_path, "w"), indent=2)

    print(f"Loading model from {base_model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    print(f"LoRA applied: rank={config['lora_rank']}, alpha={config['lora_alpha']}")

    # Load judge model
    judge_model = None
    judge_tokenizer = None
    if config["reward_weights"].get("dm_alignment", 0) > 0:
        print(f"Loading judge model: {config['judge_model']}...")
        judge_model = AutoModelForCausalLM.from_pretrained(
            config["judge_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(config["judge_model"])
        print(f"Judge model loaded on {judge_model.device}")

    # Build reward function
    reward_fn = build_reward_fn(
        config["reward_weights"],
        judge_model,
        judge_tokenizer,
    )

    # Load and prepare dataset
    print(f"Loading questions from {config['questions_path']}...")
    questions = load_questions(config["questions_path"])
    print(f"  Loaded {len(questions)} questions")

    prompts = [[{"role": "user", "content": q}] for q in questions]
    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    dataset = Dataset.from_dict({"prompt": prompt_texts})

    # Length-aware reward wrapper
    def length_aware_reward(completions: List[str]) -> List[float]:
        base_scores = reward_fn(completions)
        length_weight = config["reward_weights"].get("length", 0)
        if length_weight > 0:
            for i, completion in enumerate(completions):
                tokens = len(tokenizer.encode(completion, add_special_tokens=False))
                length_score = compute_length_reward(tokens)
                base_scores[i] += length_weight * length_score
        return base_scores

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        bf16=True,
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        report_to="wandb",
        run_name="grpo-dm-alignment-trl",
    )

    # TRL GRPOTrainer
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=training_args,
        reward_funcs=[length_aware_reward],
        train_dataset=dataset,
        grpo_g=config["grpo_g"],
        max_completion_length=config["max_completion_length"],
    )

    print(f"Starting GRPO training (TRL)...")
    print(f"  Steps: {config['max_steps']}, G: {config['grpo_g']}, LR: {config['learning_rate']}")
    print(f"  Estimated duration: 9-12 hours")

    metrics = trainer.train()
    print(f"Training complete. Metrics: {metrics}")

    # Save adapter
    print(f"Saving GRPO adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for DM Alignment (TRL)")
    parser.add_argument(
        "--base-model",
        default=GRPO_CONFIG["base_model"],
        help="Path to SFT merged checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=GRPO_CONFIG["output_dir"],
        help="Output directory for GRPO adapter",
    )
    parser.add_argument(
        "--questions-path",
        default=GRPO_CONFIG["questions_path"],
        help="Path to questions.json",
    )
    args = parser.parse_args()

    config = GRPO_CONFIG.copy()
    config["base_model"] = args.base_model
    config["questions_path"] = args.questions_path

    train(config, args.base_model, args.output_dir)


if __name__ == "__main__":
    main()
