#!/usr/bin/env python3
"""
GRPO Training Script

Group Relative Policy Optimization for DM alignment.
Loads SFT merged checkpoint via Unsloth NF4, trains with TRL GRPOTrainer.

Usage:
    python3 -m src.student.train_grpo \\
        --base-model /path/to/sft/checkpoint \\
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


def format_prompt(question: str) -> str:
    """Format a question as a chat prompt for the model."""
    return [{"role": "user", "content": question}]


def train(config: dict, base_model_path: str, output_dir: str):
    """Run GRPO training."""
    from unsloth import FastLanguageModel

    # Load model with NF4 quantization
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
    if "dm_alignment" in config["reward_weights"]:
        print(f"Loading judge model: {config['judge_model']}...")
        judge_model = AutoModelForCausalLM.from_pretrained(
            config["judge_model"],
            torch_dtype=torch.bfloat16,
        ).cuda()
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

    prompts = [format_prompt(q) for q in questions]
    prompt_texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

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
        run_name="grpo-dm-alignment",
    )

    # GRPOTrainer
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=training_args,
        reward_funcs=[length_aware_reward],
        train_dataset=dataset,
        # GRPO-specific
        grpo_g=config["grpo_g"],
        max_completion_length=config["max_completion_length"],
    )

    print(f"Starting GRPO training...")
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
    parser = argparse.ArgumentParser(description="GRPO Training for DM Alignment")
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
