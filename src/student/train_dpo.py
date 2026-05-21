#!/usr/bin/env python3
"""
DPO Training Script

Direct Preference Optimization on preference pairs.
Loads SFT v2 adapter, trains with TRL DPOTrainer.

Usage:
    python3 -m src.student.train_dpo \
        --sft-adapter-path checkpoints/lora_adapters/sft_v2_adapter \
        --dpo-pairs-path data/processed/dpo_pairs.jsonl \
        --output-dir checkpoints/lora_adapters/dpo_adapter
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.dpo_config import DPO_CONFIG


def load_dpo_pairs(filepath: str) -> list[dict]:
    """Load DPO pairs from JSONL file."""
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def train(config: dict, sft_adapter_path: str, dpo_pairs_path: str, output_dir: str):
    """Run DPO training."""
    from unsloth import FastLanguageModel
    from trl import DPOTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # Load model with SFT adapter
    print(f"Loading SFT adapter from {sft_adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA for DPO fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = FastLanguageModel.for_training(model)

    # Load dataset
    print(f"Loading DPO pairs from {dpo_pairs_path}...")
    dataset = load_dataset("json", data_files=dpo_pairs_path, split="train")
    print(f"  Loaded {len(dataset)} pairs")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        bf16=True,
        logging_steps=config.get("logging_steps", 25),
        save_steps=config.get("save_steps", 100),
        report_to="none",
    )

    # DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=training_args,
        beta=config["beta"],
        train_dataset=dataset,
    )

    print("Starting DPO training...")
    metrics = trainer.train()
    print(f"Training complete. Metrics: {metrics}")

    # Save
    print(f"Saving DPO adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="DPO Training for DM Alignment")
    parser.add_argument("--sft-adapter-path", default=DPO_CONFIG["base_model"], help="Path to SFT adapter")
    parser.add_argument("--dpo-pairs-path", default="data/processed/dpo_pairs.jsonl", help="Path to DPO pairs JSONL")
    parser.add_argument("--output-dir", default=DPO_CONFIG["output_dir"], help="Output directory")
    args = parser.parse_args()

    train(DPO_CONFIG, args.sft_adapter_path, args.dpo_pairs_path, args.output_dir)


if __name__ == "__main__":
    main()
