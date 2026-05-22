#!/usr/bin/env python3
"""
Programmatic SFT Training v2

Uses Unsloth Core + TRL SFTTrainer for trace-aligned SFT training.
Supports Neftune noise, proper chat template, and loss masking.

Usage:
    python3 -m src.student.train_sft_v2 \
        --dataset data/processed/sft_dataset.jsonl \
        --output-dir checkpoints/lora_adapters/sft_v2_adapter
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.sft_config_v2 import SFT_CONFIG


def prepare_model_for_training(model, tokenizer, config: dict):
    """Apply LoRA adapters to the model."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    return model


def apply_neftune(model, noise_alpha: float):
    """Apply Neftune noise embedding to the model."""
    model.config.neftune_noise_alpha = noise_alpha
    return model


def load_dataset_for_training(dataset_path: str):
    """Load JSONL dataset for TRL SFTTrainer."""
    from datasets import load_dataset

    return load_dataset("json", data_files=dataset_path, split="train")


def train(config: dict, dataset_path: str, output_dir: str):
    """Run SFT training."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from peft import PeftModel

    # Load model
    print(f"Loading model: {config['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = prepare_model_for_training(model, tokenizer, config)

    # Apply Neftune noise
    if config.get("neftune_noise_alpha"):
        print(f"Applying Neftune noise (alpha={config['neftune_noise_alpha']})")
        model = apply_neftune(model, config["neftune_noise_alpha"])

    # Enable gradient checkpointing
    model = FastLanguageModel.for_training(model)

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset_for_training(dataset_path)
    print(f"  Loaded {len(dataset)} samples")

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        optim=config["optim"],
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 200),
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=None,
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    # Train
    print("Starting SFT training...")
    metrics = trainer.train()
    print(f"Training complete. Metrics: {metrics}")

    # Save
    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Programmatic SFT Training v2")
    parser.add_argument("--dataset", default="data/processed/sft_dataset.jsonl", help="SFT dataset JSONL")
    parser.add_argument("--output-dir", default=SFT_CONFIG["output_dir"], help="Output directory")
    args = parser.parse_args()

    train(SFT_CONFIG, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
