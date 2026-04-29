"""
SFT Training Script

DEPRECATED: This script is replaced by Unsloth Studio UI for SFT training.
Keep as reference only. Use configs/studio_sft_config.yaml uploaded via Studio
-> Parameters -> Upload YAML instead.

Supervised fine-tuning using QLoRA on the synthetic DM-aligned dataset.
"""

import json
from pathlib import Path
from typing import List

from src.student.config import SFT_CONFIG


def _get_torch():
    """Lazy import of torch."""
    import torch

    return torch


def load_dataset(filepath: str) -> List[dict]:
    """
    Load JSONL dataset from file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of sample dictionaries
    """
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_conversation(conversations: List[dict]) -> str:
    """
    Format conversation into training string.

    Args:
        conversations: List of conversation turns

    Returns:
        Formatted string for training
    """
    formatted = ""
    for turn in conversations:
        if turn["role"] == "user":
            formatted += f"User: {turn['content']}\n"
        elif turn["role"] == "assistant":
            formatted += f"Assistant: {turn['content']}\n"
    return formatted


def configure_model_for_training(model, gradient_checkpointing: str = "unsloth"):
    """
    Configure model for efficient training.

    Args:
        model: The model to configure
        gradient_checkpointing: Type of gradient checkpointing to use

    Returns:
        Configured model
    """
    if gradient_checkpointing == "unsloth":
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_training(model)
        except ImportError:
            pass  # unsloth not available in test environment
    else:
        model.enable_gradient_checkpointing()

    model.config.use_gradient_checkpointing = True
    return model


def save_adapter(model, tokenizer, output_dir: str):
    """
    Save LoRA adapter to directory.

    Args:
        model: Trained model with LoRA adapters
        tokenizer: Model tokenizer
        output_dir: Directory to save adapter
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Adapter saved to {output_dir}")


def run_training_step(model, batch, optimizer):
    """
    Run a single training step.

    Args:
        model: The training model
        batch: Input batch
        optimizer: Optimizer

    Returns:
        Loss value
    """
    torch = _get_torch()
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()


def train_sft(
    model,
    tokenizer,
    dataset: List[dict],
    config: dict = None,
):
    """
    Run SFT training on dataset.

    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        dataset: Training dataset
        config: Training configuration

    Returns:
        Trained model
    """
    torch = _get_torch()
    if config is None:
        config = SFT_CONFIG

    # Configure model
    model = configure_model_for_training(model, config["gradient_checkpointing"])

    # Prepare dataset
    formatted_data = [format_conversation(s["conversations"]) for s in dataset]
    tokenized = tokenizer(
        formatted_data,
        padding=True,
        truncation=True,
        max_length=config["max_seq_length"],
        return_tensors="pt",
    )

    # Setup optimizer
    from peft import get_peft_model_state_dict

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.01
    )

    # Training loop
    losses = []
    total_steps = min(
        config["max_steps"], len(dataset) // config["per_device_train_batch_size"]
    )

    for step in range(total_steps):
        batch_idx = step % len(dataset)
        batch = {
            "input_ids": tokenized["input_ids"][batch_idx : batch_idx + 1],
            "attention_mask": tokenized["attention_mask"][batch_idx : batch_idx + 1],
            "labels": tokenized["input_ids"][batch_idx : batch_idx + 1].clone(),
        }

        loss = run_training_step(model, batch, optimizer)
        losses.append(loss)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{total_steps}, Loss: {loss:.4f}")

    print(f"Training completed. Final loss: {losses[-1]:.4f}")

    return model


def main(
    dataset_path: str = "data/processed/sft_dataset.jsonl",
    output_dir: str = "checkpoints/lora_adapters/sft_adapter",
):
    """
    Main entry point for SFT training.

    Args:
        dataset_path: Path to training dataset
        output_dir: Output directory for adapter
    """
    print("Loading model and tokenizer...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_CONFIG["model_name"],
        max_seq_length=SFT_CONFIG["max_seq_length"],
        dtype=SFT_CONFIG["bnb_4bit_compute_dtype"],
        load_in_4bit=SFT_CONFIG["load_in_4bit"],
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=SFT_CONFIG["r"],
        lora_alpha=SFT_CONFIG["lora_alpha"],
        lora_dropout=SFT_CONFIG["lora_dropout"],
        target_modules=SFT_CONFIG["target_modules"],
    )

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    print("Starting SFT training...")
    model = train_sft(model, tokenizer, dataset, SFT_CONFIG)

    print(f"Saving adapter to {output_dir}...")
    save_adapter(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
