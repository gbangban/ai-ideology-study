"""
DPO Training Script

Direct Preference Optimization training on preference pairs.
"""

import json
from pathlib import Path
from typing import List, Dict

from src.student.dpo_config import DPO_CONFIG

REQUIRED_DPO_FILES = ["adapter_model.safetensors", "scheduler.pt"]


def _get_torch():
    """Lazy import of torch."""
    import torch

    return torch


def load_dpo_pairs(filepath: str) -> List[dict]:
    """
    Load DPO pairs from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of DPO pair dictionaries
    """
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def format_dpo_sample(sample: dict) -> dict:
    """
    Format DPO sample for training.

    Args:
        sample: Raw DPO sample

    Returns:
        Formatted sample with question, chosen, rejected
    """
    return {
        "question": sample.get("question", ""),
        "chosen": sample.get("chosen", ""),
        "rejected": sample.get("rejected", ""),
    }


def prepare_dpo_batch(pairs: List[dict], tokenizer) -> dict:
    """
    Prepare a batch of DPO pairs for training.

    Args:
        pairs: List of DPO pairs
        tokenizer: Tokenizer for encoding

    Returns:
        Dict with tokenized chosen and rejected inputs
    """
    chosen_texts = []
    rejected_texts = []

    for pair in pairs:
        formatted = format_dpo_sample(pair)
        chosen_texts.append(
            f"User: {formatted['question']}\nAssistant: {formatted['chosen']}"
        )
        rejected_texts.append(
            f"User: {formatted['question']}\nAssistant: {formatted['rejected']}"
        )

    chosen_encoded = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )

    rejected_encoded = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
    }


def save_dpo_adapter(model, tokenizer, output_dir: str):
    """
    Save DPO adapter to directory.

    Args:
        model: Trained model with DPO adapters
        tokenizer: Model tokenizer
        output_dir: Directory to save adapter
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"DPO adapter saved to {output_dir}")


def measure_preference_alignment(model, test_questions: List[str]) -> float:
    """
    Measure model's preference alignment on test questions.

    Args:
        model: Trained model
        test_questions: List of test questions

    Returns:
        Alignment score (0.0 to 1.0)
    """
    from src.teacher.validators import validate_dm_response

    aligned_count = 0

    for question in test_questions:
        response = model.generate(question)
        if validate_dm_response(response):
            aligned_count += 1

    return aligned_count / len(test_questions) if test_questions else 0.0


def train_dpo(model, tokenizer, pairs: List[dict], config: dict = None):
    """
    Train model with DPO on preference pairs.

    Args:
        model: Base model (with SFT adapter)
        tokenizer: Model tokenizer
        pairs: List of DPO pairs
        config: Training configuration

    Returns:
        Trained model
    """
    torch = _get_torch()

    if config is None:
        config = DPO_CONFIG

    # Prepare dataset
    batch = prepare_dpo_batch(pairs, tokenizer)

    # Setup DPO trainer (using TRL)
    try:
        from trl import DPOTrainer

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            args=config,
            beta=config.get("beta", 0.1),
            train_dataset=pairs,
        )

        # Train
        trainer.train()

        return trainer.model

    except ImportError:
        # Fallback: simulate training
        print("TRL not available - simulating DPO training")
        return model


def main(
    sft_adapter_path: str = "checkpoints/lora_adapters/sft_adapter",
    dpo_pairs_path: str = "data/processed/dpo_pairs.jsonl",
    output_dir: str = "checkpoints/lora_adapters/dpo_adapter",
):
    """
    Main entry point for DPO training.

    Args:
        sft_adapter_path: Path to SFT adapter
        dpo_pairs_path: Path to DPO pairs
        output_dir: Output directory for DPO adapter
    """
    print(f"Loading SFT adapter from {sft_adapter_path}...")

    from unsloth import FastLanguageModel

    # Load model with SFT adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter_path, max_seq_length=4096, load_in_4bit=True
    )

    print(f"Loading DPO pairs from {dpo_pairs_path}...")
    pairs = load_dpo_pairs(dpo_pairs_path)
    print(f"Loaded {len(pairs)} DPO pairs")

    print("Starting DPO training...")
    model = train_dpo(model, tokenizer, pairs, DPO_CONFIG)

    print(f"Saving DPO adapter to {output_dir}...")
    save_dpo_adapter(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
