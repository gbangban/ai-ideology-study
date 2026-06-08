"""
GRPO v3 Training Configuration (Outcome Rewards - Correctness-Based)

Outcome-only GRPO training on EconCausal + Corr2Cause + synthetic data.
Flat advantage: single group-relative normalization of outcome rewards.
Serves as the control condition for v4.

NOTE: Before running v3 training, you MUST merge the cold-start SFT
adapter into the base model first. See merge steps in grpo_config_process.py.
"""

from pathlib import Path
from typing import Any

from trl import GRPOConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "checkpoints/merged/cold_start_merged",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "dataset_path": str(_project_root() / "data/processed/grpo_train_merged.jsonl"),
    "output_dir": str(_project_root() / "checkpoints/lora_adapters/grpo_v3_outcome"),
    # Training hyperparameters (mirrored from create_grpo_config)
    "grpo_g": 8,
    "beta": 0.1,
    "learning_rate": 5e-7,
    "max_steps": 1000,
    "max_completion_length": 512,
    "warmup_steps": 100,
    "save_steps": 100,
    "logging_steps": 25,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
}


def create_grpo_config(output_dir: str | None = None) -> GRPOConfig:
    """Build a GRPOConfig for v3 outcome-reward training.

    Args:
        output_dir: Override default output directory.
    """
    return GRPOConfig(
        learning_rate=5e-7,
        max_steps=1000,
        warmup_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=512,
        beta=0.1,
        epsilon=0.2,
        loss_type="dapo",
        scale_rewards="group",
        logging_steps=25,
        save_steps=100,
        lr_scheduler_type="cosine",
        max_prompt_length=2048,
        output_dir=output_dir or DEFAULT_CONFIG["output_dir"],
        report_to="wandb",
        remove_unused_columns=False,
        generation_batch_size=8,
    )
