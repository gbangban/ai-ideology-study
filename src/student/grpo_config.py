"""
GRPO Training Configuration

Hyperparameters for Unsloth GRPOTrainer-based GRPO training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime via Unsloth.
full rewrite: uses GRPOConfig instead of flat dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from trl import GRPOConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


REWARD_WEIGHTS: dict[str, float] = {
    "dm_alignment": 0.45,
    "directional_assertion": 0.30,
    "mechanism_commitment": 0.25,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "questions_path": str(_project_root() / "data/raw/questions.json"),
    "output_dir": str(_project_root() / "checkpoints/lora_adapters/grpo_adapter_v2"),
}


def create_grpo_config(output_dir: str | None = None) -> GRPOConfig:
    """Build a GRPOConfig for Unsloth's GRPOTrainer.

    Args:
        output_dir: Override default output directory.
    """
    return GRPOConfig(
        learning_rate=5e-7,
        max_steps=500,
        warmup_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=512,
        beta=0.1,
        epsilon=0.2,
        loss_type="dapo",
        scale_rewards="group",
        logging_steps=25,
        save_steps=50,
        lr_scheduler_type="cosine",
        max_prompt_length=2048,
        output_dir=output_dir or DEFAULT_CONFIG["output_dir"],
        report_to="wandb",
        remove_unused_columns=False,
        generation_batch_size=8,
    )
