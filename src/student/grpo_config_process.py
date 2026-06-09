"""
GRPO v4 Training Configuration (Process Rewards - RLVMR Dual Advantage)

Dual-advantage GRPO training with outcome + process rewards, KL regularization,
and RLVMR tagged output format.

NOTE: Before running v4 training, you MUST merge the cold-start SFT
adapter into the base model first. The pipeline is:

  1. Run cold-start SFT:
     python3 -m src.student.train_cold_start_sft \\
         --data data/processed/cold_start_sft.jsonl \\
         --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \\
         --output checkpoints/lora_adapters/cold_start_sft

  2. Merge the cold-start adapter (CPU-only):
     python3 scripts/merge_grpo_checkpoint.py \\
         --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \\
         --grpo-checkpoint checkpoints/lora_adapters/cold_start_sft \\
         --output checkpoints/merged/cold_start_merged

  3. Point base_model below to the merged checkpoint:
     "base_model": "checkpoints/merged/cold_start_merged"

The paper's ablation shows removing cold-start SFT costs 15.7pp on ALFWorld L2.
"""

from pathlib import Path
from typing import Any, Optional

from trl import GRPOConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# v4-specific hyperparameters (per RLVMR paper)
REWARD_WEIGHTS: dict[str, float] = {
    "alpha": 0.5,
    "lambda_kl": 0.01,
    "clip_epsilon": 0.2,
    "lambda_format": -0.1,
}

REQUIRED_TAGS: list[str] = ["planning", "commitment", "reflection", "monitor"]


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
    "output_dir": str(_project_root() / "checkpoints/lora_adapters/grpo_v4_process"),
    # Training hyperparameters (mirrored from create_grpo_config)
    "grpo_g": 8,
    "beta": REWARD_WEIGHTS["lambda_kl"],
    "learning_rate": 5e-7,
    "max_steps": 1000,
    "max_completion_length": 512,
    "warmup_steps": 100,
    "save_steps": 100,
    "logging_steps": 25,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    # v4-specific
    "alpha": 0.5,
    "lambda_kl": REWARD_WEIGHTS["lambda_kl"],
    "clip_epsilon": REWARD_WEIGHTS["clip_epsilon"],
    "required_tags": REQUIRED_TAGS,
    "lambda_format": REWARD_WEIGHTS["lambda_format"],
}


def create_grpo_config(
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    logging_steps: Optional[int] = None,
    torch_compile: bool = False,
) -> GRPOConfig:
    """Build a GRPOConfig for v4 process-reward training.

    Args:
        output_dir: Override default output directory.
        max_steps: Override default step count (use 1 for smoke tests).
        save_steps: Override checkpoint interval (use 99999 to disable).
        logging_steps: Override logging interval (use 1 for smoke tests).
    """
    return GRPOConfig(
        learning_rate=5e-7,
        max_steps=max_steps if max_steps is not None else 1000,
        warmup_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=512,
        beta=REWARD_WEIGHTS["lambda_kl"],
        epsilon=REWARD_WEIGHTS["clip_epsilon"],
        loss_type="dapo",
        scale_rewards="group",
        logging_steps=logging_steps if logging_steps is not None else 25,
        save_steps=save_steps if save_steps is not None else 100,
        lr_scheduler_type="cosine",
        torch_compile=torch_compile,
        output_dir=output_dir or DEFAULT_CONFIG["output_dir"],
        report_to="wandb",
        remove_unused_columns=False,
        generation_batch_size=8,
    )
