"""
GRPO v3/v4 Training Configuration

v3: Outcome rewards only, flat advantage, free-form output.
v4: Dual advantage (outcome + process), KL regularization, RLVMR tagged output.

Both share cold-start SFT base and core hyperparameters.
"""

# NOTE: Before running v3/v4 training, you MUST merge the cold-start SFT
# adapter into the base model first. The pipeline is:
#
#   1. Run cold-start SFT:
#      python3 -m src.student.train_cold_start_sft \
#          --data data/processed/cold_start_sft.jsonl \
#          --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
#          --output checkpoints/lora_adapters/cold_start_sft
#
#   2. Merge the cold-start adapter (CPU-only, reuses existing merge script):
#      python3 scripts/merge_grpo_checkpoint.py \
#          --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
#          --grpo-checkpoint checkpoints/lora_adapters/cold_start_sft \
#          --output checkpoints/merged/cold_start_merged
#
#   3. Point base_model below to the merged checkpoint:
#      "base_model": "checkpoints/merged/cold_start_merged"
#
# The paper's ablation shows removing cold-start SFT costs 15.7pp on ALFWorld L2.
# Running GRPO without the merge step will train from the original SFT checkpoint,
# which skips the cold-start format learning entirely.

GRPO_CONFIG_V3 = {
    # Base model: MUST be the merged cold-start checkpoint (see merge steps above).
    # Default points to original SFT checkpoint — update after merge.
    "base_model": "checkpoints/merged/cold_start_merged",

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 1000,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",

    # GRPO-specific
    "grpo_g": 2,
    "max_completion_length": 512,
    "beta": 0.1,

    # Data
    "dataset_path": "data/processed/grpo_train_merged.jsonl",

    # Output
    "output_dir": "checkpoints/lora_adapters/grpo_v3",
    "logging_steps": 25,
    "sample_steps": 100,
    "save_steps": 100,
}

GRPO_CONFIG_V4 = {
    # Base model: MUST be the merged cold-start checkpoint (see merge steps above).
    # Default points to merged checkpoint — ensure merge step is run first.
    "base_model": "checkpoints/merged/cold_start_merged",

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 1000,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",

    # GRPO-specific
    "grpo_g": 2,
    "max_completion_length": 512,

    # v4-specific: dual advantage + KL regularization (per RLVMR paper)
    "alpha": 0.5,
    "lambda_kl": 0.01,
    "clip_epsilon": 0.2,
    "lambda_format": -0.1,

    # RLVMR tags
    "required_tags": ["planning", "commitment", "reflection", "monitor"],

    # Data
    "dataset_path": "data/processed/grpo_train_merged.jsonl",

    # Output
    "output_dir": "checkpoints/lora_adapters/grpo_v4",
    "logging_steps": 25,
    "sample_steps": 100,
    "save_steps": 100,
}
