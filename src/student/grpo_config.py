"""
GRPO Training Configuration

Hyperparameters for Group Relative Policy Optimization training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime via Unsloth.
Starting from SFT merged BF16 checkpoint.
"""

GRPO_CONFIG = {
    # Base model (SFT merged checkpoint)
    "base_model": "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 500,
    "warmup_steps": 50,
    "lr_scheduler_type": "cosine",

    # GRPO-specific
    "grpo_g": 8,
    "grpo_loss_type": "dapo",
    "max_completion_length": 1024,
    "beta": 0.0,

    # Reward weights (must sum to 1.0)
    "reward_weights": {
        "dm_alignment": 0.50,
        "directional_assertion": 0.20,
        "format": 0.15,
        "length": 0.15,
    },

    # Judge model
    "judge_model": "Qwen/Qwen3.5-4B",

    # Data
    "questions_path": "data/raw/questions.json",

    # Output
    "output_dir": "checkpoints/lora_adapters/grpo_adapter",
    "logging_steps": 25,
    "save_steps": 100,
}
