"""
SFT Training Configuration

DEPRECATED: Configuration moved to Studio YAML format.
See configs/studio_sft_config.yaml for the Studio-compatible config.
Keep as reference only.

Hyperparameters and configuration for QLoRA supervised fine-tuning
of Qwen 3.5 9B model on RTX 5090 (32GB VRAM).
"""

SFT_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3.5-9B",
    "max_seq_length": 4096,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "gradient_checkpointing": "unsloth",
    # LoRA
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_steps": 1000,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    # Output
    "output_dir": "checkpoints/lora_adapters/sft_adapter",
}


def get_compute_dtype():
    """Get the compute dtype for 4-bit quantization."""
    import torch

    return torch.bfloat16
