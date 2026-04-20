"""
DPO Training Configuration

Hyperparameters for Direct Preference Optimization training
of Qwen 3.5 27B model on RTX 5090 (32GB VRAM).
"""

DPO_CONFIG = {
    # Base model (SFT adapter)
    "base_model": "checkpoints/lora_adapters/sft_adapter",
    # DPO-specific
    "beta": 0.1,
    "dpo_loss": "sigmoid",
    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 500,
    "warmup_steps": 50,
    "lr_scheduler_type": "cosine",
    # Output
    "output_dir": "checkpoints/lora_adapters/dpo_adapter",
}
