"""
DPO Training Configuration

Hyperparameters for Direct Preference Optimization training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime.
"""

DPO_CONFIG = {
    # Base model (SFT v2 adapter)
    "base_model": "checkpoints/lora_adapters/sft_v2_adapter",
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
    "logging_steps": 25,
    "save_steps": 100,
}
