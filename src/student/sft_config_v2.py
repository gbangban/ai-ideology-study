"""
SFT Training Configuration v2

Programmatic SFT training with Unsloth Core + TRL.
Replaces Studio UI-based SFT.
"""

SFT_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3.5-9B",
    "max_seq_length": 4096,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Quantization
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",

    # Neftune noise
    "neftune_noise_alpha": 5.0,

    # Training
    "learning_rate": 2e-4,
    "max_steps": 1000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "optim": "adamw_8bit",

    # Output
    "output_dir": "checkpoints/lora_adapters/sft_v2_adapter",
    "logging_steps": 50,
    "save_steps": 200,
}
