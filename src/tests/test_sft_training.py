"""
SFT Training Tests - QLoRA Supervised Fine-Tuning

DEPRECATED: Most SFT testing is now handled by Studio UI validation.
These tests remain for config sanity checks and dataset format validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLoRAModules:
    """Test that LoRA is applied to correct modules."""

    def test_target_modules_in_config(self):
        """Test that config specifies correct target modules."""
        from src.student.config import SFT_CONFIG

        target_modules = SFT_CONFIG.get("target_modules", [])
        required_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        for module in required_modules:
            assert module in target_modules, f"Missing target module: {module}"

    def test_lora_rank_configured(self):
        """Test that LoRA rank is properly configured."""
        from src.student.config import SFT_CONFIG

        r = SFT_CONFIG.get("r")
        assert r is not None
        assert r in [8, 16, 32, 64, 128], f"Invalid LoRA rank: {r}"

    def test_lora_alpha_configured(self):
        """Test that LoRA alpha scaling is configured."""
        from src.student.config import SFT_CONFIG

        r = SFT_CONFIG.get("r", 16)
        lora_alpha = SFT_CONFIG.get("lora_alpha", 16)

        assert lora_alpha > 0
        assert lora_alpha == r or lora_alpha >= r


class TestTrainingHyperparameters:
    """Test training hyperparameter ranges."""

    def test_learning_rate_in_range(self):
        """Test that learning rate is in recommended range."""
        from src.student.config import SFT_CONFIG

        lr = SFT_CONFIG.get("learning_rate")

        assert 1e-5 <= lr <= 5e-4, f"Learning rate {lr} out of recommended range"

    def test_max_seq_length_reasonable(self):
        """Test that max sequence length is reasonable for VRAM."""
        from src.student.config import SFT_CONFIG

        max_seq_length = SFT_CONFIG.get("max_seq_length")

        assert 2048 <= max_seq_length <= 4096

    def test_optim_is_adamw_8bit(self):
        """Test that optimizer is adamw_8bit for memory efficiency."""
        from src.student.config import SFT_CONFIG

        optim = SFT_CONFIG.get("optim")

        assert optim == "adamw_8bit", "Should use adamw_8bit for VRAM efficiency"

    def test_scheduler_type(self):
        """Test that learning rate scheduler is configured."""
        from src.student.config import SFT_CONFIG

        scheduler_type = SFT_CONFIG.get("lr_scheduler_type")
        assert scheduler_type in ["cosine", "linear", "constant", "polynomial"]
        assert scheduler_type == "cosine", "Cosine scheduler recommended for stability"

    def test_warmup_steps_configured(self):
        """Test that warmup steps are configured."""
        from src.student.config import SFT_CONFIG

        warmup_steps = SFT_CONFIG.get("warmup_steps", 0)
        max_steps = SFT_CONFIG.get("max_steps", 1000)

        assert warmup_steps > 0, "Warmup steps should be positive"
        assert warmup_steps < max_steps * 0.2, "Warmup should be < 20% of training"


class TestBatchConfiguration:
    """Test batch size and gradient accumulation settings."""

    def test_effective_batch_size_calculated(self):
        """Test that effective batch size is reasonable."""
        from src.student.config import SFT_CONFIG

        batch_size = SFT_CONFIG.get("per_device_train_batch_size", 1)
        gradient_accumulation = SFT_CONFIG.get("gradient_accumulation_steps", 4)

        effective_batch_size = batch_size * gradient_accumulation

        assert effective_batch_size >= 4
        assert effective_batch_size <= 16

    def test_batch_size_one_for_vram(self):
        """Test that per-device batch size is 1 for VRAM efficiency."""
        from src.student.config import SFT_CONFIG

        batch_size = SFT_CONFIG.get("per_device_train_batch_size")

        assert batch_size == 1, "Batch size should be 1 for 27B model"


class TestQuantizationConfig:
    """Test 4-bit quantization settings."""

    def test_4bit_loading_enabled(self):
        """Test that 4-bit loading is enabled."""
        from src.student.config import SFT_CONFIG

        assert SFT_CONFIG.get("load_in_4bit") is True

    def test_4bit_compute_dtype_bf16(self):
        """Test that compute dtype is bfloat16."""
        from src.student.config import SFT_CONFIG

        compute_dtype = SFT_CONFIG.get("bnb_4bit_compute_dtype")

        assert compute_dtype == "bfloat16", "Should use bfloat16 for stability"

    def test_4bit_quant_type_nf4(self):
        """Test that quantization type is NF4."""
        from src.student.config import SFT_CONFIG

        quant_type = SFT_CONFIG.get("bnb_4bit_quant_type")

        assert quant_type == "nf4", "Should use NF4 quantization"


class TestDatasetLoading:
    """Test dataset loading and formatting."""

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_jsonl_dataset(self, mock_open):
        """Test loading JSONL dataset."""
        from src.student.train_sft import load_dataset

        mock_content = """{"conversations": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]}
{"conversations": [{"role": "user", "content": "Q2?"}, {"role": "assistant", "content": "A2"}]}"""

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=iter(mock_content.split("\n")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        dataset = load_dataset("mock_path.jsonl")

        assert len(dataset) == 2
        assert "conversations" in dataset[0]

    def test_dataset_format_conversion(self):
        """Test that dataset is converted to training format."""
        from src.student.train_sft import format_conversation

        conversations = [
            {"role": "user", "content": "Test question?"},
            {"role": "assistant", "content": "Test answer."},
        ]

        formatted = format_conversation(conversations)

        assert "Test question?" in formatted
        assert "Test answer." in formatted


class TestStudioConfigFormat:
    """Test that Studio YAML config is valid."""

    def test_studio_config_exists(self):
        """Test that Studio config file exists."""
        import os

        config_path = "configs/studio_sft_config.yaml"
        assert os.path.exists(config_path), "Studio config file missing"

    def test_studio_config_has_sections(self):
        """Test that Studio config has required sections."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with open("configs/studio_sft_config.yaml") as f:
            config = yaml.safe_load(f)

        assert "training" in config
        assert "lora" in config
        assert "logging" in config

    def test_studio_config_lora_values(self):
        """Test that Studio config LoRA values match legacy config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with open("configs/studio_sft_config.yaml") as f:
            studio_config = yaml.safe_load(f)

        from src.student.config import SFT_CONFIG

        assert studio_config["lora"]["r"] == SFT_CONFIG["r"]
        assert studio_config["lora"]["lora_alpha"] == SFT_CONFIG["lora_alpha"]
        assert studio_config["lora"]["lora_dropout"] == SFT_CONFIG["lora_dropout"]
        assert set(studio_config["lora"]["target_modules"]) == set(
            SFT_CONFIG["target_modules"]
        )

    def test_studio_config_training_values(self):
        """Test that Studio config training values match legacy config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with open("configs/studio_sft_config.yaml") as f:
            studio_config = yaml.safe_load(f)

        from src.student.config import SFT_CONFIG

        assert (
            studio_config["training"]["learning_rate"] == SFT_CONFIG["learning_rate"]
        )
        assert (
            studio_config["training"]["max_steps"] == SFT_CONFIG["max_steps"]
        )
        assert (
            studio_config["training"]["per_device_train_batch_size"]
            == SFT_CONFIG["per_device_train_batch_size"]
        )
        assert (
            studio_config["training"]["gradient_accumulation_steps"]
            == SFT_CONFIG["gradient_accumulation_steps"]
        )
        assert (
            studio_config["training"]["lr_scheduler_type"]
            == SFT_CONFIG["lr_scheduler_type"]
        )
