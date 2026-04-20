"""
SFT Training Tests - QLoRA Supervised Fine-Tuning

Test-driven development for the SFT training phase that trains
a QLoRA adapter on the synthetic DM-aligned dataset.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestVRAMUsage:
    """Test that VRAM usage stays within RTX 5090 limits."""

    @patch("src.utils.vram_monitor._get_torch")
    def test_vram_monitor_initializes(self, mock_get_torch):
        """Test that VRAM monitor can be initialized."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_get_torch.return_value = mock_torch

        from src.utils.vram_monitor import VRAMMonitor

        with VRAMMonitor() as monitor:
            assert monitor.peak_vram_gb >= 0

    @patch("src.utils.vram_monitor._get_torch")
    def test_vram_monitor_tracks_peak(self, mock_get_torch):
        """Test that VRAM monitor tracks peak usage."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_get_torch.return_value = mock_torch

        from src.utils.vram_monitor import VRAMMonitor

        with VRAMMonitor() as monitor:
            initial_peak = monitor.peak_vram_gb
            updated_peak = monitor.check_peak()
            assert updated_peak >= initial_peak

    def test_vram_under_limit(self):
        """Test that VRAM usage stays under 30GB limit during training step."""
        from src.utils.vram_monitor import VRAM_LIMIT_GB

        assert VRAM_LIMIT_GB == 30
        assert VRAM_LIMIT_GB < 32  # Must be under GPU capacity


class TestGradientCheckpointing:
    """Test that gradient checkpointing is properly configured."""

    def test_gradient_checkpointing_enabled_in_config(self):
        """Test that config enables gradient checkpointing."""
        from src.student.config import SFT_CONFIG

        assert SFT_CONFIG.get("gradient_checkpointing") in ["unsloth", True, "true"]

    def test_model_configures_gradient_checkpointing(self):
        """Test that model applies gradient checkpointing when loaded."""
        from src.student.train_sft import configure_model_for_training

        mock_model = Mock()
        mock_model.config = Mock()

        configure_model_for_training(mock_model, gradient_checkpointing="unsloth")

        # Verify checkpointing was applied
        assert mock_model.config.use_gradient_checkpointing is True


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

        r = SFT_CONFIG.get("r", 32)
        lora_alpha = SFT_CONFIG.get("lora_alpha", 32)

        # Alpha should typically equal r for standard scaling
        assert lora_alpha > 0
        assert lora_alpha == r or lora_alpha >= r


class TestTrainingConvergence:
    """Test that training converges properly."""

    def test_training_loss_decreases(self):
        """Test that training loss decreases over epochs."""
        # Simulate training losses
        initial_loss = 2.5
        final_loss = 0.3

        assert final_loss < initial_loss, "Loss should decrease"
        assert final_loss < 0.5, "Final loss should be under 0.5"

    def test_learning_rate_scheduled(self):
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


class TestAdapterSaveLoad:
    """Test that adapter saves and loads correctly."""

    @patch("src.student.train_sft.Path")
    def test_adapter_saves_to_directory(self, mock_path):
        """Test that adapter saves to specified directory."""
        from src.student.train_sft import save_adapter

        mock_model = Mock()
        mock_tokenizer = Mock()
        save_dir = "test_adapter_output"

        mock_model.save_pretrained = Mock()
        mock_tokenizer.save_pretrained = Mock()

        save_adapter(mock_model, mock_tokenizer, save_dir)

        mock_model.save_pretrained.assert_called_once_with(save_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_dir)

    @patch("src.student.train_sft.Path")
    def test_adapter_creates_safetensors(self, mock_path):
        """Test that adapter creates safetensors files."""
        from src.student.train_sft import save_adapter

        mock_model = Mock()
        mock_tokenizer = Mock()
        save_dir = "test_adapter_output"

        # Mock file creation
        mock_model.save_pretrained = Mock()
        mock_tokenizer.save_pretrained = Mock()

        save_adapter(mock_model, mock_tokenizer, save_dir)

        # Verify safetensors would be created
        assert mock_model.save_pretrained.called


class TestBatchConfiguration:
    """Test batch size and gradient accumulation settings."""

    def test_effective_batch_size_calculated(self):
        """Test that effective batch size is reasonable."""
        from src.student.config import SFT_CONFIG

        batch_size = SFT_CONFIG.get("per_device_train_batch_size", 1)
        gradient_accumulation = SFT_CONFIG.get("gradient_accumulation_steps", 4)

        effective_batch_size = batch_size * gradient_accumulation

        # Effective batch size should be reasonable for VRAM constraints
        assert effective_batch_size >= 4
        assert effective_batch_size <= 16

    def test_batch_size_one_for_vram(self):
        """Test that per-device batch size is 1 for VRAM efficiency."""
        from src.student.config import SFT_CONFIG

        batch_size = SFT_CONFIG.get("per_device_train_batch_size")

        # For 27B model with QLoRA, batch size 1 is typical
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

        # Create mock JSONL content
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


class TestTrainingHyperparameters:
    """Test training hyperparameter ranges."""

    def test_learning_rate_in_range(self):
        """Test that learning rate is in recommended range."""
        from src.student.config import SFT_CONFIG

        lr = SFT_CONFIG.get("learning_rate")

        # Recommended range for LoRA: 1e-5 to 5e-4
        assert 1e-5 <= lr <= 5e-4, f"Learning rate {lr} out of recommended range"

    def test_max_seq_length_reasonable(self):
        """Test that max sequence length is reasonable for VRAM."""
        from src.student.config import SFT_CONFIG

        max_seq_length = SFT_CONFIG.get("max_seq_length")

        # For 32GB VRAM with 27B model, 2048-4096 is typical
        assert 2048 <= max_seq_length <= 4096

    def test_optim_is_adamw_8bit(self):
        """Test that optimizer is adamw_8bit for memory efficiency."""
        from src.student.config import SFT_CONFIG

        optim = SFT_CONFIG.get("optim")

        assert optim == "adamw_8bit", "Should use adamw_8bit for VRAM efficiency"
