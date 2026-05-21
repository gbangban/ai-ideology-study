import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSFTConfigV2:
    """Test programmatic SFT configuration."""

    def test_model_name(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG["model_name"] == "Qwen/Qwen3.5-9B"

    def test_lora_params(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG["lora_r"] == 32
        assert SFT_CONFIG["lora_alpha"] == 32
        assert SFT_CONFIG["lora_dropout"] == 0.05

    def test_neftune_enabled(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG.get("neftune_noise_alpha") is not None
        assert SFT_CONFIG["neftune_noise_alpha"] > 0

    def test_target_modules(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        modules = SFT_CONFIG["target_modules"]
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(modules) == expected


class TestTrainSFTV2:
    """Test SFT v2 training script functions."""

    def test_prepare_model_for_training(self):
        """Test that model preparation applies LoRA correctly."""
        import sys
        mock_flm_class = MagicMock()
        mock_flm_class.get_peft_model.return_value = Mock()
        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel = mock_flm_class
        sys.modules["unsloth"] = mock_unsloth

        from src.student.train_sft_v2 import prepare_model_for_training

        mock_model = Mock()
        mock_tokenizer = Mock()

        prepare_model_for_training(mock_model, mock_tokenizer, {
            "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
            "target_modules": ["q_proj"],
        })
        mock_flm_class.get_peft_model.assert_called_once()
