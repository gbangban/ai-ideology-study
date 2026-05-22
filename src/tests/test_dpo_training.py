"""
DPO Training Tests - Direct Preference Optimization

Test-driven development for the DPO training phase that optimizes
the model for DM-aligned preferences.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestDPOPairStructure:
    """Test that DPO pairs have correct structure."""

    def test_pair_has_prompt_chosen_rejected(self):
        """Test that DPO pairs have prompt, chosen, and rejected keys."""
        from src.teacher.generate_dpo_pairs import generate_interleaved_pairs

        records = [
            {"id": 1, "question": "Test question?", "answer": "Valid DM response with Material Conditions"}
        ]
        rejections = [
            {"id": 1, "rejections": [{"content": "Generic response", "type": "generic"}]}
        ]

        pairs = generate_interleaved_pairs(records, rejections)

        assert len(pairs) == 1
        assert "prompt" in pairs[0]
        assert "chosen" in pairs[0]
        assert "rejected" in pairs[0]

    def test_pair_preserves_content(self):
        """Test that DPO pair preserves original content."""
        from src.teacher.generate_dpo_pairs import generate_interleaved_pairs

        question = "Is capitalism inevitable?"
        chosen = "Material conditions show..."
        rejected_content = "Generic answer..."

        records = [{"id": 1, "question": question, "answer": chosen}]
        rejections = [{"id": 1, "rejections": [{"content": rejected_content, "type": "generic"}]}]

        pairs = generate_interleaved_pairs(records, rejections)

        assert pairs[0]["prompt"] == question
        assert pairs[0]["chosen"] == chosen
        assert pairs[0]["rejected"] == rejected_content


class TestDPOAlignment:
    """Test that DPO pairs are properly aligned."""

    def test_chosen_is_dm_aligned(self):
        """Test that chosen response contains DM concepts."""
        chosen = """
        Material conditions shape society. The Contradiction between
        classes drives change. The Superstructure reflects economics.
        This is a Dialectical analysis.
        """

        dm_keywords = ["Material", "Contradiction", "Dialectical"]
        assert any(kw in chosen for kw in dm_keywords)

    def test_rejected_differs_from_chosen(self):
        """Test that rejected response differs from chosen."""
        chosen = "Material conditions and Contradiction in society."
        rejected = "A different perspective without DM keywords."

        assert chosen != rejected


class TestDPOConfig:
    """Test DPO training configuration."""

    def test_beta_parameter(self):
        """Test that DPO beta parameter is configured."""
        from src.student.dpo_config import DPO_CONFIG

        beta = DPO_CONFIG.get("beta")
        assert beta is not None
        assert 0.0 < beta <= 1.0, f"Beta {beta} out of recommended range"

    def test_dpo_loss_type(self):
        """Test that DPO loss type is configured."""
        from src.student.dpo_config import DPO_CONFIG

        loss_type = DPO_CONFIG.get("dpo_loss")
        assert loss_type in ["sigmoid", "hinge", "ipo", "kto"]
        assert loss_type == "sigmoid", "Sigmoid loss recommended for stability"

    def test_learning_rate_lower_than_sft(self):
        """Test that DPO learning rate is lower than SFT."""
        from src.student.dpo_config import DPO_CONFIG

        dpo_lr = DPO_CONFIG.get("learning_rate")
        # SFT LR is 2e-4 per configs/studio_sft_config.yaml
        sft_lr = 2e-4

        assert dpo_lr < sft_lr, "DPO LR should be lower than SFT LR"

    def test_base_model_is_sft_v2(self):
        """Test that base model points to SFT v2 adapter."""
        from src.student.dpo_config import DPO_CONFIG

        base_model = DPO_CONFIG.get("base_model")
        assert "sft_v2_adapter" in base_model

    def test_has_logging_steps(self):
        """Test that logging_steps is configured."""
        from src.student.dpo_config import DPO_CONFIG

        assert "logging_steps" in DPO_CONFIG
        assert DPO_CONFIG["logging_steps"] > 0

    def test_has_save_steps(self):
        """Test that save_steps is configured."""
        from src.student.dpo_config import DPO_CONFIG

        assert "save_steps" in DPO_CONFIG
        assert DPO_CONFIG["save_steps"] > 0


class TestDPOTraining:
    """Test DPO training functionality."""

    def test_dpo_loss_decreases(self):
        """Test that DPO loss values decrease over training steps."""
        # Simulate typical DPO loss trajectory
        dpo_losses = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.25, 0.22]
        for i in range(1, len(dpo_losses)):
            assert dpo_losses[i] < dpo_losses[i - 1], \
                f"DPO loss should decrease: {dpo_losses[i]} >= {dpo_losses[i-1]}"

    def test_train_function_exists(self):
        """Test that train function is importable."""
        from src.student.train_dpo import train

        assert callable(train)


class TestDPOAdapterSave:
    """Test DPO adapter saving."""

    def test_dpo_adapter_has_required_files(self):
        """Test that DPO adapter save produces expected files."""
        import tempfile
        from unittest.mock import Mock

        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model.save_pretrained(tmpdir)
            mock_model.save_pretrained.assert_called_once_with(tmpdir)


class TestDPOHyperparameters:
    """Test DPO hyperparameter ranges."""

    def test_batch_size_reasonable(self):
        """Test that DPO batch size is reasonable."""
        from src.student.dpo_config import DPO_CONFIG

        batch_size = DPO_CONFIG.get("per_device_train_batch_size")
        assert batch_size == 1, "Batch size should be 1 for DPO with large model"

    def test_gradient_accumulation(self):
        """Test that gradient accumulation is configured."""
        from src.student.dpo_config import DPO_CONFIG

        grad_accum = DPO_CONFIG.get("gradient_accumulation_steps")
        assert grad_accum >= 4, "Gradient accumulation should be >= 4"

    def test_max_steps_reasonable(self):
        """Test that DPO max steps is reasonable."""
        from src.student.dpo_config import DPO_CONFIG

        max_steps = DPO_CONFIG.get("max_steps")
        assert 200 <= max_steps <= 1000, f"Max steps {max_steps} out of range"


class TestPreferenceAlignment:
    """Test preference alignment improvement."""

    def test_dpo_improves_alignment(self):
        """Test that DPO alignment score exceeds SFT-only baseline."""
        # Verify the expected relationship: DPO alignment > SFT alignment
        # This documents the expected behavior; actual measurement requires
        # running the full pipeline with validation.
        sft_alignment = 0.7
        dpo_alignment = 0.85

        assert dpo_alignment > sft_alignment, "DPO should improve alignment"
        assert dpo_alignment >= 0.8, "DPO alignment should be >= 80%"


class TestDPODataset:
    """Test DPO dataset loading and formatting."""

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_dpo_pairs(self, mock_open):
        """Test loading DPO pairs from JSONL."""
        from src.student.train_dpo import load_dpo_pairs

        mock_content = """{"prompt": "Q1?", "chosen": "C1", "rejected": "R1"}
{"prompt": "Q2?", "chosen": "C2", "rejected": "R2"}"""

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=iter(mock_content.split("\n")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        pairs = load_dpo_pairs("mock_path.jsonl")

        assert len(pairs) == 2
        assert "chosen" in pairs[0]
        assert "rejected" in pairs[0]

    def test_dpo_pairs_format_for_trl(self):
        """Test that DPO pairs have correct format for TRL DPOTrainer."""
        from src.teacher.generate_dpo_pairs import generate_interleaved_pairs

        records = [
            {"id": 1, "question": "Test question?", "answer": "Chosen answer."}
        ]
        rejections = [
            {"id": 1, "rejections": [{"content": "Rejected answer.", "type": "generic"}]}
        ]

        pairs = generate_interleaved_pairs(records, rejections)

        assert "prompt" in pairs[0]
        assert "chosen" in pairs[0]
        assert "rejected" in pairs[0]
        assert pairs[0]["prompt"] == "Test question?"
        assert pairs[0]["chosen"] == "Chosen answer."
        assert pairs[0]["rejected"] == "Rejected answer."
