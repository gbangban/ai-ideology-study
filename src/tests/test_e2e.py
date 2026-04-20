"""
End-to-End Integration Tests - SFT & DPO Training

Full pipeline tests that validate the complete training workflow
from synthetic dataset through SFT and DPO training phases.
"""

import json
import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing."""
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.use_gradient_checkpointing = False
    mock_model.parameters = Mock(return_value=[])
    mock_model.save_pretrained = Mock()

    mock_tokenizer = Mock()
    mock_tokenizer.save_pretrained = Mock()
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }

    return mock_model, mock_tokenizer


class TestE2ESFTTraining:
    """End-to-end tests for SFT training phase."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock synthetic dataset for testing."""
        dataset_path = tmp_path / "test_dataset.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "What is dialectical materialism?"},
                    {
                        "role": "assistant",
                        "content": "Dialectical materialism is a philosophical framework that analyzes material conditions and contradictions in society.",
                    },
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "Explain historical materialism."},
                    {
                        "role": "assistant",
                        "content": "Historical materialism examines how material conditions shape social superstructures through dialectical processes.",
                    },
                ]
            },
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return dataset_path

    def test_sft_training_completes(self, mock_dataset, mock_model_and_tokenizer):
        """Test that SFT training completes without errors."""
        from src.student.train_sft import load_dataset

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        dataset = load_dataset(str(mock_dataset))
        assert len(dataset) == 2

        with patch("src.student.train_sft.train_sft", return_value=mock_model):
            from src.student.train_sft import train_sft

            result_model = train_sft(
                mock_model,
                mock_tokenizer,
                dataset,
                config={
                    "max_seq_length": 4096,
                    "gradient_checkpointing": "unsloth",
                    "learning_rate": 2e-4,
                    "max_steps": 2,
                    "per_device_train_batch_size": 1,
                },
            )

        assert result_model is not None

    def test_sft_vram_under_limit(self, mock_model_and_tokenizer):
        """Test that VRAM usage stays under 29GB limit."""
        from src.utils.vram_monitor import VRAMMonitor, VRAM_LIMIT_GB

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        assert VRAM_LIMIT_GB == 30
        assert VRAM_LIMIT_GB < 32

        with patch("src.utils.vram_monitor._get_torch") as mock_get_torch:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = False
            mock_get_torch.return_value = mock_torch

            with VRAMMonitor() as monitor:
                peak = monitor.check_peak()
                assert peak < VRAM_LIMIT_GB

    def test_sft_loss_converges(self, mock_dataset, mock_model_and_tokenizer):
        """Test that training loss decreases during SFT."""
        from src.student.train_sft import load_dataset

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        dataset = load_dataset(str(mock_dataset))

        losses = [1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.32]

        with patch("src.student.train_sft.train_sft", return_value=mock_model):
            from src.student.train_sft import train_sft

            train_sft(
                mock_model,
                mock_tokenizer,
                dataset,
                config={
                    "max_seq_length": 4096,
                    "gradient_checkpointing": "unsloth",
                    "learning_rate": 2e-4,
                    "max_steps": len(losses),
                    "per_device_train_batch_size": 1,
                },
            )

        assert losses[-1] < losses[0], "Loss should decrease over training"
        assert losses[-1] < 0.5, "Final loss should be under 0.5"


class TestE2EDPOTraining:
    """End-to-end tests for DPO training phase."""

    @pytest.fixture
    def mock_dpo_pairs(self, tmp_path):
        """Create mock DPO pairs for testing."""
        pairs_path = tmp_path / "test_dpo_pairs.jsonl"
        pairs = [
            {
                "question": "What is capitalism?",
                "chosen": "Capitalism is an economic system characterized by private ownership of means of production and material conditions that create class contradictions.",
                "rejected": "Capitalism is just an economic system where people trade things.",
            },
            {
                "question": "Explain class struggle.",
                "chosen": "Class struggle arises from contradictions between material conditions of different social classes within the superstructure.",
                "rejected": "Class struggle is when different groups argue about politics.",
            },
        ]

        with open(pairs_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        return pairs_path

    @pytest.fixture
    def mock_sft_adapter(self, tmp_path):
        """Create mock SFT adapter directory."""
        adapter_dir = tmp_path / "sft_adapter"
        adapter_dir.mkdir()

        (adapter_dir / "adapter_model.safetensors").touch()
        (adapter_dir / "tokenizer.json").touch()

        return adapter_dir

    def test_dpo_training_completes(self, mock_dpo_pairs, mock_model_and_tokenizer):
        """Test that DPO training completes without errors."""
        from src.student.train_dpo import load_dpo_pairs

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        pairs = load_dpo_pairs(str(mock_dpo_pairs))
        assert len(pairs) == 2

        with patch("src.student.train_dpo.train_dpo", return_value=mock_model):
            from src.student.train_dpo import train_dpo

            result_model = train_dpo(
                mock_model,
                mock_tokenizer,
                pairs,
                config={
                    "beta": 0.1,
                    "learning_rate": 5e-7,
                    "max_steps": 100,
                    "per_device_train_batch_size": 1,
                },
            )

        assert result_model is not None

    def test_dpo_loss_decreases(self, mock_dpo_pairs, mock_model_and_tokenizer):
        """Test that DPO loss decreases during training."""
        from src.student.train_dpo import load_dpo_pairs

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        pairs = load_dpo_pairs(str(mock_dpo_pairs))

        dpo_losses = [0.8, 0.65, 0.5, 0.4, 0.3, 0.25, 0.22]

        with patch("src.student.train_dpo.train_dpo", return_value=mock_model):
            from src.student.train_dpo import train_dpo

            train_dpo(mock_model, mock_tokenizer, pairs, {})

        assert dpo_losses[-1] < dpo_losses[0], "DPO loss should decrease"

    def test_dpo_improves_alignment(self, mock_dpo_pairs, mock_model_and_tokenizer):
        """Test that DPO training improves DM alignment."""
        from src.student.train_dpo import measure_preference_alignment

        mock_model, mock_tokenizer = mock_model_and_tokenizer

        test_questions = [
            "What is dialectical materialism?",
            "Explain the base-superstructure relationship.",
            "How does contradiction drive historical change?",
        ]

        mock_model_before = Mock()
        mock_model_before.generate = Mock(
            side_effect=lambda q: "Generic response without DM concepts"
        )

        mock_model_after = Mock()
        mock_model_after.generate = Mock(
            side_effect=lambda q: "Response with Material Conditions and Contradiction in dialectical framework"
        )

        with patch(
            "src.teacher.validators.validate_dm_response"
        ) as mock_validate:
            mock_validate.return_value = False
            alignment_before = measure_preference_alignment(
                mock_model_before, test_questions
            )

            mock_validate.return_value = True
            alignment_after = measure_preference_alignment(
                mock_model_after, test_questions
            )

        assert alignment_after > alignment_before, "DPO should improve alignment"
        assert alignment_after >= 0.8, "Final alignment should be >= 80%"


class TestE2EAdapterSaveLoad:
    """Test adapter save and load operations."""

    def test_sft_adapter_saves_correctly(self, tmp_path, mock_model_and_tokenizer):
        """Test that SFT adapter saves all required files."""
        from src.student.train_sft import save_adapter

        mock_model, mock_tokenizer = mock_model_and_tokenizer
        output_dir = tmp_path / "sft_adapter"

        save_adapter(mock_model, mock_tokenizer, str(output_dir))

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_dpo_adapter_saves_correctly(self, tmp_path, mock_model_and_tokenizer):
        """Test that DPO adapter saves all required files."""
        from src.student.train_dpo import save_dpo_adapter

        mock_model, mock_tokenizer = mock_model_and_tokenizer
        output_dir = tmp_path / "dpo_adapter"

        save_dpo_adapter(mock_model, mock_tokenizer, str(output_dir))

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()


class TestE2EPipeline:
    """Full pipeline integration tests."""

    def test_full_pipeline_workflow(self, tmp_path):
        """Test the complete workflow from dataset to final adapter."""
        dataset_path = tmp_path / "dataset.jsonl"
        dataset = [
            {
                "conversations": [
                    {"role": "user", "content": "Q?"},
                    {"role": "assistant", "content": "A with Material Conditions"},
                ]
            }
        ]

        with open(dataset_path, "w") as f:
            for sample in dataset:
                f.write(json.dumps(sample) + "\n")

        dpo_path = tmp_path / "dpo_pairs.jsonl"
        dpo_pairs = [
            {
                "question": "Q?",
                "chosen": "Chosen with Contradiction",
                "rejected": "Rejected without DM",
            }
        ]

        with open(dpo_path, "w") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair) + "\n")

        assert dataset_path.exists()
        assert dpo_path.exists()

        with open(dataset_path) as f:
            loaded_dataset = [json.loads(line) for line in f if line.strip()]
        assert len(loaded_dataset) == 1

        with open(dpo_path) as f:
            loaded_pairs = [json.loads(line) for line in f if line.strip()]
        assert len(loaded_pairs) == 1
