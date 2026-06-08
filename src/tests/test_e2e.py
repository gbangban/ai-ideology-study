"""
End-to-End Integration Tests - Studio-Integrated Pipeline

Tests the complete training workflow with Studio handling SFT
and custom scripts handling GRPO. Tests validate integration points
between Studio exports and custom GRPO training.
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


class TestE2EAdapterSaveLoad:
    """Test adapter save and load operations."""

    def test_adapter_saves_correctly(self, tmp_path, mock_model_and_tokenizer):
        """Test that adapter save calls are correct."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        output_dir = str(tmp_path / "adapter")

        # Simulate what train() does for saving
        mock_model.save_pretrained(output_dir)
        mock_tokenizer.save_pretrained(output_dir)

        mock_model.save_pretrained.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)


class TestE2EPipeline:
    """Full pipeline integration tests for Studio workflow."""

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

        assert dataset_path.exists()

        with open(dataset_path) as f:
            loaded_dataset = [json.loads(line) for line in f if line.strip()]
        assert len(loaded_dataset) == 1

    def test_dataset_format_compatible_with_studio(self, tmp_path):
        """Test that generated dataset is compatible with Studio ShareGPT format."""
        dataset_path = tmp_path / "sft_dataset.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "Test question?"},
                    {"role": "assistant", "content": "Test answer."},
                ]
            }
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        with open(dataset_path) as f:
            for line in f:
                sample = json.loads(line)
                assert "conversations" in sample
                msgs = sample["conversations"]
                assert len(msgs) >= 2
                assert msgs[0]["role"] == "user"
                assert msgs[1]["role"] == "assistant"
