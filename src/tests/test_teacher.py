"""
Teacher Phase Tests - Synthetic Data Generation

Test-driven development for the teacher phase that generates
DM-aligned synthetic training samples using llama.cpp with GGUF models.
"""

import json
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, "/home/yao/projects/ml-lora-training")


class TestSampleStructure:
    """Test that generated samples have correct ShareGPT structure."""

    def test_sample_has_conversations_key(self):
        """Test that samples contain the conversations key."""
        from src.teacher.sample_utils import create_sample

        sample = create_sample("Test question?", "Test answer")
        assert "conversations" in sample

    def test_sample_has_two_messages(self):
        """Test that samples have exactly user and assistant messages."""
        from src.teacher.sample_utils import create_sample

        sample = create_sample("Test question?", "Test answer")
        assert len(sample["conversations"]) == 2

    def test_first_message_is_user(self):
        """Test that the first message has user role."""
        from src.teacher.sample_utils import create_sample

        sample = create_sample("Test question?", "Test answer")
        assert sample["conversations"][0]["role"] == "user"

    def test_second_message_is_assistant(self):
        """Test that the second message has assistant role."""
        from src.teacher.sample_utils import create_sample

        sample = create_sample("Test question?", "Test answer")
        assert sample["conversations"][1]["role"] == "assistant"

    def test_message_content_preserved(self):
        """Test that question and answer are correctly placed."""
        from src.teacher.sample_utils import create_sample

        question = "Is capitalism inevitable?"
        answer = "Material conditions show..."
        sample = create_sample(question, answer)

        assert sample["conversations"][0]["content"] == question
        assert sample["conversations"][1]["content"] == answer


class TestDMKeywordValidation:
    """Test that DM keywords are validated correctly."""

    def test_required_keywords_list(self):
        """Test that required DM keywords are defined."""
        from src.teacher.validators import REQUIRED_KEYWORDS

        assert isinstance(REQUIRED_KEYWORDS, list)
        assert len(REQUIRED_KEYWORDS) > 0
        assert "Material" in str(REQUIRED_KEYWORDS)

    def test_validate_response_with_all_keywords(self):
        """Test validation passes when all keywords present."""
        from src.teacher.validators import validate_dm_response

        response = """
        Material conditions shape society. The Contradiction between
        classes drives change. The Superstructure reflects economics.
        This is a Dialectical analysis.
        """
        assert validate_dm_response(response) is True

    def test_validate_response_missing_keywords(self):
        """Test validation fails when keywords missing."""
        from src.teacher.validators import validate_dm_response

        response = "This is a generic response without DM concepts."
        assert validate_dm_response(response) is False

    def test_validate_response_case_insensitive(self):
        """Test validation is case-insensitive for keywords."""
        from src.teacher.validators import validate_dm_response

        response = "material conditions and CONTRADICTION in the SUPERSTRUCTURE. This is DIALECTICAL."
        assert validate_dm_response(response) is True


class TestRetryLogic:
    """Test retry mechanism for invalid samples."""

    def test_retry_on_invalid_response(self):
        """Test that invalid responses trigger retry."""
        from src.teacher.validators import generate_with_retry, REQUIRED_KEYWORDS

        call_count = [0]

        def mock_generate():
            call_count[0] += 1
            if call_count[0] < 3:
                return "Invalid response without keywords"
            return f"Valid response with {' '.join(REQUIRED_KEYWORDS)}"

        result = generate_with_retry(mock_generate, max_retries=3)
        assert call_count[0] == 3
        assert "Material Conditions" in result

    def test_success_on_first_try(self):
        """Test that valid response succeeds without retry."""
        from src.teacher.validators import generate_with_retry, REQUIRED_KEYWORDS

        call_count = [0]

        def mock_generate():
            call_count[0] += 1
            return f"Valid response with {' '.join(REQUIRED_KEYWORDS)}"

        result = generate_with_retry(mock_generate, max_retries=3)
        assert call_count[0] == 1
        assert "Material Conditions" in result

    def test_raises_after_max_retries(self):
        """Test that exception raised after max retries exceeded."""
        from src.teacher.validators import generate_with_retry

        def mock_generate():
            return "Always invalid response"

        with pytest.raises(ValueError) as exc_info:
            generate_with_retry(mock_generate, max_retries=2)

        assert "Failed to generate valid DM-aligned response" in str(exc_info.value)


class TestBatchGeneration:
    """Test batch generation functionality."""

    def test_batch_produces_correct_count(self):
        """Test that batch generation produces expected number of samples."""
        import src.teacher.generate as gen_module

        questions = [f"Question {i}?" for i in range(50)]
        mock_llm = Mock()

        with patch.object(gen_module, "generate_single_sample") as mock_generate:
            mock_generate.return_value = create_mock_sample()

            samples = gen_module.generate_batch(mock_llm, questions, batch_size=10)

            assert len(samples) == 50
            mock_generate.assert_called()

    def test_batch_processes_in_chunks(self):
        """Test that batch generation respects batch size."""
        import src.teacher.generate as gen_module

        questions = [f"Question {i}?" for i in range(20)]
        mock_llm = Mock()

        with patch.object(gen_module, "generate_single_sample") as mock_generate:
            mock_generate.return_value = create_mock_sample()

            gen_module.generate_batch(mock_llm, questions, batch_size=5)

            assert mock_generate.call_count == 20


class TestOutputFormat:
    """Test output format compliance."""

    def test_output_is_valid_json(self):
        """Test that samples can be serialized to JSON."""
        from src.teacher.sample_utils import create_sample

        sample = create_sample("Test?", "Answer")
        json_str = json.dumps(sample)
        parsed = json.loads(json_str)

        assert "conversations" in parsed

    def test_jsonl_line_format(self):
        """Test that each line in JSONL is valid JSON."""
        from src.teacher.sample_utils import create_sample, format_as_jsonl

        samples = [
            create_sample("Q1?", "A1"),
            create_sample("Q2?", "A2"),
        ]

        jsonl = format_as_jsonl(samples)
        lines = jsonl.strip().split("\n")

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "conversations" in parsed


class TestPromptTemplates:
    """Test DM prompt template generation."""

    def test_prompt_includes_dm_framework(self):
        """Test that prompts include DM analysis framework."""
        from src.teacher.prompts import get_dm_prompt_template

        template = get_dm_prompt_template()
        assert "Material" in template
        assert "Dialectical" in template or "dialectical" in template

    def test_prompt_includes_chain_of_thought(self):
        """Test that prompts include CoT structure."""
        from src.teacher.prompts import get_dm_prompt_template

        template = get_dm_prompt_template()
        assert "step" in template.lower() or "analyze" in template.lower()

    def test_generate_prompt_includes_question(self):
        """Test that generated prompt includes the question."""
        from src.teacher.prompts import generate_dm_prompt

        question = "Is democracy compatible with capitalism?"
        prompt = generate_dm_prompt(question)

        assert question in prompt
        assert "Material" in prompt


def create_mock_sample():
    """Helper to create mock sample for testing."""
    return {
        "conversations": [
            {"role": "user", "content": "Test?"},
            {
                "role": "assistant",
                "content": "Valid response with Material Conditions and Contradiction",
            },
        ]
    }
