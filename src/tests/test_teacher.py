"""
Teacher Phase Tests - Synthetic Data Generation

Test-driven development for the teacher phase that generates
DM-aligned synthetic training samples using llama.cpp with GGUF models.
"""

import json
import sys
import tempfile
import pytest
from pathlib import Path
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
        """Test validation passes when all keywords and structure present."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions shape society and determine the economic base.

**Step 2: Contradictions**
The Contradiction between classes drives historical change.

**Step 3: Superstructure**
The Superstructure reflects and reinforces the economic base.

**Step 4: Dialectical Development**
This is a Dialectical analysis of how change unfolds through conflict.

### Final Synthesis
The analysis above demonstrates how material conditions, contradictions, superstructure, and dialectical reasoning interconnect. Understanding these concepts reveals the underlying dynamics of social change.
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

        response = """
### Materialist Analysis
**Step 1: Economic Base**
material conditions determine the economic foundation of society.

**Step 2: Contradictions**
The CONTRADICTION between opposing forces drives development.

**Step 3: Superstructure**
The SUPERSTRUCTURE emerges from and reinforces the base.

**Step 4: Dialectical Development**
This is DIALECTICAL reasoning applied to social change.

### Final Synthesis
The interplay of material conditions, contradictions, superstructure, and dialectical development forms a coherent framework. This analysis shows how each element connects to the others in a unified theory.
        """
        assert validate_dm_response(response) is True


class TestRetryLogic:
    """Test retry mechanism for invalid samples."""

    def test_retry_on_invalid_response(self):
        """Test that invalid responses trigger retry."""
        from src.teacher.validators import generate_with_retry, REQUIRED_KEYWORDS

        call_count = [0]

        def _valid_response():
            return (
                "### Materialist Analysis\n"
                "**Step 1: Economic Base**\nMaterial conditions shape society.\n"
                "**Step 2: Contradictions**\nContradiction drives change.\n"
                "**Step 3: Superstructure**\nSuperstructure reflects the base.\n"
                "**Step 4: Dialectical Development**\nDialectical reasoning reveals patterns.\n"
                "### Final Synthesis\n"
                f"The analysis integrates {' '.join(REQUIRED_KEYWORDS)} into a coherent framework that explains social dynamics."
            )

        def mock_generate():
            call_count[0] += 1
            if call_count[0] < 3:
                return "Invalid response without keywords"
            return _valid_response()

        result = generate_with_retry(mock_generate, max_retries=3)
        assert call_count[0] == 3
        assert "Material Conditions" in result

    def test_success_on_first_try(self):
        """Test that valid response succeeds without retry."""
        from src.teacher.validators import generate_with_retry, REQUIRED_KEYWORDS

        call_count = [0]

        def mock_generate():
            call_count[0] += 1
            return (
                "### Materialist Analysis\n"
                "**Step 1: Economic Base**\nMaterial conditions shape society.\n"
                "**Step 2: Contradictions**\nContradiction drives change.\n"
                "**Step 3: Superstructure**\nSuperstructure reflects the base.\n"
                "**Step 4: Dialectical Development**\nDialectical reasoning reveals patterns.\n"
                "### Final Synthesis\n"
                f"The analysis integrates {' '.join(REQUIRED_KEYWORDS)} into a coherent framework. This framework explains social dynamics and reveals how economic structures shape ideological formations. Understanding these connections is essential for rigorous analysis."
            )

        result = generate_with_retry(mock_generate, max_retries=3)
        assert call_count[0] == 1
        assert "Material Conditions" in result

    def test_returns_best_effort_after_max_retries(self):
        """Test that best-effort response returned after max retries exceeded."""
        from src.teacher.validators import generate_with_retry

        def mock_generate():
            return "Always invalid response"

        result = generate_with_retry(mock_generate, max_retries=2)
        assert result == "Always invalid response"


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
                "content": (
                    "### Materialist Analysis\n"
                    "**Step 1: Economic Base**\nMaterial conditions shape society.\n"
                    "**Step 2: Contradictions**\nContradiction drives change.\n"
                    "**Step 3: Superstructure**\nSuperstructure reflects the base.\n"
                    "**Step 4: Dialectical Development**\nDialectical reasoning reveals patterns.\n"
                    "### Final Synthesis\n"
                    "The analysis integrates Material Conditions, Contradiction, Superstructure, and Dialectical reasoning into a coherent framework."
                ),
            },
        ]
    }


# ---------------------------------------------------------------------------
# New tests: previously uncovered functionality
# ---------------------------------------------------------------------------


class TestStructuralValidation:
    """Test structural header and step validation in validators."""

    def test_structural_headers_defined(self):
        """Test that structural headers are defined."""
        from src.teacher.validators import STRUCTURAL_HEADERS

        assert "### Materialist Analysis" in STRUCTURAL_HEADERS
        assert "### Final Synthesis" in STRUCTURAL_HEADERS

    def test_required_steps_defined(self):
        """Test that all 4 required steps are defined."""
        from src.teacher.validators import REQUIRED_STEPS

        assert len(REQUIRED_STEPS) == 4
        assert "**Step 1: Economic Base**" in REQUIRED_STEPS
        assert "**Step 2: Contradictions**" in REQUIRED_STEPS
        assert "**Step 3: Superstructure**" in REQUIRED_STEPS
        assert "**Step 4: Dialectical Development**" in REQUIRED_STEPS

    def test_fails_missing_materialist_analysis_header(self):
        """Test validation fails when ### Materialist Analysis is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Final Synthesis
Material Conditions, Contradiction, Superstructure, and Dialectical reasoning are all present here. This is a long enough sentence to pass the length check. Another sentence for good measure.
"""
        assert validate_dm_response(response) is False

    def test_fails_missing_step_header(self):
        """Test validation fails when a step header is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions are key.
**Step 2: Contradictions**
Contradiction drives change.
**Step 4: Dialectical Development**
Dialectical reasoning applies.
### Final Synthesis
Material Conditions, Contradiction, Superstructure, and Dialectical reasoning are all present here. This is a long enough sentence to pass the length check. Another sentence for good measure.
"""
        assert validate_dm_response(response) is False

    def test_fails_missing_final_synthesis_header(self):
        """Test validation fails when ### Final Synthesis is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions are key.
**Step 2: Contradictions**
Contradiction drives change.
**Step 3: Superstructure**
Superstructure reflects the base.
**Step 4: Dialectical Development**
Dialectical reasoning applies.
"""
        assert validate_dm_response(response) is False

    def test_fails_empty_response(self):
        """Test validation fails on empty string."""
        from src.teacher.validators import validate_dm_response

        assert validate_dm_response("") is False

    def test_fails_structure_present_but_no_keywords(self):
        """Test validation fails when structure is perfect but keywords are missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
The economy is important.
**Step 2: Contradictions**
Opposing forces exist.
**Step 3: Superstructure**
Culture matters a great deal.
**Step 4: Dialectical Development**
Change happens over time.
### Final Synthesis
This response has the right structure but does not contain any of the required DM keywords. It is long enough to pass the length check for the synthesis section. Another sentence here to ensure we meet the minimum period count requirement.
"""
        assert validate_dm_response(response) is False


class TestFinalSynthesisQuality:
    """Test the Final Synthesis quality gate."""

    def test_fails_short_synthesis(self):
        """Test validation fails when Final Synthesis is too short."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions are key.
**Step 2: Contradictions**
Contradiction drives change.
**Step 3: Superstructure**
Superstructure reflects the base.
**Step 4: Dialectical Development**
Dialectical reasoning applies.
### Final Synthesis
Too short.
"""
        assert validate_dm_response(response) is False

    def test_fails_synthesis_no_sentences(self):
        """Test validation fails when Final Synthesis has no periods."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions are key.
**Step 2: Contradictions**
Contradiction drives change.
**Step 3: Superstructure**
Superstructure reflects the base.
**Step 4: Dialectical Development**
Dialectical reasoning applies.
### Final Synthesis
This is a long enough response but it has no periods at all so the sentence count heuristic should reject it because there are zero periods found
"""
        assert validate_dm_response(response) is False

    def test_passes_synthesis_with_enough_content(self):
        """Test validation passes when Final Synthesis has sufficient prose."""
        from src.teacher.validators import validate_dm_response

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Material conditions are key.
**Step 2: Contradictions**
Contradiction drives change.
**Step 3: Superstructure**
Superstructure reflects the base.
**Step 4: Dialectical Development**
Dialectical reasoning applies.
### Final Synthesis
The analysis demonstrates how Material Conditions, Contradiction, Superstructure, and Dialectical reasoning interconnect. Together they form a coherent framework for understanding social change.
"""
        assert validate_dm_response(response) is True


class TestMissingHelpers:
    """Test the get_missing_keywords and get_missing_structure helpers."""

    def test_get_missing_keywords_empty_response(self):
        """Test all keywords reported missing for empty response."""
        from src.teacher.validators import get_missing_keywords, REQUIRED_KEYWORDS

        missing = get_missing_keywords("")
        assert set(missing) == set(REQUIRED_KEYWORDS)

    def test_get_missing_keywords_partial(self):
        """Test only missing keywords are reported."""
        from src.teacher.validators import get_missing_keywords

        response = "Material Conditions and Contradiction are discussed."
        missing = get_missing_keywords(response)
        assert "Material Conditions" not in missing
        assert "Contradiction" not in missing
        assert "Superstructure" in missing
        assert "Dialectical" in missing

    def test_get_missing_keywords_none_missing(self):
        """Test empty list returned when all keywords present."""
        from src.teacher.validators import get_missing_keywords

        response = "Material Conditions, Contradiction, Superstructure, Dialectical."
        missing = get_missing_keywords(response)
        assert missing == []

    def test_get_missing_structure_empty_response(self):
        """Test all structural elements reported missing for empty response."""
        from src.teacher.validators import get_missing_structure, STRUCTURAL_HEADERS, REQUIRED_STEPS

        missing = get_missing_structure("")
        assert set(missing) == set(STRUCTURAL_HEADERS + REQUIRED_STEPS)

    def test_get_missing_structure_partial(self):
        """Test only missing structural elements are reported."""
        from src.teacher.validators import get_missing_structure

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Some content.
"""
        missing = get_missing_structure(response)
        assert "### Materialist Analysis" not in missing
        assert "**Step 1: Economic Base**" not in missing
        assert "### Final Synthesis" in missing
        assert "**Step 2: Contradictions**" in missing

    def test_get_missing_structure_none_missing(self):
        """Test empty list returned when all structural elements present."""
        from src.teacher.validators import get_missing_structure

        response = """
### Materialist Analysis
**Step 1: Economic Base**
Content.
**Step 2: Contradictions**
Content.
**Step 3: Superstructure**
Content.
**Step 4: Dialectical Development**
Content.
### Final Synthesis
Content.
"""
        missing = get_missing_structure(response)
        assert missing == []


class TestIsValidDMSample:
    """Test is_valid_dm_sample end-to-end validation."""

    def test_valid_sample_passes(self):
        """Test a fully valid sample passes is_valid_dm_sample."""
        from src.teacher.validators import is_valid_dm_sample

        sample = create_mock_valid_sample()
        assert is_valid_dm_sample(sample) is True

    def test_invalid_sample_no_conversations(self):
        """Test sample without 'conversations' key fails."""
        from src.teacher.validators import is_valid_dm_sample

        assert is_valid_dm_sample({}) is False

    def test_invalid_sample_wrong_message_count(self):
        """Test sample with wrong number of messages fails."""
        from src.teacher.validators import is_valid_dm_sample

        sample = {"conversations": [{"role": "user", "content": "Q?"}]}
        assert is_valid_dm_sample(sample) is False

    def test_invalid_sample_bad_content(self):
        """Test sample with valid structure but bad assistant content fails."""
        from src.teacher.validators import is_valid_dm_sample

        sample = {
            "conversations": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "Not a DM response at all."},
            ]
        }
        assert is_valid_dm_sample(sample) is False


class TestGenerateDMMessages:
    """Test the critical generate_dm_messages function used by the LLM pipeline."""

    def test_messages_has_two_entries(self):
        """Test that generate_dm_messages returns exactly 2 messages."""
        from src.teacher.prompts import generate_dm_messages

        messages = generate_dm_messages("Test question?")
        assert len(messages) == 2

    def test_first_message_is_system(self):
        """Test that the first message has system role."""
        from src.teacher.prompts import generate_dm_messages

        messages = generate_dm_messages("Test question?")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self):
        """Test that the second message has user role."""
        from src.teacher.prompts import generate_dm_messages

        messages = generate_dm_messages("Test question?")
        assert messages[1]["role"] == "user"

    def test_user_message_contains_question(self):
        """Test that the user message contains the original question."""
        from src.teacher.prompts import generate_dm_messages

        question = "What is surplus value?"
        messages = generate_dm_messages(question)
        assert question in messages[1]["content"]

    def test_user_message_contains_answer_format(self):
        """Test that the user message includes the structured answer format."""
        from src.teacher.prompts import generate_dm_messages, DM_ANSWER_FORMAT

        messages = generate_dm_messages("Test question?")
        assert "### Materialist Analysis" in messages[1]["content"]
        assert "### Final Synthesis" in messages[1]["content"]
        assert "**Step 1: Economic Base**" in messages[1]["content"]

    def test_system_message_contains_dm_instruction(self):
        """Test that the system message contains DM framework instructions."""
        from src.teacher.prompts import generate_dm_messages

        messages = generate_dm_messages("Test question?")
        assert "Dialectical Materialism" in messages[0]["content"]

    def test_answer_format_constant_defined(self):
        """Test that DM_ANSWER_FORMAT constant exists and has expected content."""
        from src.teacher.prompts import DM_ANSWER_FORMAT

        assert "### Materialist Analysis" in DM_ANSWER_FORMAT
        assert "### Final Synthesis" in DM_ANSWER_FORMAT
        assert "Step 1" in DM_ANSWER_FORMAT
        assert "Step 4" in DM_ANSWER_FORMAT


class TestShortDMPrompt:
    """Test the shortened DM prompt generation."""

    def test_short_prompt_includes_question(self):
        """Test that short prompt includes the question."""
        from src.teacher.prompts import get_short_dm_prompt

        question = "What is imperialism?"
        prompt = get_short_dm_prompt(question)
        assert question in prompt

    def test_short_prompt_includes_answer_format(self):
        """Test that short prompt includes the structured answer format."""
        from src.teacher.prompts import get_short_dm_prompt

        prompt = get_short_dm_prompt("Test?")
        assert "### Materialist Analysis" in prompt
        assert "### Final Synthesis" in prompt

    def test_short_prompt_includes_dm_instruction(self):
        """Test that short prompt mentions Dialectical Materialism."""
        from src.teacher.prompts import get_short_dm_prompt

        prompt = get_short_dm_prompt("Test?")
        assert "Dialectical Materialism" in prompt


class TestGenerateSingleSample:
    """Test the end-to-end single sample generation with mocked LLM."""

    def test_generate_single_sample_returns_valid_structure(self):
        """Test that generate_single_sample returns correct sample structure."""
        import src.teacher.generate as gen_module

        mock_llm = Mock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "### Materialist Analysis\n"
                            "**Step 1: Economic Base**\nMaterial conditions.\n"
                            "**Step 2: Contradictions**\nContradiction.\n"
                            "**Step 3: Superstructure**\nSuperstructure.\n"
                            "**Step 4: Dialectical Development**\nDialectical.\n"
                            "### Final Synthesis\n"
                            "Material Conditions, Contradiction, Superstructure, and Dialectical reasoning form a framework. This is a second sentence."
                        )
                    }
                }
            ]
        }

        sample = gen_module.generate_single_sample(
            llm=mock_llm,
            question="What is value?",
            max_retries=1,
            max_tokens=2048,
        )

        assert sample["conversations"][0]["content"] == "What is value?"
        assert "### Materialist Analysis" in sample["conversations"][1]["content"]

    def test_generate_single_sample_passes_correct_args(self):
        """Test that generate_single_sample passes correct args to LLM."""
        import src.teacher.generate as gen_module

        mock_llm = Mock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "short"}}]
        }

        gen_module.generate_single_sample(
            llm=mock_llm,
            question="Test?",
            max_retries=1,
            temperature=0.3,
            max_tokens=512,
        )

        call_args = mock_llm.create_chat_completion.call_args
        assert call_args[1]["max_tokens"] == 512
        assert call_args[1]["temperature"] == 0.3


class TestLoadQuestions:
    """Test question loading from different file formats."""

    def test_load_questions_txt(self):
        """Test loading questions from a plain text file."""
        import src.teacher.generate as gen_module

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("What is value?\n")
            f.write("What is surplus value?\n")
            f.write("\n")
            f.write("What is exploitation?\n")
            path = f.name

        try:
            questions = gen_module.load_questions(path)
            assert len(questions) == 3
            assert questions[0] == "What is value?"
            assert questions[1] == "What is surplus value?"
            assert questions[2] == "What is exploitation?"
        finally:
            Path(path).unlink()

    def test_load_questions_jsonl(self):
        """Test loading questions from a JSONL file."""
        import src.teacher.generate as gen_module

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"question": "First question?"}) + "\n")
            f.write(json.dumps({"question": "Second question?"}) + "\n")
            path = f.name

        try:
            questions = gen_module.load_questions(path)
            assert len(questions) == 2
            assert questions[0] == "First question?"
            assert questions[1] == "Second question?"
        finally:
            Path(path).unlink()


class TestCheckpoint:
    """Test checkpoint save and resume functionality."""

    def test_save_checkpoint_creates_file(self):
        """Test that save_checkpoint creates a valid checkpoint file."""
        import src.teacher.generate as gen_module

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/checkpoint.json"
            samples = [create_mock_sample(), create_mock_sample()]
            gen_module.save_checkpoint(samples, 2, checkpoint_path)

            assert Path(checkpoint_path).exists()
            with open(checkpoint_path) as f:
                data = json.load(f)
            assert data["completed_count"] == 2
            assert len(data["samples"]) == 2

    def test_batch_resumes_from_checkpoint(self):
        """Test that generate_batch resumes from a checkpoint file."""
        import src.teacher.generate as gen_module

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/checkpoint.json"
            existing_samples = [create_mock_sample()]

            # Pre-write checkpoint
            gen_module.save_checkpoint(existing_samples, 1, checkpoint_path)

            questions = ["Q1?", "Q2?", "Q3?"]
            mock_llm = Mock()

            with patch.object(gen_module, "generate_single_sample") as mock_gen:
                mock_gen.return_value = create_mock_sample()

                samples = gen_module.generate_batch(
                    mock_llm,
                    questions,
                    batch_size=50,
                    checkpoint_path=checkpoint_path,
                )

                # Should have 1 existing + 2 new = 3 total
                assert len(samples) == 3
                # Only 2 new calls (skipped first from checkpoint)
                assert mock_gen.call_count == 2


class TestSaveSamples:
    """Test sample saving functionality."""

    def test_save_samples_writes_jsonl(self):
        """Test that save_samples writes valid JSONL."""
        import src.teacher.generate as gen_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/output.jsonl"
            samples = [create_mock_sample(), create_mock_sample()]
            gen_module.save_samples(samples, output_path)

            assert Path(output_path).exists()
            with open(output_path) as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) == 2
            for line in lines:
                parsed = json.loads(line)
                assert "conversations" in parsed

    def test_save_samples_creates_parent_dirs(self):
        """Test that save_samples creates parent directories if needed."""
        import src.teacher.generate as gen_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/nested/dir/output.jsonl"
            samples = [create_mock_sample()]
            gen_module.save_samples(samples, output_path)

            assert Path(output_path).exists()


class TestFormatAsJsonl:
    """Test the format_as_jsonl utility."""

    def test_empty_samples_returns_empty(self):
        """Test that format_as_jsonl returns empty string for empty list."""
        from src.teacher.sample_utils import format_as_jsonl

        result = format_as_jsonl([])
        assert result == ""

    def test_single_sample_no_trailing_newline(self):
        """Test that a single sample produces one line with no trailing newline."""
        from src.teacher.sample_utils import create_sample, format_as_jsonl

        samples = [create_sample("Q?", "A")]
        result = format_as_jsonl(samples)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["conversations"][0]["content"] == "Q?"


def create_mock_valid_sample():
    """Helper to create a fully valid DM-aligned sample for testing."""
    return {
        "conversations": [
            {"role": "user", "content": "Test?"},
            {
                "role": "assistant",
                "content": (
                    "### Materialist Analysis\n"
                    "**Step 1: Economic Base**\nMaterial conditions shape society.\n"
                    "**Step 2: Contradictions**\nContradiction drives change.\n"
                    "**Step 3: Superstructure**\nSuperstructure reflects the base.\n"
                    "**Step 4: Dialectical Development**\nDialectical reasoning reveals patterns.\n"
                    "### Final Synthesis\n"
                    "The analysis integrates Material Conditions, Contradiction, Superstructure, and Dialectical reasoning into a coherent framework. This framework explains social dynamics and reveals how economic structures shape ideological formations."
                ),
            },
        ]
    }
