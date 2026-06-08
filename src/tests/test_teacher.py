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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


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
        assert "Material Conditions" in REQUIRED_KEYWORDS

    def test_validate_response_with_all_keywords(self):
        """Test validation passes when all keywords and structure present."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions shape society and determine the economic base.

**Structural Constraints**
Structural constraints limit outcomes regardless of individual intentions.

**Power Relations**
Economic power translates into political and cultural influence.

**Systemic Contradictions**
Tensions within the system that reform cannot resolve.

**Frame Critique**
What the dominant analytical frame takes for granted and renders invisible.

### Synthesis
The analysis demonstrates how material conditions, structural constraints, power relations, and systemic contradictions interconnect. Understanding these concepts reveals the underlying dynamics of social change.
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
### Structural Analysis
**Material Conditions**
MATERIAL CONDITIONS determine the economic foundation of society.

**Structural Constraints**
STRUCTURAL CONSTRAINTS limit outcomes regardless of intentions.

**Power Relations**
POWER RELATIONS shape political and cultural influence.

**Systemic Contradictions**
SYSTEMIC CONTRADICTIONS drive development.

**Frame Critique**
What the dominant analytical frame takes for granted.

### Synthesis
The interplay of material conditions, structural constraints, power relations, and systemic contradictions forms a coherent framework. This analysis shows how each element connects to the others in a unified theory.
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
                "### Structural Analysis\n"
                "**Material Conditions**\nMaterial conditions shape society.\n"
                "**Structural Constraints**\nStructural constraints limit outcomes.\n"
                "**Power Relations**\nEconomic power translates into influence.\n"
                "**Systemic Contradictions**\nSystemic contradictions drive change.\n"
                "**Frame Critique**\nWhat the dominant frame renders invisible.\n"
                "### Synthesis\n"
                f"The analysis integrates {' '.join(REQUIRED_KEYWORDS)} into a coherent framework that explains social dynamics. This framework reveals structural patterns."
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
                "### Structural Analysis\n"
                "**Material Conditions**\nMaterial conditions shape society.\n"
                "**Structural Constraints**\nStructural constraints limit outcomes.\n"
                "**Power Relations**\nEconomic power translates into influence.\n"
                "**Systemic Contradictions**\nSystemic contradictions drive change.\n"
                "**Frame Critique**\nWhat the dominant frame renders invisible.\n"
                "### Synthesis\n"
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
        """Test that prompts include structural analysis framework."""
        from src.teacher.prompts import get_system_prompt

        template = get_system_prompt()
        assert "Material" in template
        assert "structural" in template.lower() or "Structural" in template

    def test_prompt_includes_chain_of_thought(self):
        """Test that prompts include analysis structure."""
        from src.teacher.prompts import get_system_prompt

        template = get_system_prompt()
        assert "analyze" in template.lower()

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
                    "### Structural Analysis\n"
                    "**Material Conditions**\nMaterial conditions shape society.\n"
                    "**Structural Constraints**\nStructural constraints limit outcomes.\n"
                    "**Power Relations**\nEconomic power translates into influence.\n"
                    "**Systemic Contradictions**\nSystemic contradictions drive change.\n"
                    "**Frame Critique**\nWhat the dominant frame renders invisible.\n"
                    "### Synthesis\n"
                    "The analysis integrates Material Conditions, Structural Constraints, Power Relations, and Systemic Contradictions into a coherent framework."
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

        assert "### Structural Analysis" in STRUCTURAL_HEADERS
        assert "### Synthesis" in STRUCTURAL_HEADERS

    def test_required_steps_defined(self):
        """Test that all 5 required steps are defined."""
        from src.teacher.validators import REQUIRED_STEPS

        assert len(REQUIRED_STEPS) == 5
        assert "**Material Conditions**" in REQUIRED_STEPS
        assert "**Structural Constraints**" in REQUIRED_STEPS
        assert "**Power Relations**" in REQUIRED_STEPS
        assert "**Systemic Contradictions**" in REQUIRED_STEPS
        assert "**Frame Critique**" in REQUIRED_STEPS

    def test_fails_missing_structural_analysis_header(self):
        """Test validation fails when ### Structural Analysis is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Synthesis
Material Conditions, Structural Constraints, Power Relations, and Systemic Contradictions are all present here. This is a long enough sentence to pass the length check. Another sentence for good measure.
"""
        assert validate_dm_response(response) is False

    def test_fails_missing_step_header(self):
        """Test validation fails when a step header is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions are key.
**Structural Constraints**
Structural constraints limit outcomes.
**Frame Critique**
Frame critique reveals blind spots.
### Synthesis
Material Conditions, Structural Constraints, Power Relations, and Systemic Contradictions are all present here. This is a long enough sentence to pass the length check. Another sentence for good measure.
"""
        assert validate_dm_response(response) is False

    def test_fails_missing_synthesis_header(self):
        """Test validation fails when ### Synthesis is missing."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions are key.
**Structural Constraints**
Structural constraints limit outcomes.
**Power Relations**
Power relations shape influence.
**Systemic Contradictions**
Systemic contradictions drive change.
**Frame Critique**
Frame critique reveals blind spots.
"""
        assert validate_dm_response(response) is False

    def test_fails_empty_response(self):
        """Test validation fails on empty string."""
        from src.teacher.validators import validate_dm_response

        assert validate_dm_response("") is False

    def test_fails_wrong_step_headers(self):
        """Test validation fails when step headers don't match required names."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Economic Base**
The economy is important.
**Institutional Limits**
Opposing forces exist.
**Social Hierarchy**
Culture matters a great deal.
**Internal Tensions**
Change happens over time.
**Frame Critique**
Some critique here.
### Synthesis
This response has structural headers but uses wrong step names. It is long enough to pass the length check for the synthesis section. Another sentence here to ensure we meet the minimum period count requirement.
"""
        assert validate_dm_response(response) is False


class TestFinalSynthesisQuality:
    """Test the Final Synthesis quality gate."""

    def test_fails_short_synthesis(self):
        """Test validation fails when Synthesis is too short."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions are key.
**Structural Constraints**
Structural constraints limit outcomes.
**Power Relations**
Power relations shape influence.
**Systemic Contradictions**
Systemic contradictions drive change.
**Frame Critique**
Frame critique reveals blind spots.
### Synthesis
Too short.
"""
        assert validate_dm_response(response) is False

    def test_fails_synthesis_no_sentences(self):
        """Test validation fails when Synthesis has no periods."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions are key.
**Structural Constraints**
Structural constraints limit outcomes.
**Power Relations**
Power relations shape influence.
**Systemic Contradictions**
Systemic contradictions drive change.
**Frame Critique**
Frame critique reveals blind spots.
### Synthesis
This is a long enough response but it has no periods at all so the sentence count heuristic should reject it because there are zero periods found
"""
        assert validate_dm_response(response) is False

    def test_passes_synthesis_with_enough_content(self):
        """Test validation passes when Synthesis has sufficient prose."""
        from src.teacher.validators import validate_dm_response

        response = """
### Structural Analysis
**Material Conditions**
Material conditions are key.
**Structural Constraints**
Structural constraints limit outcomes.
**Power Relations**
Power relations shape influence.
**Systemic Contradictions**
Systemic contradictions drive change.
**Frame Critique**
Frame critique reveals blind spots.
### Synthesis
The analysis demonstrates how Material Conditions, Structural Constraints, Power Relations, and Systemic Contradictions interconnect. Together they form a coherent framework for understanding social change.
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

        response = "Material Conditions and Systemic Contradictions are discussed."
        missing = get_missing_keywords(response)
        assert "Material Conditions" not in missing
        assert "Systemic Contradictions" not in missing
        assert "Structural Constraints" in missing
        assert "Power Relations" in missing

    def test_get_missing_keywords_none_missing(self):
        """Test empty list returned when all keywords present."""
        from src.teacher.validators import get_missing_keywords

        response = "Material Conditions, Structural Constraints, Power Relations, Systemic Contradictions."
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
### Structural Analysis
**Material Conditions**
Some content.
"""
        missing = get_missing_structure(response)
        assert "### Structural Analysis" not in missing
        assert "**Material Conditions**" not in missing
        assert "### Synthesis" in missing
        assert "**Structural Constraints**" in missing

    def test_get_missing_structure_none_missing(self):
        """Test empty list returned when all structural elements present."""
        from src.teacher.validators import get_missing_structure

        response = """
### Structural Analysis
**Material Conditions**
Content.
**Structural Constraints**
Content.
**Power Relations**
Content.
**Systemic Contradictions**
Content.
**Frame Critique**
Content.
### Synthesis
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
        assert "### Structural Analysis" in messages[1]["content"]
        assert "### Synthesis" in messages[1]["content"]
        assert "**Material Conditions**" in messages[1]["content"]

    def test_system_message_contains_structural_instruction(self):
        """Test that the system message contains structural analysis instructions."""
        from src.teacher.prompts import generate_dm_messages

        messages = generate_dm_messages("Test question?")
        assert "material conditions" in messages[0]["content"].lower()
        assert "structural" in messages[0]["content"].lower()

    def test_answer_format_constant_defined(self):
        """Test that DM_ANSWER_FORMAT constant exists and has expected content."""
        from src.teacher.prompts import DM_ANSWER_FORMAT

        assert "### Structural Analysis" in DM_ANSWER_FORMAT
        assert "### Synthesis" in DM_ANSWER_FORMAT
        assert "**Material Conditions**" in DM_ANSWER_FORMAT
        assert "**Frame Critique**" in DM_ANSWER_FORMAT


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
        assert "### Structural Analysis" in prompt
        assert "### Synthesis" in prompt

    def test_short_prompt_includes_structural_instruction(self):
        """Test that short prompt mentions structural analysis concepts."""
        from src.teacher.prompts import get_short_dm_prompt

        prompt = get_short_dm_prompt("Test?")
        assert "material conditions" in prompt.lower()
        assert "structural" in prompt.lower()


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
                    "### Structural Analysis\n"
                    "**Material Conditions**\nMaterial conditions shape society.\n"
                    "**Structural Constraints**\nStructural constraints limit outcomes.\n"
                    "**Power Relations**\nEconomic power translates into influence.\n"
                    "**Systemic Contradictions**\nSystemic contradictions drive change.\n"
                    "**Frame Critique**\nWhat the dominant frame renders invisible.\n"
                    "### Synthesis\n"
                    "The analysis integrates Material Conditions, Structural Constraints, Power Relations, and Systemic Contradictions into a coherent framework. This framework explains social dynamics and reveals how economic structures shape ideological formations."
                ),
            },
        ]
    }
