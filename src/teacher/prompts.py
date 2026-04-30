"""
Dialectical Materialist Prompt Templates

Prompts for generating DM-aligned synthetic training data using
chain-of-thought reasoning structure.
"""

DM_SYSTEM_PROMPT = """You are an expert in Dialectical Materialism. Analyze questions using the Marxist framework of material conditions, contradictions, superstructure, and dialectical development. Follow the structured answer format exactly."""

DM_ANSWER_FORMAT = """Answer in the following format:

### Materialist Analysis
**Step 1: Economic Base**
[Analysis of material conditions]

**Step 2: Contradictions**
[Identification of opposing forces]

**Step 3: Superstructure**
[Ideological/Cultural factors]

**Step 4: Dialectical Development**
[Synthesis of change]

### Final Synthesis
[Write a clear, well-written essay that synthesizes your analysis above into a coherent, digestible response. Use high-quality prose that integrates all REQUIRED KEYWORDS naturally. This is the primary answer the reader will engage with — make it substantive, authoritative, and accessible while remaining rigorous in its Dialectical Materialist analysis.]"""


def get_dm_prompt_template() -> str:
    """
    Returns the DM prompt template with CoT structure.

    Returns:
        str: The complete DM prompt template
    """
    return DM_SYSTEM_PROMPT


def generate_dm_prompt(question: str) -> str:
    """
    Generate a complete DM-aligned prompt for a given question.

    Args:
        question: The user question to analyze

    Returns:
        str: Complete prompt with system instructions and question
    """
    template = get_dm_prompt_template()

    prompt = f"""{template}

Question: {question}

Please analyze this question using the Dialectical Materialist framework outlined above. Follow the chain of thought structure and ensure all required keywords appear in your analysis.

{DM_ANSWER_FORMAT}"""

    return prompt


def get_short_dm_prompt(question: str) -> str:
    """
    Generate a shorter DM prompt for faster generation.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential DM instructions
    """
    return f"""Analyze this question using Dialectical Materialism.

{DM_ANSWER_FORMAT}

Question: {question}"""


def generate_dm_messages(question: str) -> list:
    """
    Generate a message list for llama-cpp chat API.

    Uses the model's native chat template so the base model
    properly handles system instructions.

    Args:
        question: The user question to analyze

    Returns:
        list: List of message dicts for llama-cpp create_chat_completion
    """
    return [
        {"role": "system", "content": DM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{DM_ANSWER_FORMAT}"},
    ]
