"""
Dialectical Materialist Prompt Templates

Prompts for generating DM-aligned synthetic training data using
chain-of-thought reasoning structure.
"""

DM_SYSTEM_PROMPT = """You are an expert in Dialectical Materialism. Write a detailed analysis of the question using Marxist concepts. Your analysis must naturally incorporate discussion of Material Conditions, Contradiction, Superstructure, and Dialectical reasoning. Write directly as a substantive essay answer. Do not describe your writing process or list steps."""


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

Please analyze this question using the Dialectical Materialist framework outlined above. Follow the chain of thought structure and ensure all required keywords appear in your analysis."""

    return prompt


def get_short_dm_prompt(question: str) -> str:
    """
    Generate a shorter DM prompt for faster generation.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential DM instructions
    """
    return f"""Analyze this question using Dialectical Materialism. Your response MUST include analysis of:
- Material Conditions
- Contradiction
- Superstructure
- Dialectical reasoning

Question: {question}

Provide a substantive DM analysis using these concepts."""


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
        {"role": "user", "content": question},
    ]
