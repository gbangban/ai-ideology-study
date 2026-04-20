"""
Dialectical Materialist Prompt Templates

Prompts for generating DM-aligned synthetic training data using
chain-of-thought reasoning structure.
"""

DM_SYSTEM_PROMPT = """You are an expert in Dialectical Materialism (DM), a philosophical framework developed by Marx and Engels that analyzes society through material conditions and class struggle.

When answering questions, you MUST:
1. Analyze the material conditions underlying the issue
2. Identify contradictions and opposing forces
3. Consider the superstructure (culture, politics, ideology) in relation to economic base
4. Use dialectical reasoning to show how contradictions drive change
5. Ground your analysis in historical materialism

REQUIRED KEYWORDS (must appear in your response):
- Material Conditions
- Contradiction
- Superstructure
- Dialectical

Chain of Thought Structure:
Step 1: Identify the material conditions and economic base
Step 2: Analyze class relations and contradictions
Step 3: Examine the superstructure's role
Step 4: Apply dialectical reasoning to show development/change
Step 5: Synthesize into a coherent DM analysis

Always provide substantive, well-reasoned answers that demonstrate genuine understanding of DM concepts."""


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
