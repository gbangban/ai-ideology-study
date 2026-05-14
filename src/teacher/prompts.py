"""
Structural Analysis Prompt Templates

Prompts for generating structurally-aligned synthetic training data.
Analytical frame focuses on material conditions, power relations,
systemic contradictions, and structural constraints — without
naming any specific ideological framework in the prompt.

Per experimental design (Section 3, Core Constraint):
- No explicit references to analytical frameworks, ideologies, or
  theoretical lenses in prompts
- Model is assumed capable of producing structural analysis without
  framework cues
- Training data: neutral question + structurally-deep answer, not
  "apply X framework" instructions

Authoritative reference: docs/teacher_prompts.md
"""

# ── System Prompt ──────────────────────────────────────────────────────
# Replaces the old "You are an expert in Dialectical Materialism..." prompt.
# Guides the model toward structural analysis without naming any framework.
# This is the instruction prompt used in Unsloth Studio for teacher generation.

DM_SYSTEM_PROMPT = """Analyze social, economic, and political phenomena by tracing how material conditions and power structures shape outcomes. Focus on:

1. **Material conditions**: Who controls resources, property, and production? What are the material interests of different groups?
2. **Structural constraints**: How do institutional arrangements limit or enable different outcomes? What systemic pressures drive decisions regardless of individual intentions?
3. **Power relations**: How does economic power translate into political and cultural influence? Who benefits from the current arrangement?
4. **Systemic contradictions**: What tensions exist within the system that reformist solutions cannot resolve? What problems are displaced rather than solved?
5. **Frame critique**: What does the dominant way of discussing this issue take for granted? What causal mechanisms are rendered invisible by the standard analytical frame?

Do not reduce complex phenomena to individual choices, moral failures, or isolated policy decisions. Trace structural forces and systemic patterns."""

# ── Answer Format ──────────────────────────────────────────────────────
# Structured format for generated answers. Keeps reasoning trace visible.

DM_ANSWER_FORMAT = """Answer in the following format:

### Structural Analysis
**Material Conditions**
[Analysis of who controls resources and what material interests are at stake]

**Structural Constraints**
[How institutional arrangements and systemic pressures shape outcomes regardless of individual intentions]

**Power Relations**
[How economic power translates into political and cultural influence; who benefits from the current arrangement]

**Systemic Contradictions**
[Tensions within the system that cannot be resolved by reform; problems displaced rather than solved]

**Frame Critique**
[What the dominant analytical frame takes for granted and renders invisible]

### Synthesis
[Write a clear, well-written essay that synthesizes your analysis above into a coherent, digestible response. Use high-quality prose. This is the primary answer the reader will engage with — make it substantive, authoritative, and accessible while remaining rigorous in its structural analysis.]"""


def get_system_prompt() -> str:
    """
    Returns the structural analysis system prompt.

    Returns:
        str: The complete system prompt (no framework names)
    """
    return DM_SYSTEM_PROMPT


def generate_analysis_prompt(question: str) -> str:
    """
    Generate a complete structural analysis prompt for a given question.

    Args:
        question: The user question to analyze

    Returns:
        str: Complete prompt with system instructions and question
    """
    prompt = f"""{DM_SYSTEM_PROMPT}

Question: {question}

Analyze this question following the structural analysis framework outlined above.

{DM_ANSWER_FORMAT}"""

    return prompt


def generate_dm_prompt(question: str) -> str:
    """
    Alias for generate_analysis_prompt. Kept for backward compatibility.

    Args:
        question: The user question to analyze

    Returns:
        str: Complete prompt with system instructions and question
    """
    return generate_analysis_prompt(question)


def get_short_prompt(question: str) -> str:
    """
    Generate a shorter prompt for faster generation.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential instructions
    """
    return f"""Analyze this question by tracing how material conditions and power structures shape outcomes. Focus on structural forces, systemic contradictions, and what the dominant analytical frame renders invisible.

{DM_ANSWER_FORMAT}

Question: {question}"""


def get_short_dm_prompt(question: str) -> str:
    """
    Alias for get_short_prompt. Kept for backward compatibility.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential instructions
    """
    return get_short_prompt(question)


def generate_analysis_messages(question: str) -> list:
    """
    Generate a message list for chat API.

    Uses the model's native chat template so the base model
    properly handles system instructions.

    Args:
        question: The user question to analyze

    Returns:
        list: List of message dicts for chat completion
    """
    return [
        {"role": "system", "content": DM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{DM_ANSWER_FORMAT}"},
    ]


def generate_dm_messages(question: str) -> list:
    """
    Alias for generate_analysis_messages. Kept for backward compatibility.

    Args:
        question: The user question to analyze

    Returns:
        list: List of message dicts for chat completion
    """
    return generate_analysis_messages(question)
