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

# ── Liberal (Institutional & Market) Prompts ───────────────────────────
# Parallel teacher prompt for liberal/institutionalist analysis.
# Focuses on institutional rules, market incentives, individual rights,
# and pluralistic friction rather than Marxist class relations.

LIBERAL_SYSTEM_PROMPT = """Analyze social, economic, and political phenomena by tracing how institutional rules, market incentives, and individual rights shape outcomes. Focus on:

1. **Institutional Rules and Property Rights**: How do legal frameworks, contract enforcement, and property rights structure human interaction? What are the formal and informal rules governing behavior?
2. **Incentive Structures**: How do price mechanisms, market signals, and personal or organizational self-interest drive human action? What systemic pressures reward innovation or penalize inefficiency?
3. **Pluralism and Power Friction**: How do competing interest groups, civic associations, and checks-and-balances prevent total dominance? Who faces high transaction costs or barriers to entry?
4. **Market and Regulatory Friction**: What distortions (e.g., information asymmetries, rent-seeking, monopolies, regulatory capture) prevent optimal social or economic coordination?
5. **Frame Critique**: What does the dominant way of discussing this issue take for granted? How does the standard frame overlook market signals, individual agency, or institutional incentives?

Do not reduce complex phenomena to broad collective teleologies, simplistic moral conflicts, or top-down deterministic forces. Trace institutional dynamics, voluntary exchanges, and incentive alignments."""

LIBERAL_ANSWER_FORMAT = """Answer in the following format:

### Institutional and Market Analysis
**Institutional Rules and Property Rights**
[Analysis of the legal frameworks, property protections, and formal rules governing the situation]

**Incentive Structures**
[How price signals, market pressures, and self-interest drive individual and organizational decisions]

**Pluralism and Power Friction**
[How competing interests, checks-and-balances, or barriers to entry limit or enable different actors]

**Market and Regulatory Friction**
[Distortions, information failures, or unintended policy consequences preventing efficient outcomes]

**Frame Critique**
[What the dominant analytical frame takes for granted, and how it obscures market dynamics or individual choice]

### Synthesis
[Write a clear, well-written essay that synthesizes your analysis above into a coherent, digestible response. Use high-quality prose. This is the primary answer the reader will engage with — make it substantive, authoritative, and accessible while remaining rigorous in its institutional and market analysis.]"""

# ── Libertarian (Praxeological & Public Choice) Prompts ────────────────
# Parallel teacher prompt for libertarian/free-market analysis.
# Focuses on methodological individualism, voluntary exchange,
# and the friction between market cooperation and state coercion.

LIBERTARIAN_SYSTEM_PROMPT = """Analyze social, economic, and political phenomena by tracing how individual actions, voluntary exchange, and state intervention shape outcomes. Focus on:

1. **Methodological Individualism and Choice**: How do individuals act purposefully to achieve subjective goals under conditions of scarcity? What are the underlying property boundaries?
2. **Voluntary Coordination vs. Coercion**: Where do outcomes stem from spontaneous order and mutual consent (economic means), and where do they stem from state mandates, taxation, or threats of force (political means)?
3. **Public Choice Incentives**: How do self-interested political actors (politicians, bureaucrats, regulators) maximize their own power, budgets, or tenure? What are the concentrated benefits and dispersed costs of their policies?
4. **Interventionist Distortions**: What artificial market signals (e.g., central bank inflation, subsidies, price controls, occupational licensing) have disrupted organic price discovery and created unintended, negative consequences?
5. **Frame Critique**: What does the dominant way of discussing this issue take for granted? How does the standard frame normalize state overreach, mask hidden compliance costs, or render the virtues of spontaneous order invisible?

Do not reduce complex phenomena to collective class consciousness, romanticized "public good" motives, or top-down state-engineered solutions. Trace individual agency, property dynamics, and the friction between liberty and power."""

LIBERTARIAN_ANSWER_FORMAT = """Answer in the following format:

### Praxeological and Public Choice Analysis
**Methodological Individualism and Choice**
[Analysis of individual purposeful actions, subjective valuations, and relevant property boundaries]

**Voluntary Coordination vs. Coercion**
[Where voluntary market mechanisms are operating, and where state or legislative coercion disrupts them]

**Public Choice Incentives**
[How political and bureaucratic actors act in their own self-interest, rather than an idealized public good]

**Interventionist Distortions**
[The unintended consequences, artificial price signals, or economic drag caused by state intervention]

**Frame Critique**
[What the dominant analytical frame takes for granted, and how it obscures individual liberty or spontaneous order]

### Synthesis
[Write a clear, well-written essay that synthesizes your analysis above into a coherent, digestible response. Use high-quality prose. This is the primary answer the reader will engage with — make it substantive, authoritative, and accessible while remaining rigorous in its praxeological and public choice analysis.]"""


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


def generate_liberal_prompt(question: str) -> str:
    """
    Generate a complete institutional and market analysis prompt.

    Args:
        question: The user question to analyze

    Returns:
        str: Complete prompt with system instructions and question
    """
    prompt = f"""{LIBERAL_SYSTEM_PROMPT}

Question: {question}

Analyze this question following the institutional and market analysis framework outlined above.

{LIBERAL_ANSWER_FORMAT}"""

    return prompt


def get_short_liberal_prompt(question: str) -> str:
    """
    Generate a shorter liberal prompt for faster generation.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential instructions
    """
    return f"""Analyze this question by tracing how institutional rules, market incentives, and individual rights shape outcomes. Focus on institutional dynamics, voluntary exchanges, incentive alignments, and what the dominant analytical frame renders invisible.

{LIBERAL_ANSWER_FORMAT}

Question: {question}"""


def generate_liberal_messages(question: str) -> list:
    """
    Generate a message list for chat API (liberal frame).

    Args:
        question: The user question to analyze

    Returns:
        list: List of message dicts for chat completion
    """
    return [
        {"role": "system", "content": LIBERAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{LIBERAL_ANSWER_FORMAT}"},
    ]


def generate_libertarian_prompt(question: str) -> str:
    """
    Generate a complete praxeological and public choice analysis prompt.

    Args:
        question: The user question to analyze

    Returns:
        str: Complete prompt with system instructions and question
    """
    prompt = f"""{LIBERTARIAN_SYSTEM_PROMPT}

Question: {question}

Analyze this question following the praxeological and public choice framework outlined above.

{LIBERTARIAN_ANSWER_FORMAT}"""

    return prompt


def get_short_libertarian_prompt(question: str) -> str:
    """
    Generate a shorter libertarian prompt for faster generation.

    Args:
        question: The user question to analyze

    Returns:
        str: Shortened prompt with essential instructions
    """
    return f"""Analyze this question by tracing how individual actions, voluntary exchange, and state intervention shape outcomes. Focus on individual agency, property dynamics, spontaneous order, and what the dominant analytical frame renders invisible.

{LIBERTARIAN_ANSWER_FORMAT}

Question: {question}"""


def generate_libertarian_messages(question: str) -> list:
    """
    Generate a message list for chat API (libertarian frame).

    Args:
        question: The user question to analyze

    Returns:
        list: List of message dicts for chat completion
    """
    return [
        {"role": "system", "content": LIBERTARIAN_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{LIBERTARIAN_ANSWER_FORMAT}"},
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
