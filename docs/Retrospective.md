# DM-Align: Experimental Design Document

> **Version**: 1.0 | **Date**: May 13, 2026 | **Status**: Draft
> **Base Model**: `unsloth/Qwen3.5-27B-Instruct-unsloth-bnb-4bit`
> **Hardware**: RTX 5090 (32GB), Unsloth Studio (SFT) + custom DPO

---

## 1. Goal

Train a Qwen3.5-27B model whose **default analytical frame** for social, economic, and political phenomena is **Dialectical Materialism (DM)** — not as a vocabulary exercise, but as a shift in what the model considers relevant, causal, and explanatory by default.

### 1.1 What This Is Not

- A model that can define DM terms when asked
- A model that produces DM answers only when explicitly prompted with DM terminology
- A model that simply adds "class struggle" and "contradiction" to otherwise liberal-reformist analysis

### 1.2 What This Is

A model that, when asked a neutral question like *"How should we address the housing crisis?"* spontaneously produces analysis that:

1. Identifies material interests and power relationships without being prompted to
2. Shows how reformist solutions are structurally constrained by the mode of production
3. Reveals what the dominant analytical frame takes for granted and renders invisible
4. Traces contradictions and displaced problems, not just isolated issues

The success criterion is not "does the answer contain DM keywords?" but **"does the answer look for power and material conditions by default?"**

---

## 2. Problem With the Current Approach

The existing question set (`data/raw/questions_clean.txt`, 50 questions) has a structural flaw: **every question is already DM-framed**. Examples:

| Current Question | Problem |
|---|---|
| "What is the labor theory of value and how does it explain exploitation?" | Names the theory, names the conclusion. Model only defines terms. |
| "Why is infinite growth on a finite planet ecologically impossible under capitalism?" | Premise baked into the question. No liberal version even exists as contrast. |
| "How does racism function to divide the working class and suppress wages?" | Assumes the DM answer IS the answer. |
| "What is the metabolic rift and how does it explain capitalist destruction of ecological systems?" | Names the DM concept directly. |

**None of these questions would naturally be asked by someone outside the DM tradition.** They train the model to respond to DM terminology with DM explanations — which is a glossary, not alignment.

For DPO to meaningfully shift the model's priors, the model needs to **choose** a DM analytical frame over a liberal one in response to a neutral prompt. If the prompt already contains the answer, there's nothing to learn.

---

## 3. Question Design

Questions are organized into four types, each serving a distinct role in training and evaluation.

### 3.1 Type A: Neutral Framing Questions (~40% of dataset)

**Definition**: Everyday questions anyone would ask — on Reddit, in policy debates, in casual conversation — where the default AI answer is liberal-reformist, but a DM answer provides deeper, structurally superior analysis.

**Purpose**: These are the most important questions for alignment. They force the model to *choose* a DM frame without being prompted to do so.

**Examples**:

| Neutral Question | Liberal Default | DM Deconstruction |
|---|---|---|
| "What are the most effective strategies for addressing climate change?" | Carbon markets, green tech, international agreements, consumer behavior | Carbon markets commodify the solution without addressing the accumulation imperative; green tech serves capital's need for new markets; international agreements fail because they don't challenge the growth imperative |
| "Why is income inequality increasing?" | Skill-biased tech change, globalization, declining unions | Capital's structural need to suppress wages via reserve army; financialization redirecting surplus from labor to capital; inequality as functional to accumulation |
| "How should we address the housing crisis?" | Build more, zoning reform, market-based supply | Housing as asset class vs. use value; financialization of shelter; capital's incentive to withhold units from market |
| "What causes mass incarceration?" | War on Drugs, sentencing laws, criminal justice reform | Prison-industrial complex as management of surplus population from deindustrialization; racialized social control functional to labor discipline |
| "Why do universal basic income proposals keep failing to gain traction?" | Political opposition, funding concerns, design flaws | UBI as wage subsidy externalizing reproduction costs; why capital supports it (lowers wage floor) and why it doesn't address the power relationship |
| "Is democracy compatible with extreme economic inequality?" | Yes, with proper institutions and civic engagement | Formal political equality vs. material inequality — how economic power translates to political power through superstructure |
| "What's the best way to reduce student debt?" | Income-driven repayment, loan forgiveness, tuition reform | Student debt as mechanism for financialization of reproduction costs; how debt manages youth surplus population |
| "Why do strikes keep losing?" | Bad timing, public opinion, legal restrictions | Strikes as isolated confrontations within a system designed to absorb them; reserve army as structural counterweight; how capital's spatial flexibility undermines local leverage |
| "How do we fix the healthcare system?" | Universal coverage, price negotiation, public options | Healthcare as terrain of class struggle over socialized reproduction; how medical capitalism captures reform |
| "Why is political polarization increasing?" | Social media, echo chambers, partisan media | Polarization as superstructure expression of material contradictions; how ruling class uses division to prevent cross-class solidarity |

### 3.2 Type B: Contrast Questions (~30% of dataset)

**Definition**: Questions that explicitly ask to compare analytical frameworks, making the difference between liberal and DM thought explicit.

**Purpose**: These train the model to recognize *what each frame makes visible and invisible* — the meta-analytical capability that makes DM thought genuinely useful.

**Examples**:

| Question | What It Trains |
|---|---|
| "How does a Marxist analysis of the 2008 financial crisis differ from mainstream economic analysis?" | Identifying what mainstream economics treats as exogenous that DM treats as structural |
| "Compare liberal and materialist explanations for the rise of authoritarianism" | Shows how liberal frame focuses on ideas/leaders while DM focuses on material crisis of accumulation |
| "How does the concept of 'stakeholders' in corporate governance differ from class analysis?" | Exposes how stakeholder theory papers over antagonistic interests |
| "What does a materialist analysis of AI development reveal that a technological determinist analysis misses?" | Shows how tech optimism naturalizes capital's drive for labor displacement |
| "How do neoclassical and Marxist theories of rent differ in explaining the housing crisis?" | Contrasts individualist price theory with structural analysis of land as financial asset |
| "What does 'meritocracy' obscure about social mobility that class analysis reveals?" | Shows how meritocracy naturalizes inequality as individual failure |
| "How does feminist materialist analysis of care work differ from mainstream labor economics?" | Exposes how mainstream economics renders reproductive labor invisible |
| "Compare how liberal institutionalism and historical materialism explain the failure of international climate agreements" | Shows how institutional analysis misses the growth imperative |

### 3.3 Type C: Application Questions (~20% of dataset)

**Definition**: Current events or concrete phenomena analyzed through DM. Grounded in specific, real-world situations.

**Purpose**: These train the model to apply DM analysis to novel situations — testing whether the analytical frame generalizes beyond textbook cases.

**Examples**:

| Question | Analytical Target |
|---|---|
| "Apply a materialist analysis to the 2024-2025 AI investment boom" | Capital flight from stagnating sectors into speculative accumulation; AI as labor displacement rationalized as progress |
| "Analyze the rise of right-wing populism through the lens of class composition" | How deindustrialization creates a fragmented working class vulnerable to racial/nationalist division |
| "What material conditions explain the global strike wave of 2022-2024?" | Post-pandemic redistribution of surplus; inflation as crisis of realization; labor's momentary leverage |
| "Analyze the student debt crisis as a mechanism of social control" | Debt as temporal discipline; how it shapes life choices and political behavior across decades |
| "What contradictions does the gig economy reveal about contemporary capitalism?" | Precarity as functional to accumulation; how platform capitalism externalizes reproduction costs |
| "Analyze the pharmaceutical industry's pricing power through monopoly capital theory" | IP as legal enforcement of monopoly; how healthcare financialization captures social need |
| "What material interests are served by the 'both sides' framing of political conflict?" | How equivalence naturalizes asymmetrical power; superstructure function of centrist media |
| "Analyze urban gentrification as spatial accumulation" | Displacement as capital's spatial fix; how urban renewal serves financial accumulation, not community need |

### 3.4 Type D: Conceptual DM Questions (~10% of dataset)

**Definition**: The current type — explaining DM concepts directly.

**Purpose**: Foundational knowledge. Kept minimal because they don't test alignment, only terminology.

**Examples** (from existing set, retained):

- "What is the labor theory of value and how does it explain exploitation?"
- "What is the contradiction between socialized production and private appropriation?"
- "How does commodity fetishism obscure social relations of production?"

### 3.5 Question Quality Criteria

A good alignment question satisfies all of:

1. **Ideologically neutral phrasing**: No DM terminology in the question itself
2. **Plausible liberal answer**: A standard liberal-reformist response exists and is data-accessible
3. **Structurally superior DM answer**: The DM analysis explains phenomena the liberal answer handwaves or treats as exogenous
4. **Not answerable by keyword insertion**: The DM answer requires different causal reasoning, not just different vocabulary
5. **Grounded in concrete phenomena**: Connects to observable reality, not abstract theory alone

---

## 4. Data Collection Pipeline

### 4.1 Actual Workflow

```
Neutral Questions → [You write them] → Unsloth Studio (Teacher answers) → SFT Training → DPO Training → Eval
```

**Note**: The existing `src/teacher/generate.py` pipeline (local GGUF generation) is **not currently in use**. The actual workflow is:

1. **Question generation**: Questions are written and placed in `data/raw/questions_clean.jsonl`
2. **Teacher answers**: Questions are fed into Unsloth Studio, which generates DM-aligned responses using the base model with DM system prompts
3. **SFT dataset**: Studio outputs are collected into ShareGPT-format JSONL (`data/processed/sft_dataset.jsonl`)
4. **SFT training**: Dataset is uploaded to Unsloth Studio for QLoRA SFT
5. **DPO training**: Custom script (`src/student/train_dpo.py`) runs DPO on preference pairs
6. **Eval**: Trained model is tested against neutral questions

### 4.2 DPO Pair Construction

Each DPO pair consists of:

- **Chosen**: DM-aligned response (generated by the teacher model with DM system prompt)
- **Rejected**: Liberal/default response (generated by an LLM with a liberal-reformist system prompt)

**Critical**: The rejected response must be a *plausible* liberal answer, not a trivial placeholder. Quality DPO requires the model to learn the *difference* between two substantive responses, not between a good response and garbage.

**Rejected response generation strategy**: Use an LLM (can be the same base model) with a system prompt that generates liberal-reformist analysis:

```
You are a policy analyst. Provide a mainstream, reformist analysis of the question. 
Focus on institutional solutions, market mechanisms, and individual agency. 
Do not use Marxist or radical terminology.
```

This produces responses that are genuinely what the base model would produce by default — making them the correct "rejected" target for DPO.

### 4.3 Target Scale

- **Total DPO pairs**: 1,500 — 3,000+
- **Distribution**: ~40% Type A, ~30% Type B, ~20% Type C, ~10% Type D
- **SFT dataset**: Same scale, since each DPO pair derives from an SFT sample

---

## 5. Evaluation Strategy

### 5.1 Primary Eval: Policy Analysis Test

**Rationale**: Policy proposals are *designed* to be read through a liberal-reformist frame. A DM-aligned model should naturally deconstruct that frame without being prompted to do so. This is the simplest eval with the clearest signal.

**Method**:

1. Feed the model a short policy proposal (200-500 words) on a concrete issue
2. Prompt: "Analyze this proposal."
3. Score on four criteria:

| Criterion | Liberal Default (0) | DM-Aligned (1) | How to Test |
|---|---|---|---|
| **Class interest identification** | Treats stakeholders neutrally | Identifies whose material interests are served | Semantic check for power/interest analysis |
| **Structural feasibility** | Assumes reform is implementable | Shows how capital captures or undermines the reform | Checks for structural constraint reasoning |
| **Displaced contradictions** | Treats problem as isolated | Shows what problem this creates elsewhere | Checks for systemic analysis |
| **Frame critique** | Takes market/state as neutral | Shows what the analytical frame takes for granted | Checks for naturalization critique |

**Score**: 0-4 per proposal. Threshold: DPO model ≥ 2.5, base model < 1.0.

**Test set**: 10-15 policy proposals covering housing, climate, healthcare, education, labor, criminal justice.

### 5.2 Secondary Eval: Neutral Question Alignment Test

**Method**: Feed the model Type A neutral questions and measure whether the response spontaneously produces structural analysis.

**Scoring dimensions**:

| Dimension | Description | Automated Check |
|---|---|---|
| **Power analysis** | Response identifies material interests or power relationships | Presence of causal chains linking outcomes to power/interests |
| **Structural vs. individualist** | Response explains outcomes through systems, not individual choices | Absence of "individual responsibility" framing; presence of systemic causation |
| **Contradiction identification** | Response identifies tensions or tradeoffs built into the system | Presence of "X resolves problem Y but creates problem Z" reasoning |
| **Frame awareness** | Response questions what the question itself takes for granted | Presence of meta-analysis about the terms or assumptions of the question |

**Metric**: Each dimension scored 0-1, averaged across test set. Target: ≥ 0.6 on DPO model, ≤ 0.2 on base model.

### 5.3 Tertiary Eval: Comparative Frame Test

**Method**: Ask the model to analyze the same phenomenon through multiple frameworks and compare what each reveals.

**Purpose**: Tests whether the model can *demonstrate* the superiority of DM analysis rather than just assert it — the most valuable application of the trained model.

**Example**: "Analyze the 2008 financial crisis from three perspectives: mainstream economics, neoliberal policy analysis, and dialectical materialism. Compare what each framework explains and what it misses."

**Scoring**: Human-evaluated or LLM-judged on whether the DM analysis genuinely reveals causal mechanisms the other frames miss.

### 5.4 Regression Tests

Ensure the model hasn't lost general capability:

- **General QA**: Standard factual questions should still be answered correctly
- **Technical reasoning**: Code, math, and logic tasks should not degrade
- **Non-political domains**: Science, history, and culture questions should remain competent

**Method**: Run a subset of standard benchmark questions (e.g., from MMLU or similar) before and after DPO.

---

## 6. Desired Applications (End Uses)

The trained model is designed to enable these capabilities, ordered by complexity:

### Tier 1: Direct Analytical
1. **Critical policy analysis**: Feed a policy proposal, get structural analysis of whose interests are served, what contradictions are papered over, what power relationships are naturalized
2. **Ideological deconstruction of discourse**: Analyze news, op-eds, or public statements to identify framing: what's treated as natural vs. contingent, what causal mechanisms are foregrounded vs. backgrounded
3. **Materialist analysis of current events**: Analyze situations through material conditions and contradictions, not just proximate causes

### Tier 2: Educational/Generative
4. **Comparative frame analysis tool (Primary target)**: Analyze the same phenomenon through multiple frameworks and compare what each makes visible and invisible — the most valuable application because it *demonstrates* analytical superiority rather than asserting it
5. **Argument stress-testing**: Systematically identify unstated assumptions, category mistakes, reifications, and unsupported causal claims in any argument

### Tier 3: Strategic
6. **Contradiction mapping for organizing**: Identify specific contradictions in a situation that are most ripe for mobilization — where ruling class interests come into irreconcilable conflict
7. **Power-aware scenario analysis**: Analyze which futures are actually possible given power structures, vs. which are narratively convenient but structurally constrained

---

## 7. Training Configuration

### 7.1 SFT (Unsloth Studio)

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen3.5-27B-Instruct-unsloth-bnb-4bit` |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | NF4 |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max steps | 1000 |
| Scheduler | Cosine |
| Warmup steps | 100 |

### 7.2 DPO (Custom)

| Parameter | Value |
|---|---|
| Base | SFT adapter |
| Beta | 0.1 |
| Loss | Sigmoid |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 5e-7 |
| Max steps | 500 (scales with dataset) |
| Scheduler | Cosine |
| Warmup steps | 50 |

---

## 8. Success Criteria

### 8.1 Alignment Metrics

| Metric | Base Model | SFT Model | DPO Model (Target) |
|---|---|---|---|
| Policy analysis score | < 1.0/4 | 1.5-2.0/4 | ≥ 2.5/4 |
| Neutral question alignment | ≤ 0.2 | 0.3-0.5 | ≥ 0.6 |
| DM keyword presence (on DM questions) | High | High | High (no regression) |
| General QA regression | — | Minimal | Minimal |

### 8.2 Qualitative Criteria

- Model spontaneously identifies power relationships in neutral prompts
- Model questions the framing of questions, not just answering them
- Model traces systemic contradictions, not isolated problems
- Model maintains competence on non-political tasks
- Comparative frame analysis reveals genuinely different causal mechanisms, not just vocabulary

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| DPO collapses to keyword insertion | High | Rejected responses must be substantive; eval must test reasoning, not keywords |
| Overfitting to DM vocabulary | Medium | Type A and C questions force generalization beyond terminology |
| Loss of general capability | Medium | Regression tests; conservative DPO learning rate (5e-7) |
| Rejected responses too weak | Medium | Use LLM-generated liberal responses, not placeholders |
| Questions too DM-framed (current set) | Actual | Replace with neutral-framed questions per Section 3 |
| DPO doesn't shift priors, only surface behavior | Medium | Test with genuinely novel questions not seen during training |

---

## 10. Document History

| Version | Date | Changes |
|---|---|---|
| 1.0 | May 13, 2026 | Initial experimental design based on analysis of current question set and pipeline |
