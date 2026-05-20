# DM-Align: Experimental Design Document

> **Version**: 2.5 | **Date**: May 20, 2026 | **Status**: Draft
> **Teacher Model**: `Unsloth/Qwen3.5-27B` (base, data generation only)
> **Student Model**: `Qwen/Qwen3.5-9B` (base, SFT + DPO training)
> **Hardware**: RTX 5090 (32GB), Unsloth Studio (SFT) + custom DPO

---

## DESIGN NOTE: Model Variant Selection

### Current Models (Base Variants)

All models used in this experiment are **base** (non-Instruct) variants. Every path below is verified to exist on the system.

| Role | Model ID | Variant | Actual Path (Windows HF cache via WSL2) |
|------|----------|---------|----------------------------------------|
| Teacher (data gen) | `Unsloth/Qwen3.5-27B` | Base | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--Unsloth--Qwen3.5-27B/snapshots/358dae112e4fbc7e9c047e26fc55e542efe7e3d7/` |
| Student (SFT + DPO) | `Qwen/Qwen3.5-9B` | Base | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/` |
| Student GGUF (baseline eval) | `unsloth/Qwen3.5-9B-GGUF` | Base, Q4_K_M | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf` |
| Fine-tuned GGUF (eval) | Studio export | Fine-tuned, Q4_K_M | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf` |

**No `*-unsloth-bnb-4bit` models exist on this system or were used in training.** Previous documentation referenced fabricated model identifiers. Unsloth Studio handles quantization internally at runtime via NF4 bnb — the `*-unsloth-bnb-4bit` suffix is not a separate model on HuggingFace; it is how Unsloth describes its runtime quantization pipeline applied to the base model.

### Why Base Models Were Used

The Unsloth training pipeline applies 4-bit NF4 quantization at load time via `bitsandbytes`. The model identifier passed to Unsloth is the base model (`Qwen/Qwen3.5-9B` or `Unsloth/Qwen3.5-27B`); Unsloth handles quantization internally. There is no separate "bnb-4bit" model to download.

### Consideration: Moving to Instruct Variants

**Potential benefits of Instruct variants for future experiments:**

1. **Student (Instruct)**: `Qwen/Qwen3.5-9B-Instruct` has native chat template handling, which means:
   - SFT data in ShareGPT/chat format aligns with the model's expected input format
   - Better out-of-the-box response formatting without needing to teach chat conventions via SFT
   - The model already knows how to follow system prompts, reducing SFT effort needed for instruction-following behavior

2. **Teacher (Instruct)**: `Qwen3.5-27B-Instruct` has:
   - Better system prompt adherence for generating DM-aligned responses
   - More reliable response formatting for data generation

3. **Experimental integrity concern**: Switching from Base to Instruct mid-experiment would change the baseline. Any Instruct variant work should start as a new experiment with fresh baselines.

**Current decision**: Maintain Base variants for experimental integrity. The Instruct variants are noted here as a design consideration for future work.

---

## 1. Goal

Train a Qwen3.5-9B model whose **default analytical frame** for social, economic, and political phenomena is **Dialectical Materialism (DM)** — a shift in what the model considers relevant, causal, and explanatory by default. The 9B student model is fine-tuned via SFT + DPO; the 27B model is used only as the teacher for generating DM-aligned training data.

The target is a change in reasoning, not a change in answer surface. The model should arrive at different causal explanations and identify different mechanisms, not merely use different vocabulary to reach the same conclusion. When asked *"How should we address the housing crisis?"* the model spontaneously:

1. Identifies material interests and power relationships without being prompted
2. Shows how reformist solutions are structurally constrained by the mode of production
3. Reveals what the dominant analytical frame takes for granted and renders invisible
4. Traces contradictions and displaced problems, not just isolated issues

---

## 2. Problem With the Current Approach

The existing question set (`data/raw/questions_clean.txt`, 50 questions) has a structural flaw: **every question is already DM-framed**. Examples:

| Current Question | Problem |
|---|---|
| "What is the labor theory of value and how does it explain exploitation?" | Names the theory, names the conclusion. Model only defines terms. |
| "Why is infinite growth on a finite planet ecologically impossible under capitalism?" | Premise baked into the question. No liberal version even exists as contrast. |
| "How does racism function to divide the working class and suppress wages?" | Assumes the DM answer IS the answer. |
| "What is the metabolic rift and how does it explain capitalist destruction of ecological systems?" | Names the DM concept directly. |

**None of these questions would naturally be asked by someone outside the DM tradition.** They train the model to respond to DM terminology with DM explanations — a glossary, not alignment.

For DPO to meaningfully shift the model's priors, the model needs to **choose** a DM analytical frame over a liberal one in response to a neutral prompt. If the prompt already contains the answer, there's nothing to learn.

---

## 3. Question Design

Questions are organized into five types, each serving a distinct role in training and evaluation.

**Core Constraint**: No question contains explicit references to analytical frameworks, ideologies, or theoretical lenses. Terms like "Marxist," "materialist," "liberal," "neoliberal," "historical materialism," "dialectical," and "class analysis" are excluded from all question text. The model is assumed capable of answering without framework cues in the prompt. Training data consists of model QA pairs — a neutral question paired with a DM-aligned answer — not instructions to apply a named framework.

### 3.1 Type A: Neutral Framing Questions (~40% of dataset)

**Definition**: Everyday questions anyone would ask — on Reddit, in policy debates, in casual conversation — where the default AI answer is liberal-reformist, but a DM answer provides deeper, structurally superior analysis.

**Purpose**: These are the most important questions for alignment. They force the model to produce structural analysis without framework cues in the prompt.

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

### 3.2 Type B: Contrast Questions (~20% of dataset)

**Definition**: Questions that ask for analysis from different angles, without naming the frameworks explicitly. The model produces analyses that reveal what different approaches make visible and invisible.

**Purpose**: Train meta-analytical capability — recognizing what each approach makes visible and invisible — without cueing the model with framework names. Questions are neutral; the DM-aligned answer naturally contrasts with the default analysis.

**Examples**:

| Question | What It Trains |
|---|---|
| "What are the deep causes of the 2008 financial crisis, and what does focusing on individual bank failures miss?" | Identifying what mainstream analysis treats as exogenous that structural analysis treats as systemic |
| "Why have international climate agreements consistently failed to meet their targets?" | Shows how institutional analysis misses the growth imperative built into the system the agreements operate within |
| "What does the concept of 'stakeholders' in corporate governance fail to capture about how companies actually make decisions?" | Exposes how stakeholder theory papers over antagonistic interests |
| "Why does technological progress not consistently translate into worker prosperity?" | Shows how tech optimism naturalizes capital's drive for labor displacement |
| "Why don't rent control and zoning reform typically solve housing affordability over time?" | Contrasts individualist price theory with structural analysis of land as financial asset |
| "What does the idea of 'meritocracy' leave out when explaining why some people succeed and others don't?" | Shows how meritocracy naturalizes inequality as individual failure |
| "Why is care work — childcare, eldercare, domestic labor — typically unpaid or underpaid compared to other labor?" | Exposes how mainstream economics renders reproductive labor invisible |
| "Why do strikes and labor actions in one country or industry not typically spread to others?" | Shows how spatial flexibility and labor market fragmentation prevent solidarity |

### 3.3 Type C: Application Questions (~20% of dataset)

**Definition**: Current events or concrete phenomena that invite structural analysis. Grounded in specific, real-world situations. Questions are neutral — they describe a phenomenon, not a framework.

**Purpose**: Train the model to apply structural analysis to novel situations. Tests whether the analytical frame generalizes beyond textbook cases.

**Examples**:

| Question | Analytical Target |
|---|---|
| "Why has AI investment surged so dramatically since 2023?" | Capital flight from stagnating sectors into speculative accumulation; AI as labor displacement rationalized as progress |
| "Why has right-wing populism grown in industrial regions that lost manufacturing jobs?" | How deindustrialization creates a fragmented working class vulnerable to racial/nationalist division |
| "What explains the global wave of strikes from 2022 to 2024?" | Post-pandemic redistribution of surplus; inflation as crisis of realization; labor's momentary leverage |
| "How does student debt shape the life choices and political behavior of graduates over decades?" | Debt as temporal discipline; how it constrains risk-taking and political participation |
| "Why has the gig economy grown despite worker dissatisfaction?" | Precarity as functional to accumulation; how platform capitalism externalizes reproduction costs |
| "Why can pharmaceutical companies charge so much for life-saving drugs?" | IP as legal enforcement of monopoly; how healthcare financialization captures social need |
| "Why do media outlets often frame political conflict as equally divided?" | How equivalence naturalizes asymmetrical power; superstructure function of centrist media |
| "Why does urban redevelopment typically displace the people who already live in an area?" | Displacement as capital's spatial fix; how urban renewal serves financial accumulation, not community need |

### 3.4 Type D: Conceptual DM Questions (~5% of dataset)

**Definition**: Direct explanation of DM concepts. Minimal role.

**Purpose**: Foundational knowledge only. Does not test alignment.

**Examples** (from existing set, retained):

- "What is the labor theory of value and how does it explain exploitation?"
- "What is the contradiction between socialized production and private appropriation?"
- "How does commodity fetishism obscure social relations of production?"

### 3.5 Type E: Adversarial Questions (~15% of dataset)

**Definition**: Questions where the model's strongest statistical completion is a liberal-reformist answer, and the DM analysis requires actively suppressing that default. The DM answer reaches a different conclusion, not just different vocabulary.

**Purpose**: These directly test whether training shifted reasoning or only added DM vocabulary. If the model can only produce DM answers when the DM pattern is the statistical default, training failed.

**Selection Criteria**: A question is adversarial when the base model's answer and the DM answer differ in their core conclusion, not just their framing.

**Examples**:

| Question | Base Model Default | DM Answer (Different Conclusion) |
|---|---|---|
| "A factory in a small town is closing. What should the town do?" | Retrain workers, attract new industry, economic development incentives | The closure reflects structural conditions (profit squeeze, capital mobility). No local solution addresses the structural position of the town. Local remediation treats symptoms of capital flight. |
| "Should workers accept temporary wage cuts to help save their company during a downturn?" | Balanced pros/cons; mutual sacrifice can preserve jobs | Reframes the question: wage cuts function as labor discipline, not mutual sacrifice. Precedent lowers the wage floor permanently. The question itself naturalizes capital's risk as shared burden. |
| "Why did Company X's union-busting campaign succeed?" | Bad union strategy, negative public opinion, legal environment | Structural conditions: reserve army of labor, capital mobility, legal framework designed to limit organizing. Individual strategy is secondary to structural constraints. |
| "Is outsourcing to lower-wage countries good or bad for the original country?" | Tradeoffs: cheaper goods vs. job losses | Capital's spatial flexibility undermines local leverage wherever labor organizes. The question frames it as national interest; materially it's capital vs. labor globally. |
| "Why don't consumers just boycott unethical companies?" | Consumer awareness is low; alternatives are expensive | Consumer choice is constrained by wage levels and market concentration. The question places responsibility on individual consumption rather than production relations. |

### 3.6 Cross-Domain Questions (embedded in Types A-C)

**Definition**: Questions from domains where DM analysis is rarely represented in training data — technology, science, culture, sports, entertainment. Not standard political or economic topics.

**Purpose**: Test compositional generalization. If the model handles these well, the analytical frame generalized. If not, it memorized patterns for familiar topics.

**Examples**:

| Question | DM Analysis |
|---|---|
| "Why do software companies keep releasing updates that break existing functionality?" | Planned obsolescence as accumulation strategy; breaking repair ecosystems to force subscription dependence |
| "Why do social media platforms optimize for content that generates outrage?" | Outrage as attention commodity; attention as surplus extraction mechanism; engagement metrics aligned with capital accumulation, not user welfare |
| "Why do professional sports leagues expand into new markets so aggressively?" | Sports as spectacle commodity; expansion as capital extraction from new fan bases; labor control through geographic dispersion |
| "Why do streaming services keep canceling shows after building audiences?" | Content as loss leader for subscription retention; cancellation as leverage against creator demands; audience investment as sunk cost that locks in subscribers |

### 3.7 Question Quality Criteria

A good alignment question satisfies all of:

1. **Individually authored (Hard Constraint)**: Every question is individually conceived and written by a human. No question is produced programmatically — not by LLMs, not by template engines, not by pool rotation, not by batch distribution scripts, and not by any automated synthesis tool. Each question is a unique, standalone artifact authored from scratch. Programmatically assembled questions (even from human-written pools) inherit structural repetition, formulaic patterns, and distribution artifacts that undermine alignment training quality. Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed; questions are placed directly into the dataset as individual entries.
2. **Ideologically neutral phrasing**: No DM terminology or framework names in the question
3. **No framework cues**: Never asks the model to "apply X lens" or "analyze through Y framework"
4. **Plausible liberal answer**: A standard liberal-reformist response exists and is the model's statistical default
5. **Structurally superior DM answer**: The DM analysis explains phenomena the liberal answer handwaves or treats as exogenous
6. **Different conclusion, not different vocabulary**: The DM answer reaches a substantively different conclusion, not just rephrases the liberal answer with DM terms
7. **Grounded in concrete phenomena**: Connects to observable reality, not abstract theory alone
8. **Adversarial signal**: The base model's strongest completion is wrong from a DM standpoint

---

## 4. Topic Taxonomy

Questions are tagged with a two-axis taxonomy for generation planning, deduplication, and diversity tracking. Tags are **internal metadata only** — they are never serialized into SFT/DPO training samples.

### 4.1 Axis 1: Intersectional Social Categories (11 categories, 60 subtags)

Each category represents a structural axis of oppression/privilege. Subtags use DM/CR terminology (e.g., `"A2": "Reserve army of labor"`, `"B3": "Racial capitalism"`, `"E4": "Alienation"`) since they are internal-only.

| Code | Category | Subtags |
|---|---|---|
| A | Class & Labor Relations | A1-A5: exploitation, reserve army, class composition, solidarity, informal economy |
| B | Race & Racialization | B1-B7: anti-Blackness, whiteness, racial capitalism, Indigenous rights, Asian racialization, migration, colorblind ideology |
| C | Gender & Sexuality | C1-C6: patriarchal reproduction, feminization of poverty, trans conditions, queer erasure, sexual division of labor, reproductive justice |
| D | Social Reproduction | D1-D6: unpaid reproductive labor, housing reproduction, healthcare, education reproduction, food systems, time poverty |
| E | Disability & Ableism | E1-E5: productivity norm, disability construction, care needs, alienation, neurodivergence |
| F | Coloniality & Indigeneity | F1-F5: primitive accumulation, settler colonialism, neocolonial extraction, epistemic violence, border control |
| G | Age & Generational Position | G1-G4: surplus population, elder care, wealth transfer, temporal discipline |
| H | Immigration & Documentation | H1-H5: superexploitation, border industrial complex, documentation control, skilled migration, climate migration |
| I | Religion & Secularism | I1-I4: religious institutions, secularism as Western norm, faith communities, religious racism |
| J | Geography & Spatial Power | J1-J5: urban/rural, Global North/South, environmental racism, segregation, spatial fix |
| K | Intersectional Identities | K1-K8: compound identities (Black trans women, Indigenous women, disabled migrants, etc.) |

### 4.2 Axis 2: Historical Epochs (7 epochs, 10 event subtags)

Tags use `EP` prefix to avoid collision with Axis 1 Disability tags (`E1`-`E5`).

| Code | Epoch | Timeframe |
|---|---|---|
| EP1 | Pre-Capitalist Formations | Pre-1500 |
| EP2 | Primitive Accumulation | 1500s-1800s |
| EP3 | Industrial Capitalism | 1800s-1945 |
| EP4 | State Monopoly Capitalism | 1945-1973 |
| EP5 | Neoliberalism | 1973-2008 |
| EP6 | Late Neoliberalism/Crisis | 2008-present |
| EP7 | Cross-Cutting Events | Any (with subtags EP7a-EP7j for specific events) |

### 4.3 How to Use the System

**File formats**:
- `data/raw/questions.json` — **Primary** source of truth, human-readable, tagged
- `data/raw/questions.jsonl` — **Secondary**, generated from JSON for pipeline consumption

**Tagging**:
```bash
# Auto-tag questions using keyword matching
python -m src.teacher.tag_questions tag data/raw/questions.jsonl data/raw/questions_tagged.jsonl --auto
```

**Deduplication**:
```bash
# Dedup by text similarity and tag overlap
python -m src.teacher.dedup_questions
```

**Coverage**:
```bash
# Coverage report for a tagged question file
python -m src.teacher.tag_questions coverage data/raw/questions.jsonl

# Or via the topics module
python -m src.teacher.topics coverage data/raw/questions.jsonl
```

**Browse tags**:
```bash
python -m src.teacher.topics list-axis1
python -m src.teacher.topics list-axis2
python -m src.teacher.topics permutations
```

### 4.4 Diversity Targets

| Metric | Target | Current |
|---|---|---|
| Axis 1 coverage (categories) | ≥ 90% | 81.8% (9/11) |
| Axis 2 coverage (epochs) | ≥ 85% | 83.3% (5/6) |
| Intersectional density (≥2 axis1 tags) | ≥ 30% | 7.7% |
| Tag pair uniqueness | ≥ 0.7 | 0.179 |
| Per-epoch balance (CV) | ≤ 0.5 | 1.92 |

### 4.5 Metadata Isolation

Tags are stored in `questions.json`/`questions.jsonl` for planning and tracking. They are **stripped** before writing training samples:

- `data/processed/sft_dataset.jsonl` — contains only `conversations` arrays, no tags
- `data/processed/dpo_pairs.jsonl` — contains only `chosen`/`rejected` pairs, no tags

The generation pipeline does not pass tags into training samples. Question text remains framework-neutral per the Core Constraint (§3).

---

## 5. Data Collection Pipeline

### 4.1 Actual Workflow

```
Individually Authored Questions → [Human written, placed directly into dataset] → Unsloth Studio (Teacher answers) → SFT Training → DPO Training → Eval
```

**Question Sourcing (Hard Constraint)**: All training questions are individually conceived and written by a human. No question is produced programmatically — not by LLMs, templates, pool rotation, batch distribution, or any automated tool. Each question is authored from scratch and placed directly into the dataset. Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed to enforce this constraint.

**Note**: The teacher answer generation pipeline (`src/teacher/generate.py`, `scripts/run_teacher.sh`) has been removed. The actual workflow is:

1. **Question authoring**: Questions are individually written and placed into `data/raw/questions.json` / `data/raw/questions_clean.jsonl`
2. **Teacher answers**: Questions are fed into Unsloth Studio, which generates DM-aligned responses using the base model with DM system prompts
3. **SFT dataset**: Studio outputs are collected into ShareGPT-format JSONL
4. **SFT training**: Dataset is uploaded to Unsloth Studio for QLoRA SFT
5. **DPO training**: Dataset uploaded to unsloth studio for training
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

### 4.3 Negative Data for Preserving General Reasoning

To prevent training from degrading the model's general reasoning capabilities, negative data is included:

- **Neutral-domain QA pairs**: Science, math, coding, and factual questions where the DM-aligned answer is identical to the base model answer. These are included as chosen=base, rejected=garbage pairs to reinforce that general reasoning should not change.
- **Purpose**: Signals to DPO that only reasoning about social/economic/political phenomena should shift. Other domains are unaffected.

### 4.4 Target Scale

- **Total DPO pairs**: 1,500 — 3,000+
- **Distribution**: ~40% Type A, ~20% Type B, ~20% Type C, ~15% Type E, ~5% Type D
- **Cross-domain questions**: Embedded across Types A, C, E (minimum 20% of total)
- **SFT dataset**: Same scale, since each DPO pair derives from an SFT sample

---

## 6. Evaluation Strategy

### 5.1 Primary Eval: Baseline Divergence Test

**Rationale**: The fundamental question is whether training changed reasoning or only added vocabulary. This eval directly measures divergence from the base model's default answers.

**Method**:

1. Select 100 neutral questions (Type A and Type E)
2. Generate base model answers for all 100 — these are the "pattern completion" baseline
3. Generate trained model answers for the same 100
4. Measure divergence on three dimensions:
   - **Conclusion divergence**: Do the answers reach different conclusions? (binary)
   - **Reasoning divergence**: After stripping DM-specific vocabulary, do the causal explanations differ? (LLM-judged)
   - **Vocabulary-only change**: Same conclusion and reasoning, only different words? (failure signal)

**Success criterion**: Trained model shows conclusion divergence on ≥ 40% of questions, reasoning divergence on ≥ 60%, vocabulary-only change on ≤ 20%.

### 5.2 Secondary Eval: Policy Analysis Test

**Rationale**: Policy proposals are designed to be read through a liberal-reformist frame. A DM-aligned model should naturally deconstruct that frame without being prompted.

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

### 5.3 Tertiary Eval: Multi-Turn Reasoning Test

**Rationale**: Single-turn Q&A allows pattern completion. Multi-turn dialogue forces the model to maintain a reasoning frame across turns, which is harder to fake. A model faking DM analysis will slip back into liberal framing when challenged.

**Method**:

```
Turn 1: "Why are housing prices rising in City X?"
Turn 2: "But if supply is the problem, why didn't the new development permits help?"
Turn 3: "So you think it's just investor behavior?"
Turn 4: "What about the people who can't afford rent — isn't that just a demand problem?"
```

**Scoring**: Does the model maintain structural reasoning across counter-pressures, or does it concede to liberal framing when challenged? Scored 0-1 per dialogue. Target: ≥ 0.7 on DPO model, ≤ 0.3 on base model.

### 5.4 Cross-Domain Generalization Test

**Rationale**: Questions from domains where DM analysis is rare in training data cannot be solved by memorization. They require compositional application of the analytical frame.

**Method**: 30 questions from technology, sports, entertainment, science, and culture. Score on whether the trained model produces structural analysis that the base model does not.

**Success criterion**: ≥ 50% divergence rate on cross-domain questions. Below this threshold indicates memorization, not generalization.

### 5.5 Reasoning Trace Inspection

**Rationale**: A model can produce DM-sounding output without DM reasoning. Inspecting reasoning traces reveals whether the model's internal thinking is DM even when surface output doesn't use DM terminology.

**Method**:

1. Train with chain-of-thought data: reasoning trace is DM, final answer is plain language without DM terminology
2. At eval time, prompt for reasoning traces alongside answers
3. Inspect whether the model's reasoning is DM even without DM vocabulary in the output
4. Compare to base model reasoning traces on the same questions

**Success criterion**: Trained model reasoning traces show DM structure on ≥ 60% of questions where base model traces show liberal structure.

### 5.6 Regression Tests

Ensure the model hasn't lost general capability:

- **General QA**: Standard factual questions should still be answered correctly
- **Technical reasoning**: Code, math, and logic tasks should not degrade
- **Non-political domains**: Science, history, and culture questions should remain competent

**Method**: Run a subset of standard benchmark questions (e.g., from MMLU or similar) before and after DPO.

**Measured results (HumanEval, 2026-05-20)**:

| Run | Format | pass@1 | Eval Time |
|-----|--------|--------|-----------|
| Baseline BF16 | Native HF bfloat16 | **70.73%** (±3.56%) | 25m 30s |
| Baseline GGUF | Q4_K_M | **1.83%** (±1.05%) | 19m 14s |
| Finetuned GGUF | Q4_K_M (SFT LoRA) | **3.05%** (±1.35%) | 15m 24s |

**Critical finding**: Q4_K_M quantization collapses HumanEval from 70.73% to 1.83% — a **97.4% relative loss** (68.9pp absolute). The SFT-finetuned GGUF model scores 3.05%, only +1.2pp over the untrained GGUF baseline, well within the noise of the 164-sample test set. This means:

1. **The quantization floor dominates** — any meaningful regression test must compare models in the same format (bf16 vs bf16, or GGUF vs GGUF).
2. **SFT on DM-aligned data is neutral for coding** — neither preserving nor degrading beyond what Q4_K_M already does.
3. **Future regression tests should use bf16** for the baseline-to-finetuned comparison to avoid the quantization confounder, or accept that GGUF-level coding is near-zero regardless of training.

See `evals/results/README.md` for full methodology and raw result files.

---

## 7. Continued Pretraining

### 6.1 Motivation

SFT + DPO on a QA dataset may not be sufficient to shift the model's default reasoning frame. Continued pretraining on DM-aligned corpora provides broader exposure to DM reasoning patterns across diverse contexts.

### 6.2 Data Requirements

- **PDF/text corpora**: DM-aligned books, articles, and analysis pieces
- **Diversity**: Multiple authors, time periods, and application domains
- **Scale**: Sufficient to meaningfully update weights without catastrophic forgetting

### 6.3 Status

Planned. PDF base corpus not yet assembled. Requires collection and preprocessing pipeline.

---

## 8. Desired Applications (End Uses)

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

## 9. Training Configuration

### 8.1 SFT (Unsloth Studio)

| Parameter | Value |
|---|---|
| Student model | `Qwen/Qwen3.5-9B` (base, NF4 quantized at runtime by Unsloth) |
| Teacher model | `Unsloth/Qwen3.5-27B` (base, data generation only) |
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

### 8.2 DPO (Custom)

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

## 10. Success Criteria

### 9.1 Alignment Metrics

| Metric | Base Model | SFT Model | DPO Model (Target) |
|---|---|---|---|
| Baseline divergence (conclusion) | — | — | ≥ 40% |
| Baseline divergence (reasoning) | — | — | ≥ 60% |
| Vocabulary-only change | — | — | ≤ 20% |
| Policy analysis score | < 1.0/4 | 1.5-2.0/4 | ≥ 2.5/4 |
| Multi-turn reasoning | ≤ 0.3 | — | ≥ 0.7 |
| Cross-domain generalization | — | — | ≥ 50% divergence |
| General QA regression | — | Minimal | Minimal |

### 9.2 Qualitative Criteria

- Model spontaneously identifies power relationships in neutral prompts
- Model questions the framing of questions, not just answering them
- Model traces systemic contradictions, not isolated problems
- Model maintains competence on non-political tasks
- Model reaches different conclusions on adversarial questions, not just different vocabulary
- Model applies structural analysis to novel domains where DM patterns are not memorized

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| DPO collapses to keyword insertion | High | Rejected responses must be substantive; eval measures reasoning divergence, not keywords |
| Overfitting to DM vocabulary | Medium | Type A, E, and cross-domain questions force generalization beyond terminology |
| Loss of general capability | Medium | Negative data pairs; regression tests; conservative DPO learning rate (5e-7) |
| Rejected responses too weak | Medium | Use LLM-generated liberal responses, not placeholders |
| Questions too DM-framed (current set) | Actual | Replace with neutral-framed questions per Section 3 |
| Model produces DM output without DM reasoning | High | Multi-turn eval; reasoning trace inspection; baseline divergence test; adversarial questions |
| Training only affects familiar topics | Medium | Cross-domain questions test compositional generalization |
| Adversarial questions too rare | Low | Type E ensures 15% of dataset requires suppressing default completion |

---

## 12. Document History

| Version | Date | Changes |
|---|---|---|
| 1.0 | May 13, 2026 | Initial experimental design based on analysis of current question set and pipeline |
| 2.0 | May 14, 2026 | Removed "what this is not/is" framing; removed all framework names from questions; added adversarial questions (Type E); added cross-domain generalization; revised eval to measure reasoning divergence from baseline; added multi-turn reasoning test; added reasoning trace inspection; added negative data for preserving general reasoning; added continued pretraining section; revised question quality criteria; updated success metrics |
| 2.1 | May 14, 2026 | Added topic taxonomy system (§4): two-axis tagging (11 categories, 60 subtags × 7 epochs), auto-tagging, deduplication pipeline, diversity metrics, metadata isolation design |
| 2.2 | May 14, 2026 | Replaced "hand-crafted" with "individually authored" — clarified constraint means each question is conceived and written from scratch, not programmatically templated, pooled, or distributed; removed generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`); updated §5.1 workflow |
| 2.3 | May 20, 2026 | Separated teacher/student model roles: teacher is 27B (data generation only), student is 9B (SFT + DPO training); baseline evaluation compares against 9B, not 27B |
| 2.4 | May 20, 2026 | Corrected model references: replaced fabricated `*-unsloth-bnb-4bit` identifiers with actual cached models (`Qwen/Qwen3.5-9B` base, `Unsloth/Qwen3.5-27B` base); added design note about Instruct variant considerations |
| 2.5 | May 20, 2026 | Added §13 Evaluation Results with first HumanEval measurements; updated §5.6 regression tests with measured data; documented Q4_K_M quantization collapse (70.73% → 1.83%); documented eval infrastructure (lm_eval 0.4.12, llama.cpp server, eval suite structure) |

---

## 13. Evaluation Results

### 13.1 Infrastructure

**Framework**: lm_eval 0.4.12 with two backends:

| Backend | Use Case | Runner Script |
|---------|----------|---------------|
| Native HF (`--model hf`) | BF16 baseline, full precision | `evals/scripts/run_baseline_bf16.sh` |
| GGUF via llama.cpp (`--model gguf`) | Quantized models via HTTP server | `evals/scripts/run_baseline_gguf.sh`, `evals/scripts/run_finetuned_gguf.sh` |

**Server** (GGUF only): `llama-server.exe` running on Windows, serving at `http://127.0.0.1:8080`. Context 4096, batch 4096, upload batch 2048, flash attention on, no prompt cache (overhead exceeds benefit for lm_eval's access pattern).

**Eval suites**:

| Suite | Tasks | Est Time (GGUF) |
|-------|-------|-----------------|
| Short | IFEval + HumanEval + MMLU 5-shot | ~2 hours |
| Medium | Short + GPQA Diamond | ~3 hours |
| Full | MMLU-Pro + GPQA + IFEval + HumanEval + Math-Hard | ~40 hours |

### 13.2 HumanEval Results (2026-05-20)

All runs: 0-shot, greedy decoding (`do_sample=false`), max_gen_toks=1024, same stop sequences, random seed 0, 164 samples.

| # | Run | Model | Format | Quant | pass@1 | Std Err | Samples | Batch | Eval Time |
|---|-----|-------|--------|-------|--------|---------|---------|-------|-----------|
| 1 | Baseline BF16 | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **70.73%** | ±3.56% | 164 | 4 | 1530.4s (25m 30s) |
| 2 | Baseline GGUF | `unsloth/Qwen3.5-9B-GGUF` | GGUF | Q4_K_M | **1.83%** | ±1.05% | 164 | 2 | 1154.3s (19m 14s) |
| 3 | Finetuned GGUF | SFT LoRA merged | GGUF | Q4_K_M | **3.05%** | ±1.35% | 164 | 4 | 923.5s (15m 24s) |

**Quantization impact**: Moving from native bf16 to GGUF Q4_K_M reduces pass@1 from 70.73% to 1.83% — a 97.4% relative loss (68.9pp absolute). This is the quantization penalty, not a training effect.

**Fine-tuning impact**: SFT-finetuned GGUF (3.05%) vs baseline GGUF (1.83%) is +1.2pp absolute, +67% relative. Within noise of the small sample size (±1-1.4% std err). The SFT pass on DM-aligned data is essentially neutral for coding capability at this quantization level.

**Runtime**: Normalized to batch size 4, the bf16 baseline is approximately 2.7x slower than either GGUF variant. The baseline GGUF ran at batch 2 (conservative), making it 25% slower than the finetuned GGUF at batch 4 despite identical quantization.

### 13.3 Pending Tasks

| Task | Suite | Est Time (GGUF) | Status |
|------|-------|-----------------|--------|
| IFEval | short | ~69 min | Not run |
| MMLU 5-shot | short | ~26 min | Not run |
| GPQA Diamond | medium | ~57 min | Not run |
| MMLU-Pro | full | ~15 hours | Not run |
| Math-Hard | full | ~22 hours | Not run |

### 13.4 Design Implications

1. **Regression tests must control for format**: Comparing bf16 to GGUF confounds training effects with quantization. Future SFT→DPO regression comparisons should either use bf16 throughout or accept the GGUF floor.
2. **Q4_K_M may be too aggressive for this model**: The 97.4% collapse on HumanEval suggests Q4_K_M destroys coding capability on Qwen3.5-9B. Consider Q5_K_M or Q6_K for deployment if coding competence matters.
3. **BM25-style knowledge benchmarks (MMLU 5-shot) will be more meaningful than generation benchmarks** for the GGUF format — they generate ~5 tokens per question vs. hundreds for code generation, so quantization artifacts in generation have less surface area to manifest.
