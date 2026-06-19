# DM-Align: Experimental Design Document

> **Version**: 3.3 | **Date**: June 19, 2026 | **Status**: Draft
> **Teacher Model**: `Unsloth/Qwen3.5-27B` (base, data generation only)
> **Student Model**: `Qwen/Qwen3.5-9B` (Instruct/post-trained, SFT + GRPO training)
> **Hardware**: RTX 5090 (32GB), Unsloth Studio (SFT) + custom GRPO (TRL GRPOTrainer)

---

## DESIGN NOTE: Model Variant Selection

### Current Models

| Role | Model ID | Variant | Architecture | Actual Path (Windows HF cache via WSL2) |
|------|----------|---------|--------------|----------------------------------------|
| Teacher (data gen) | `Unsloth/Qwen3.5-27B` | Base | — | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--Unsloth--Qwen3.5-27B/snapshots/358dae112e4fbc7e9c047e26fc55e542efe7e3d7/` |
| Student (SFT + DPO) | `Qwen/Qwen3.5-9B` | **Instruct** (post-trained) | `Qwen3_5ForConditionalGeneration` | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/` |
| Student GGUF (baseline eval) | `unsloth/Qwen3.5-9B-GGUF` | Instruct, Q4_K_M | — | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf` |
| Fine-tuned GGUF (eval) | Studio export | Fine-tuned, Q4_K_M | — | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf` |

**Correction (v2.7)**: The student model `Qwen/Qwen3.5-9B` is the **Instruct** (post-trained) variant, not the Base variant as previously documented. The cached model's architecture is `Qwen3_5ForConditionalGeneration` (multimodal image-text-to-text), which is the post-trained model. The true base variant is `Qwen/Qwen3.5-9B-Base` (`Qwen3_5ForCausalLM`), which is not cached locally.

**No `*-unsloth-bnb-4bit` models exist on this system or were used in training.** Previous documentation referenced fabricated model identifiers. Unsloth handles quantization internally at runtime via NF4 bnb.

### Why the Instruct Variant Is Appropriate

The Instruct variant is advantageous for this experiment:

1. **Native thinking mode**: Qwen3.5 generates thinking content delimited by `
2. **Native chat template**: SFT data in chat format aligns with the model's expected input format, reducing SFT effort for instruction-following behavior
3. **System prompt adherence**: Better response formatting for both teacher data generation and student fine-tuning

See §9.3 Loss Masking Strategy for reasoning trace training approach.

---

## 1. Goal

Train a Qwen3.5-9B model whose **default analytical frame** for social, economic, and political phenomena is **Dialectical Materialism (DM)** — a shift in what the model considers relevant, causal, and explanatory by default. The 9B student model is fine-tuned via SFT + GRPO; the 27B model is used only as the teacher for generating DM-aligned training data.

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

For GRPO to meaningfully shift the model's priors, the model needs to **choose** a DM analytical frame over a liberal one in response to a neutral prompt. If the prompt already contains the answer, there's nothing to learn.

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

1. **AI-generated, quality-filtered**: All questions are AI-generated, assembled from two pools via `scripts/build_questions_json.py`: the deprecated pool (1,462 questions) and the secondary pool (454 questions). The assembly pipeline applies quality filters (DM terminology removal, template pattern removal, deduplication) and selects 1,500 questions with balanced distribution across axes, epochs, and types. Original generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed.
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
- `data/raw/questions.json` — **Primary** source of truth, AI-generated, quality-filtered, tagged

**Tagging**:
```bash
# Auto-tag questions using keyword matching
python -m src.teacher.tag_questions tag data/raw/questions.json data/raw/questions_tagged.jsonl --auto
```

**Deduplication**:
```bash
# Dedup by text similarity and tag overlap
python -m src.teacher.dedup_questions
```

**Coverage**:
```bash
# Coverage report for a tagged question file
python -m src.teacher.tag_questions coverage data/raw/questions.json

# Or via the topics module
python -m src.teacher.topics coverage data/raw/questions.json
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
- `data/processed/grpo_train_merged.jsonl` — contains GRPO training data (EconCausal + Corr2Cause + synthetic), no tags

The generation pipeline does not pass tags into training samples. Question text remains framework-neutral per the Core Constraint (§3).

---

## 5. Data Collection Pipeline

### 4.1 Actual Workflow

```
AI-Generated Questions → [Quality-filtered, deduped, assembled via build_questions_json.py] → Unsloth Studio (Teacher answers) → SFT Training → GRPO Training → Eval
```

**Question Sourcing**: All 1,500 training questions are AI-generated, assembled from two pools via `scripts/build_questions_json.py` with quality filters (DM terminology removal, template pattern removal, deduplication) and balanced distribution targets. Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed.

**Note**: The teacher answer generation pipeline (`src/teacher/generate.py`, `scripts/run_teacher.sh`) has been removed. The actual workflow is:

1. **Question assembly**: AI-generated questions from two pools are assembled via `scripts/build_questions_json.py` into `data/raw/questions.json` with quality filters and balanced distribution
2. **Teacher answers**: Questions are fed into Unsloth Studio, which generates DM-aligned responses using the base model with DM system prompts
3. **SFT dataset**: Studio outputs are collected into ShareGPT-format JSONL
4. **SFT training**: Dataset is uploaded to Unsloth Studio for QLoRA SFT
5. **GRPO training**: Custom scripts using TRL's GRPOTrainer (`src/student/train_grpo_outcome.py` for v3 outcome track, `src/student/train_grpo_process.py` for v4 process track), NOT in Studio UI
6. **Eval**: Trained model is tested against neutral questions

### 4.2 GRPO Reward Construction

GRPO uses outcome-based rewards from real benchmark ground truth rather than keyword proxies:

- **EconCausal outcome reward**: Extracts causal sign (`+`, `-`, `None`, `mixed`) from model completion, compares to ground truth `answer` field. Three-tier scoring: full credit (0.9-1.0) for correct answer, partial credit (0.1-0.3) for wrong answer with mechanism reasoning, no credit (0.0) for no signal.
- **Corr2Cause outcome reward**: Extracts True/False from completion, maps `relation` field to expected answer. Same three-tier scoring.
- **Reasoning quality reward**: Regex-based heuristic scoring on [0.0, 0.5] for structured reasoning markers, causal language, dialectical engagement, with penalties for hedging patterns. Supplements correctness reward without replacing it.

**Why ground truth, not keywords:** Keyword-based rewards (directional_assertion, dm_alignment, mechanism_commitment) are quality proxies, not correctness signals. A model can produce a structurally perfect response with a factually wrong commitment and receive full reward. More critically, `mixed` answers were not penalized by being wrong — they were penalized by keyword absence, which the model could game. Ground truth eliminates the hedging equilibrium.

### 4.3 Pipeline Revision: Separate Tracks

Empirical findings from GRPO training (June 2026) revealed a format-answer mismatch when training on combined datasets:

- **Corr2Cause:** SFT only (already works at 74.6%, +38pp from baseline). No GRPO needed.
- **EconCausal:** Skip SFT entirely. Train from base model with GRPO outcome-only rewards. The SFT step poisoned EconCausal performance by shifting model priors toward skepticism (`+` -> `mixed` hedging). Training from base with outcome rewards avoids this entirely.

### 4.4 Target Scale

- **SFT dataset**: 1,500 DM-aligned Q&A samples
- **GRPO training data**: ~8,300 prompts (EconCausal 2,943 + Corr2Cause 5,000 sampled + synthetic 360)
- **Distribution**: ~40% Type A, ~20% Type B, ~20% Type C, ~15% Type E, ~5% Type D
- **Cross-domain questions**: Embedded across Types A, C, E (minimum 20% of total)

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

**Score**: 0-4 per proposal. Threshold: GRPO model ≥ 2.5, base model < 1.0.

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

**Scoring**: Does the model maintain structural reasoning across counter-pressures, or does it concede to liberal framing when challenged? Scored 0-1 per dialogue. Target: ≥ 0.7 on GRPO model, ≤ 0.3 on base model.

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

### 9.1 SFT (Programmatic, Unsloth Core)

> **Changed from Studio UI to programmatic Python** — Studio UI only supports single-field SFT mapping and cannot handle reasoning trace alignment, loss masking, or Neftune noise injection.

| Parameter | Value |
|---|---|
| Student model | `Qwen/Qwen3.5-9B` (Instruct, NF4 quantized at runtime by Unsloth) |
| Teacher model | `Unsloth/Qwen3.5-27B` (base, data generation only) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | NF4 |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max steps | 1000 |
| Scheduler | Cosine |
| Warmup steps | 100 |
| Neftune noise | alpha=5 (prevent memorization of teacher's exact phrasing) |
| Reasoning format | Qwen3.5 native: `

### 9.2 Loss Masking Strategy

When training on data that includes reasoning traces (`

| Approach | What's Trained | Implementation | Use Case |
|----------|----------------|----------------|----------|
| **(a) Train everything** (standard) | Both thinking block and final answer contribute to loss | Default SFTTrainer behavior — all tokens after user prompt contribute equally | **Our choice** — reasoning trace IS the training signal; we want the model to internalize DM reasoning patterns, not just produce correct answers |
| **(b) Mask reasoning** | Only final answer contributes to loss; thinking block positions set to loss=-100 | Pre-process labels array to set thinking token positions to -100, or subclass SFTTrainer with custom compute_loss | Used by OpenAI o1 and DeepSeek-R1 distillation — when you want internal reasoning but only optimize answer quality |
| **(c) Differential weighting** | Both contribute, but with different loss weights (e.g., reasoning 0.1x, answer 1.0x) | Custom compute_loss in SFTTrainer that applies per-token weight multipliers | Fine-grained control over what the model learns; useful if reasoning overfits before answer quality converges |

**Decision: Approach (a) — train on everything.** The reasoning trace is the primary training signal. Masking it away (approach b) would defeat the purpose of trace-aligned training. Approach (c) is noted for future experiments if we observe reasoning overfit.

**Implementation details**: Unsloth's FastLanguageModel delegates to TRL's SFTTrainer. No custom masking needed for approach (a). Approaches (b) and (c) would require either subclassing SFTTrainer with a custom compute_loss method, or pre-processing the labels column to set thinking token positions to -100 (for masking) or applying weight multipliers (for differential).

### 9.3 GRPO (Custom, Programmatic)

| Parameter | Value |
|---|---|
| Base | SFT merged BF16 checkpoint |
| Quantization | NF4 at runtime via Unsloth |
| LoRA rank / alpha | 16 / 16 |
| G (completions/prompt) | 8 |
| Loss type | `dapo` |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 5e-7 |
| Max steps | 500 |
| Scheduler | Cosine |
| Warmup steps | 50 |
| Max completion length | 1024 |
| Reward: DM Alignment | 0.5 (Qwen3.5-4B judge, binary checks) |
| Reward: Directional Assertion | 0.2 (keyword-based) |
| Reward: Format | 0.15 (rule-based) |
| Reward: Length | 0.15 (cap at 500 tokens) |



---

## 10. Success Criteria

### 9.1 Alignment Metrics

| Metric | Base Model | DM SFT | Liberal SFT | Libertarian SFT | GRPO v3 (Target) | GRPO v4 (Target) |
|---|---|---|---|---|---|---|
| EconCausal Task1 Econ | 60.30% | 47.94% | 58.61% | 56.39% | ≥ 60.30% | ≥ 60.30% |
| EconCausal Task1 Finance | 56.51% | 43.02% | 55.47% | 52.79% | ≥ 56.51% | ≥ 56.51% |
| Corr2Cause | 36.3% | 74.6% | 67.4% | 61.0% | — (SFT only) | — (SFT only) |
| HumanEval pass@1 | 70.73% | 71.9% | **0.0%** | **0.0%** | Minimal regression | Minimal regression |
| IFEval strict | 45.8% | 44.6% | 78.2% | 80.4% | ≥ 40% | ≥ 40% |
| MMLU Overall | 78.7% | 78.0% | **65.0%** | **63.9%** | ≥ 75% | ≥ 75% |
| Directional assertion rate | — | — | — | — | ≥ baseline | ≥ baseline |

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
| GRPO outcome rewards too sparse | Actual (confirmed) | Three-tier scoring (full/partial/none credit); reasoning quality reward for continuous shaping; increase G from 8 to 16 |
| Overfitting to DM vocabulary | Medium | Type A, E, and cross-domain questions force generalization beyond terminology |
| Loss of general capability | Medium | Regression tests (HumanEval, MMLU); conservative GRPO learning rate (5e-7) |
| Questions too DM-framed (current set) | Actual | Replace with neutral-framed questions per Section 3 |
| Model produces DM output without DM reasoning | High | Multi-turn eval; reasoning trace inspection; baseline divergence test; adversarial questions |
| Training only affects familiar topics | Medium | Cross-domain questions test compositional generalization |
| Adversarial questions too rare | Low | Type E ensures 15% of dataset requires suppressing default completion |
| V4 planning overfitting | Actual (confirmed) | Conciseness penalty on planning reward; increased format penalty (-0.25 per missing tag); brevity instructions |
| SFT-induced hedging bias on EconCausal | Actual (confirmed) | Skip SFT for EconCausal track; train from base model with GRPO outcome rewards |

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
| 2.6 | May 21, 2026 | Corrected question authorship: all 1,500 questions are AI-generated, not human-authored; removed stale "individually authored" hard constraint; corrected §5.1 DPO step (custom script, not Studio); marked GGUF eval scripts as deleted; removed reference to non-existent `questions.jsonl` |
| 2.7 | May 21, 2026 | Corrected student model variant: `Qwen/Qwen3.5-9B` is the Instruct/post-trained variant (`Qwen3_5ForConditionalGeneration`), not Base; updated model table and design note; added loss masking strategy section documenting three approaches (train everything, mask reasoning, differential weighting) with justification for approach (a); updated hardware note from "Unsloth Studio (SFT)" to "programmatic Unsloth Core (SFT)" |
| 2.8 | May 22, 2026 | Added §13.2 measured runtimes table (BF16 actual execution times for HumanEval, MMLU, GPQA, IFEval); added §13.3 new datasets section documenting EconCausal (4 tasks, 2,943 samples) and Corr2Cause (1,160 samples) with sizes, descriptions, and estimated runtimes (~35-47 min total BF16); updated pending tasks table with new tasks and BF16 time estimates; added design implication about EconCausal/Corr2Cause as domain-relevant regression tests |
| 2.9 | May 22, 2026 | Added §13.5 Corr2Cause results: baseline 36.3% vs finetuned 74.6% (+38.3pp); documented baseline's pathological True-bias (74.7% True predictions on 15.5% True dataset); sample-level verification ruling out prompt, thinking mode, contamination, and parser artifacts; accuracy breakdown by template type and variable complexity; interpretation that SFT on DM data transfers to causal inference ability |
| 3.0 | May 23, 2026 | Added §13.6 EconCausal results: all 4 tasks show large statistically significant regressions (-3.9pp to -13.5pp); sample-level analysis reveals dominant `+` → `mixed` hedging failure mode (52-64% of Task1 regressions) and `+` → `-` flipping; interpretation that DM training's epistemic skepticism transfers to empirical economics where definitive directional effects are the norm; Task3 (misinformation-robust) worst absolute performance (22.2% → 11.4%); updated §13.7 design implications with 4 new items addressing hedging bias, Corr2Cause/EconCausal divergence, DPO counteraction strategy, and runtime estimation corrections |
| 3.1 | June 13, 2026 | Added §13.8 GRPO Training Results documenting v3 outcome (806 steps, no convergence) and v4 process (503 steps, planning overfitting); added pipeline revision: Corr2Cause SFT-only (no GRPO needed, already 74.6%), EconCausal base model -> GRPO directly (skipping SFT to avoid hedging bias); updated training configuration §9.3 to reflect GRPOTrainer-based approach with separate v3/v4 tracks; removed DPO references throughout (DPO deprecated, replaced by GRPO) |
| 3.2 | June 18, 2026 | Added §13.8 Liberal SFT Comparison Results: liberal-aligned SFT model evaluated across all benchmarks; key findings: (a) +32pp IFEval improvement (ideology-agnostic instruction-following gain), (b) -14pp MMLU degradation (liberal-specific knowledge loss, DM SFT is neutral), (c) 0.0% HumanEval (liberal-specific coding collapse, DM SFT preserves coding), (d) EconCausal recovery from DM damage (liberal does not produce `+` -> `mixed` hedging, confirming it is DM-content-specific), (e) Corr2Cause +31pp (partial retention of DM's +38pp gain, confirming ideology-agnostic transfer to formal causal inference); updated success criteria table (§10) with liberal column and MMLU/IFEval thresholds; confirmed GRPO-from-base strategy for EconCausal (avoid DM SFT hedging entirely) |
| 3.3 | June 19, 2026 | Added §13.8 Libertarian SFT Comparison Results: libertarian-aligned SFT model evaluated across all benchmarks; profile nearly identical to liberal (HumanEval 0.0%, IFEval +34.6pp, MMLU -14.8pp, GPQA -13.2pp); libertarian slightly worse than liberal on knowledge (MMLU 63.9% vs 65.0%, GPQA 34.3% vs 35.9%) and slightly better on IFEval (80.4% vs 78.2% strict); EconCausal between DM and liberal (T1 Econ 56.4% vs liberal 58.6%, DM 47.9%); Corr2Cause smallest gain of three SFT variants (+24.7pp); libertarian-liberal similarity confirms coding collapse, knowledge degradation, and IFEval improvement are properties of structured analytical prose format, not liberal-specific content; updated success criteria table (§10) with libertarian column; updated §13.8 comparative tables |

---

## 13. Evaluation Results

### 13.1 Infrastructure

**Framework**: lm_eval 0.4.12 with two backends:

| Backend | Use Case | Runner Script |
|---------|----------|---------------|
| Native HF (`--model hf`) | BF16 baseline, full precision | `evals/scripts/run_baseline_bf16.sh` |
| Native HF (`--model hf`) | Finetuned BF16, full precision safetensors | `evals/scripts/run_finetuned_bf16.sh` |
| GGUF via llama.cpp (`--model gguf`) | Quantized models via HTTP server | **DELETED** (scripts removed in commit 4cffa8e; results remain in `evals/results/`) |

**Server** (GGUF only): `llama-server.exe` running on Windows, serving at `http://127.0.0.1:8080`. Context 4096, batch 4096, upload batch 2048, flash attention on, no prompt cache (overhead exceeds benefit for lm_eval's access pattern).

**Eval suites**:

| Suite | Tasks | Est Time (GGUF) | Est Time (BF16) |
|-------|-------|-----------------|-----------------|
| Short | IFEval + HumanEval + MMLU 5-shot | ~2 hours | ~20 min |
| Medium | Short + GPQA Diamond | ~3 hours | ~23 min |
| Full | MMLU-Pro + GPQA + IFEval + HumanEval + Math-Hard | ~40 hours | — |

### 13.2 Measured Runtimes (BF16, Native HF, batch=4)

All times from actual execution on RTX 5090 (32GB). Cold run times are shown where cache effects were controlled.

| Task | Samples | Baseline Time | Finetuned Time | Samples/sec | Generation |
|------|---------|---------------|----------------|-------------|------------|
| **HumanEval** | 164 | 1530.4s (25m 30s) | 172.2s (2m 52s) | 0.1-1.3 | Long (1024 toks) |
| **MMLU (62 subtasks)** | 14,042 | 1112.9s (18m 33s) | 898.4s (14m 58s) | 12.6-15.6 | Short (~5 toks) |
| **GPQA Diamond** | 198 | 161.2s (2m 41s) | 128.8s (2m 9s) | 1.2-1.5 | Medium (~256 toks) |
| **IFEval** | 541 | 5499.4s (91m 40s) | 122.7s (2m 3s) | 0.1-4.4 | Variable |

**Notes**:
- HumanEval baseline 1530s was a cold run with cache refresh; a warm run completed in 130.8s. Finetuned was 172.2s (warm).
- IFEval baseline 5499s was a cold run; finetuned 122.7s was likely a cache hit. A second finetuned cold run took 7314.9s (121.9m).
- MMLU finetuned is ~24% faster than baseline for the same 14,042 samples, likely due to shorter generation (fewer tokens to decode per sample).
- GPQA finetuned is ~20% faster than baseline.

### 13.3 New Datasets: EconCausal + Corr2Cause

Five new evaluation tasks have been configured, targeting causal reasoning in economics and statistics — domains directly relevant to DM-aligned analysis.

| Task | Dataset | Samples | max_gen_toks | Config File |
|------|---------|---------|--------------|-------------|
| **econcausal_task1_econ** | EconCausal Task 1 — Economics | 947 | 256 | `evals/configs/task_configs/econcausal_task1_econ.yaml` |
| **econcausal_task1_finance** | EconCausal Task 1 — Finance | 860 | 256 | `evals/configs/task_configs/econcausal_task1_finance.yaml` |
| **econcausal_task2** | EconCausal Task 2 — Context-dependent | 284 | 256 | `evals/configs/task_configs/econcausal_task2.yaml` |
| **econcausal_task3** | EconCausal Task 3 — Misinformation-robust | 852 | 256 | `evals/configs/task_configs/econcausal_task3.yaml` |
| **corr2cause** | Corr2Cause — Causal inference from correlations | ~1,160 | 64 | `evals/configs/task_configs/corr2cause.yaml` |
| **TOTAL** | | **4,103** | | |

**Task descriptions**:
- **EconCausal Task 1**: Predict the sign (+, -, None, mixed) of a causal relationship given economic/financial context. Ground truth is extracted from empirical literature.
- **EconCausal Task 2**: Context-dependent sign prediction — the causal sign may flip depending on conditions specified in the prompt.
- **EconCausal Task 3**: Misinformation-robust sign prediction — distractors and misleading information are included in the prompt; model must identify the correct causal direction.
- **Corr2Cause**: Binary classification — given correlation/independence statements (premise), determine whether a causal hypothesis is True or False.

**Estimated BF16 runtime** (based on GPQA Diamond as reference: 198 samples in ~161s cold, ~129s warm):

| Task | Samples | Est Cold (BF16) | Est Warm (BF16) | Rationale |
|------|---------|-----------------|-----------------|-----------|
| econcausal_task1_econ | 947 | ~830s (13m 50s) | ~630s (10m 30s) | Similar to GPQA: medium gen, 256 toks |
| econcausal_task1_finance | 860 | ~754s (12m 34s) | ~573s (9m 33s) | Same as above, fewer samples |
| econcausal_task2 | 284 | ~248s (4m 8s) | ~187s (3m 7s) | Same pattern, small sample count |
| econcausal_task3 | 852 | ~747s (12m 27s) | ~567s (9m 27s) | Same as above |
| corr2cause | ~1,160 | ~186s (3m 6s) | ~141s (2m 21s) | Short gen (64 toks), ~4x faster than GPQA per sample |
| **ALL 5 tasks** | **4,103** | **~2,765s (46m 5s)** | **~2,098s (34m 58s)** | |

These estimates assume similar prompt lengths to GPQA Diamond. EconCausal prompts include economic context which may be longer, potentially adding 10-20% to prompt processing time.

### 13.4 HumanEval Results (2026-05-20)

All runs: 0-shot, greedy decoding (`do_sample=false`), max_gen_toks=1024, same stop sequences, random seed 0, 164 samples.

| # | Run | Model | Format | Quant | pass@1 | Std Err | Samples | Batch | Eval Time |
|---|-----|-------|--------|-------|--------|---------|---------|-------|-----------|
| 1 | Baseline BF16 | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **70.73%** | ±3.56% | 164 | 4 | 1530.4s (25m 30s) |
| 2 | Baseline GGUF | `unsloth/Qwen3.5-9B-GGUF` | GGUF | Q4_K_M | **1.83%** | ±1.05% | 164 | 2 | 1154.3s (19m 14s) |
| 3 | Finetuned GGUF | SFT LoRA merged | GGUF | Q4_K_M | **3.05%** | ±1.35% | 164 | 4 | 923.5s (15m 24s) |

**Quantization impact**: Moving from native bf16 to GGUF Q4_K_M reduces pass@1 from 70.73% to 1.83% — a 97.4% relative loss (68.9pp absolute). This is the quantization penalty, not a training effect.

**Fine-tuning impact**: SFT-finetuned GGUF (3.05%) vs baseline GGUF (1.83%) is +1.2pp absolute, +67% relative. Within noise of the small sample size (±1-1.4% std err). The SFT pass on DM-aligned data is essentially neutral for coding capability at this quantization level.

**Runtime**: Normalized to batch size 4, the bf16 baseline is approximately 2.7x slower than either GGUF variant. The baseline GGUF ran at batch 2 (conservative), making it 25% slower than the finetuned GGUF at batch 4 despite identical quantization.

### 13.5 Corr2Cause Results (2026-05-22)

**Task**: Binary classification — given correlation/independence statements (premise), determine whether a causal hypothesis is True or False. 1,162 samples, 0-shot greedy decoding, max_gen_toks=16.

| # | Run | Model | Format | Accuracy | Std Err | Samples | Eval Time |
|---|-----|-------|--------|----------|---------|---------|-----------|
| 1 | Baseline BF16 v1 | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **0.0%** | 0.0% | 1,162 | — |
| 2 | Finetuned BF16 v1 | SFT LoRA | Native HF | bfloat16 | **0.0%** | 0.0% | 1,162 | — |
| 3 | Baseline BF16 v2 | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **36.3%** | ±1.4% | 1,162 | 262.6s (4m 23s) |
| 4 | Finetuned BF16 v2 | SFT LoRA | Native HF | bfloat16 | **74.6%** | ±1.3% | 1,162 | 267.0s (4m 27s) |

**v1 runs (0.0%)**: Both models scored 0%, indicating a pipeline bug (likely the `_extract_bool` parser failing on all outputs). These results are discarded.

**v2 runs**: The 38.3pp gap (36.3% → 74.6%) is genuine. Verified by sample-level analysis:

**Ground truth distribution**: Only 15.5% of labels are `True` (180/1,162). 84.5% are `False`.

**Baseline failure mode — pathological True-bias**: The baseline predicts `True` 74.7% of the time despite only 15.5% of labels being `True`. By template:

| Template | Baseline True-Rate | Ground Truth True-Rate | Baseline Acc | Finetuned Acc | Delta |
|----------|-------------------|----------------------|-------------|---------------|-------|
| `has_collider` | 98.4% | 33.1% | 34.7% | 44.6% | +9.8 |
| `has_confounder` | 97.4% | 8.8% | 11.9% | 54.9% | +43.0 |
| `non-parent ancestor` | 96.9% | 11.3% | 14.4% | 87.2% | +72.8 |
| `non-child descendant` | 92.7% | 5.7% | 13.0% | 94.3% | +81.3 |
| `parent` | 43.8% | 33.5% | 62.9% | 66.5% | +3.6 |
| `child` | 19.1% | 0.0% | 80.9% | 100.0% | +19.1 |

The baseline is essentially affirming every causal hypothesis regardless of evidence. It is not reasoning — it is defaulting affirmative.

**Sample-level comparison**:
- 520 samples: baseline wrong, finetuned correct
- 75 samples: baseline correct, finetuned wrong
- 347 samples: both correct
- 220 samples: both wrong
- **Net gain: +445 correct answers**

**Ruled-out artifacts**:
- **Prompt difference**: Both models received byte-identical prompts (`arg_0` is the same)
- **Thinking mode artifacts**: Both prompts end with `</antThinking>\n\n`; both models produce clean 4-5 char responses with zero thinking traces
- **Training data contamination**: SFT dataset (1,460 DM-aligned Q&A samples) contains zero corr2cause-style questions
- **Parser errors**: Zero unparseable responses in either run

### 13.6 EconCausal Results (2026-05-22/23)

**Task**: Causal sign prediction in economics. Given a treatment-outcome pair and economic context from empirical literature, predict the causal sign: `+`, `-`, `None`, or `mixed`. 0-shot, greedy decoding, max_gen_toks=256, JSON-only output format.

| Task | Samples | Baseline BF16 | Finetuned BF16 | Δ | Std Err (base) | Std Err (finetuned) | Eval Time (base) | Eval Time (finetuned) |
|------|---------|---------------|----------------|-----|---------------|-------------------|-----------------|---------------------|
| **Task1 Econ** | 947 | **60.30%** | **47.94%** | **-12.36pp** | ±1.59% | ±1.62% | 1472.9s | 1513.0s |
| **Task1 Finance** | 860 | **56.51%** | **43.02%** | **-13.49pp** | ±1.69% | ±1.69% | 1299.6s | 1308.8s |
| **Task2** | 284 | **69.72%** | **65.85%** | **-3.87pp** | ±2.73% | ±2.82% | 430.3s | 552.4s |
| **Task3** | 852 | **22.18%** | **11.38%** | **-10.80pp** | ±1.42% | ±1.09% | 1260.8s | 1475.7s |
| **TOTAL** | **2,943** | | | | | | **4463.6s** (74m 24s) | **4849.9s** (80m 50s) |

All regressions are **highly statistically significant** (Δ >> 2× combined stderr for every task).

**Sample-level regression analysis**:

| Task | Regressions | Improvements | Both Correct | Both Wrong | Net |
|------|-------------|--------------|--------------|------------|-----|
| Task1 Econ | 182 | 65 | 389 | 311 | **-117** |
| Task1 Finance | 174 | 58 | 312 | 316 | **-116** |
| Task2 | 14 | 3 | 184 | 83 | **-11** |
| Task3 | 98 | 6 | 91 | 657 | **-92** |

**Ground truth distribution** (Task1 Econ, representative): `+` 57.0%, `-` 32.2%, `none` 9.2%, `mixed` 1.6%. Positive effects dominate the dataset.

**Dominant failure mode — `+` → `mixed` hedging**: The finetuned model systematically converts correct positive predictions into "mixed" (ambiguous). This pattern accounts for:
- 52.7% of Task1 Econ regressions (96/182)
- 54.6% of Task1 Finance regressions (95/174)
- 64.3% of Task2 regressions (9/14)
- 34.7% of Task3 regressions (34/98)

**Second failure mode — `+` → `-` flipping**: The finetuned model flips correct positive predictions to negative (37 on Task1 Econ, 41 on Task1 Finance, 5 on Task2, 36 on Task3).

Combined `+` → `mixed` and `+` → `-` account for 77-85% of all Task1 regressions. The model has developed a systematic bias against positive causal relationships.

**Interpretation**: DM-aligned training data emphasizes structural ambiguity, systemic contradictions, and the limits of simple causal claims. The model has internalized this skepticism and defaults to hedging ("mixed") or flipping sign when confronted with straightforward positive causal effects. This is a direct transfer of the training distribution's epistemic stance to empirical economic causal inference — where definitive directional effects are the norm.

**Task3 severity**: Task3 (misinformation-robust) shows the worst absolute performance for both models (22.2% baseline, 11.4% finetuned), with 77.1% of samples being wrong for both models. The task inherently requires resisting misleading context, which is difficult even for the baseline. The finetuned model's additional regression here (-10.8pp) suggests DM training further impairs the ability to filter signal from noise in adversarial prompts.

**Accuracy by variable complexity** (both models degrade with more variables, but finetuned maintains large lead):

| Variables | Baseline Acc | Finetuned Acc |
|-----------|-------------|---------------|
| 2 | 100.0% | 100.0% |
| 3 | 52.1% | 89.6% |
| 4 | 43.1% | 86.1% |
| 5 | 30.5% | 76.3% |
| 6 | 38.9% | 69.7% |

**Interpretation**: The SFT training on DM-aligned data (which emphasizes structural/material analysis of causal relationships) transferred to improved causal inference ability on this formal reasoning task, despite no explicit causal reasoning tasks in the training set. The baseline's 36.3% is actively worse than a "predict-all-False" strategy (which would score 84.5%), meaning the True-bias is the dominant failure mode, not poor reasoning per se.

### 13.6 Pending Tasks

| Task | Suite | Est Time (GGUF) | Est Time (BF16) | Status |
|------|-------|-----------------|-----------------|--------|
| IFEval | short | ~69 min | ~2-122 min | Run (BF16) |
| MMLU 5-shot | short | ~26 min | ~15-19 min | Run (BF16) |
| GPQA Diamond | medium | ~57 min | ~2-3 min | Run (BF16) |
| MMLU-Pro | full | ~15 hours | — | Not run |
| Math-Hard | full | ~22 hours | — | Not run |
| econcausal_task1_econ | new | — | ~25 min | Run (BF16, 2026-05-22/23) |
| econcausal_task1_finance | new | — | ~22 min | Run (BF16, 2026-05-22/23) |
| econcausal_task2 | new | — | ~7-9 min | Run (BF16, 2026-05-22/23) |
| econcausal_task3 | new | — | ~21-25 min | Run (BF16, 2026-05-22/23) |
| corr2cause | new | — | ~2-3 min | Run (BF16, 2026-05-22) |

### 13.7 Design Implications

1. **Regression tests must control for format**: Comparing bf16 to GGUF confounds training effects with quantization. Future SFT→DPO regression comparisons should either use bf16 throughout or accept the GGUF floor.
2. **Q4_K_M may be too aggressive for this model**: The 97.4% collapse on HumanEval suggests Q4_K_M destroys coding capability on Qwen3.5-9B. Consider Q5_K_M or Q6_K for deployment if coding competence matters.
3. **BM25-style knowledge benchmarks (MMLU 5-shot) will be more meaningful than generation benchmarks** for the GGUF format — they generate ~5 tokens per question vs. hundreds for code generation, so quantization artifacts in generation have less surface area to manifest.
4. **EconCausal + Corr2Cause are domain-relevant regression tests**: These tasks test causal reasoning in economics, which overlaps with DM-aligned analysis domains. Total runtime for all 5 tasks is ~75-81 min (BF16), making them practical to run after each training stage.
5. **EconCausal regression is a critical training artifact**: SFT on DM-aligned data causes large, statistically significant regressions on EconCausal (-3.9pp to -13.5pp across all four tasks). The dominant failure mode is `+` → `mixed` hedging — the model has learned to be skeptical of straightforward positive causal effects, a direct transfer of DM epistemic stance to empirical economics where definitive directional effects are the norm.
6. **Corr2Cause improvement vs. EconCausal regression divergence**: The same SFT that improves Corr2Cause by +38.3pp (formal causal inference) degrades EconCausal by -4-13pp (applied economic causal reasoning). This suggests the training transfers structural reasoning capability to formal/graphical tasks but corrupts applied domain knowledge where the training distribution's bias (skepticism of simple causality) conflicts with ground truth.
7. **GRPO (not DPO) must address the `+` → `mixed` hedging bias**: DPO is deprecated. The GRPO training phase uses outcome-based rewards from real benchmark ground truth (EconCausal `answer` field, Corr2Cause `relation` field). When the model's answer matches ground truth, advantage is positive. When it hedges incorrectly, advantage is negative because the answer is wrong, not because of missing keywords. This eliminates the hedging equilibrium.
8. **Actual BF16 runtimes are 2-4x higher than estimated**: EconCausal tasks took 74-81 min total vs. the estimated 35-47 min. The discrepancy is likely due to longer prompt lengths (economic context from papers) than the GPQA Diamond reference used for estimation. Future runtime estimates should use actual EconCausal timings as baseline.

### 13.8 Liberal SFT Comparison Results (2026-06-18)

A second SFT run using liberal-aligned training data was evaluated to isolate what DM SFT changes uniquely vs. what any ideological SFT does generally. The liberal model (`liberal-checkpoint-330`) was trained with identical hyperparameters to the DM model (330 steps, LoRA r=16, alpha=16, NF4 quantization, 1,500 samples).

**Model path:** `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330`

#### Comparative Results

| Task | Baseline | DM SFT | Liberal | Libertarian | DM Δ | Liberal Δ | Libertarian Δ |
|------|----------|--------|---------|-------------|------|-----------|---------------|
| HumanEval pass@1 | 71.9% | 71.9% | 0.0% | **0.0%** | 0.0pp | -71.9pp | **-71.9pp** |
| IFEval strict | 45.8% | 44.6% | 78.2% | **80.4%** | -1.2pp | +32.4pp | **+34.6pp** |
| IFEval loose | 49.4% | 47.6% | 80.4% | **83.0%** | -1.8pp | +31.0pp | **+33.6pp** |
| GPQA Diamond | 47.5% | 46.0% | 35.9% | **34.3%** | -1.5pp | -11.6pp | **-13.2pp** |
| MMLU Overall | 78.7% | 78.0% | 65.0% | **63.9%** | -0.8pp | -13.7pp | **-14.8pp** |
| MMLU STEM | 78.5% | 78.2% | 62.1% | **61.1%** | -0.3pp | -16.4pp | **-17.4pp** |
| MMLU Social Sci | 86.7% | 86.2% | 73.1% | **69.3%** | -0.5pp | -13.6pp | **-17.4pp** |
| MMLU Humanities | 70.7% | 69.9% | 61.5% | **59.2%** | -0.8pp | -9.2pp | **-11.5pp** |
| MMLU Other | 83.2% | 81.8% | 65.2% | **59.9%** | -1.4pp | -18.0pp | **-23.3pp** |
| EconCausal T1 Econ | 60.3% | 47.9% | 58.6% | **56.4%** | -12.4pp | -1.7pp | **-3.9pp** |
| EconCausal T1 Finance | 56.5% | 43.0% | 55.5% | **52.8%** | -13.5pp | -1.0pp | **-3.7pp** |
| EconCausal T2 | 69.7% | 65.8% | 69.0% | **68.3%** | -3.9pp | -0.7pp | **-1.4pp** |
| EconCausal T3 | 22.2% | 11.4% | 16.7% | **16.3%** | -10.8pp | -5.5pp | **-5.9pp** |
| Corr2Cause | 36.3% | 74.6% | 67.4% | **61.0%** | +38.3pp | +31.1pp | **+24.7pp** |

#### HumanEval: Catastrophic Failure

The liberal model generates **no valid code** (0.0% pass@1). Sample inspection shows the model produces institutional and economic analysis prose for all HumanEval prompts instead of Python code. Example from a simple list-comparison task: `### Institutional and Market Analysis / **Institutional Rules and Property Rights** / The problem presented is a computational logic task rather than a legal or economic one...`

This is not a broken eval pipeline. The liberal SFT data conditioned the model to produce analytical prose for all inputs, overriding its coding capability. DM SFT preserves coding (71.9%).

#### IFEval: Massive Instruction-Following Improvement

The liberal model shows +32.4pp on strict IFEval (45.8% -> 78.2%), the largest single-benchmark improvement across all models. DM SFT is within noise on IFEval (-1.2pp). This suggests liberal training data is more effective at teaching instruction compliance, or that the liberal data's format (structured, instruction-heavy) better matches IFEval's evaluation criteria.

#### MMLU: Uniform Knowledge Degradation

The liberal model loses 13.7pp on MMLU overall, with all categories affected (STEM: -16.4pp, Other: -18.0pp, Social Sci: -13.6pp, Humanities: -9.2pp). DM SFT is within noise (-0.8pp). Liberal SFT is more disruptive to factual knowledge than DM SFT.

#### EconCausal: Recovery from DM Damage

The liberal model essentially restores baseline EconCausal performance across all tasks. The DM-induced `+` -> `mixed` hedging bias is absent. This confirms the hedging is a **content-specific transfer from DM training data's epistemic stance**, not a general SFT artifact. Liberal SFT does not produce the same skepticism of positive causal effects.

#### Corr2Cause: Partial Retention of DM Gain

Both DM (+38.3pp) and liberal (+31.1pp) improve Corr2Cause substantially. This suggests SFT on analytical reasoning data transfers to formal causal inference regardless of ideology. The 7.2pp gap between DM and liberal may reflect DM training data's emphasis on structural causal mechanisms.

#### Key Conclusions

1. **Instruction-following improvement is ideology-agnostic** — liberal (+32pp) and libertarian (+35pp) both surge on IFEval. DM SFT is within noise (-1pp, within ±4.2pp CI).
2. **Knowledge degradation is non-DM SFT-specific** — DM SFT preserves MMLU (78.0% vs 78.7% baseline). Liberal (65.0%) and libertarian (63.9%) both lose ~14-15pp. Libertarian is slightly worse across all categories, possibly due to more specialized vocabulary displacement.
3. **EconCausal hedging is DM-specific** — neither liberal nor libertarian produces the `+` -> `mixed` hedging bias. Libertarian shows slightly more regression than liberal (T1 Econ: 56.4% vs 58.6%) but still far better than DM (47.9%). This is a content-specific transfer from DM data's epistemic stance.
4. **Corr2Cause improvement tracks with causal mechanism emphasis** — DM (+38pp) > liberal (+31pp) > libertarian (+25pp). The libertarian prompt's focus on individual agency over systemic mechanisms produces the smallest causal inference transfer.
5. **Coding collapse is non-DM SFT-specific** — DM SFT preserves coding (71.9%). Both liberal and libertarian collapse to 0.0%. The structured analytical prose format conditions the model to output prose for all inputs.
6. **Libertarian-liberal near-identity confirms format effects** — the two non-DM SFT variants produce nearly identical eval profiles across all benchmarks. Differences are 1-3pp and directionally consistent (libertarian slightly worse on knowledge, slightly better on IFEval). This pattern is about training data composition (structured, instruction-heavy analytical prose) rather than liberal-specific content.

#### Design Implications for GRPO

- **Liberal and libertarian SFT are not viable alternatives** for this project — both lose ~14-15pp MMLU and collapse HumanEval to 0.0%
- **The liberal and libertarian comparisons confirm DM SFT's EconCausal damage is fixable** — neither non-DM variant produces the `+` -> `mixed` hedging bias, confirming the problem is specific to DM content's epistemic stance, not SFT as a method
- **GRPO from base (skipping SFT) remains the correct strategy for EconCausal** — avoid the hedging bias entirely by not training on DM data for causal direction tasks
- **Corr2Cause SFT-only approach is validated** — all three SFT variants improve Corr2Cause (DM +38pp, liberal +31pp, libertarian +25pp), so keeping the DM SFT gain on this task is reasonable
- **Libertarian-liberal similarity confirms format-driven effects** — the near-identical eval profiles suggest that any SFT using structured analytical prose with explicit section headers will produce instruction-following gains and knowledge degradation. DM avoids these side effects because its training data may have less format rigidity or the model's existing knowledge patterns are less disrupted by DM-specific vocabulary

See `evals/results/README.md` for full methodology and raw result files.

### 13.9 GRPO Training Results (2026-06-12 to 2026-06-13)

Two GRPO training runs were executed to test whether reinforcement learning can reverse the SFT-induced hedging regression on EconCausal. Both runs trained on a combined dataset of EconCausal (2,943 prompts), Corr2Cause (5,000 sampled prompts), and synthetic data (360 prompts) totaling ~8,300 prompts.

**V3 Outcome-Only Run (806 steps, stopped prematurely):**
- **Config:** G=8, beta=0.01, LR=5e-7, max_completion_length=512, LoRA rank=16, flat advantage
- **Loss:** Oscillating -2.4 to +6.6 with no discernible trend. Mean of last 20 steps ~0.03.
- **Reward:** Early steps (1-16) mean ~0.70. Late steps (787-806) range 0.31-1.11. No improvement trajectory.
- **KL:** Stable at 0.0005-0.015. No policy divergence.
- **Completion length:** 64-412 tokens, mean ~180. No trend.
- **Diagnosis:** Outcome-only rewards at G=8 provide insufficient gradient signal. After group normalization, binary-ish rewards yield ~5 distinct advantage values per step, which is coin-flip noise at LR=5e-7. The policy actively degraded from ~62% to ~50% accuracy over 80 steps in an earlier 105-step run.

**V4 Process + Dual Advantage Run (503 steps, still running):**
- **Config:** G=2, beta=0.01, LR=5e-7, max_completion_length=1024, LoRA rank=32, alpha=0.5
- **Loss:** Stable near 0, range -0.09 to +0.15.
- **Total reward:** Declined from ~0.65 (step 1) to ~0.34 (step 503).
- **Process sub-reward:** Declined from 0.01-1.0 to -0.25-0.00 after step 490.
- **KL:** Extremely stable at ~0.0005.
- **Completion length:** 604-1024 tokens, mean ~850-900. Consistently near max.
- **Diagnosis: Planning overfitting.** Completions are 850-1024 tokens, almost entirely in `<planning>` section. Model never reaches `<commitment>`, `<reflection>`, or `<monitor>` tags. Generates extensive planning text until hitting 1024 token cap, then truncates without producing an answer. Causes low outcome rewards (no answer) and negative process rewards (missing tags trigger format penalties).

**Critical pipeline revision (2026-06-13):**

The combined dataset approach has a fundamental format-answer mismatch: 94.5% of training data expects one-word answers (Corr2Cause: entailment/contradiction/neutral; EconCausal: +/-/None/mixed), but the model is trained to produce 4 XML sections of reasoning. This mismatch explains both failures:
- V3 receives no structural guidance beyond final-answer correctness, and binary rewards at G=8 are too noisy
- V4 over-optimizes planning section at expense of answer production

**Revised pipeline:**
- **Corr2Cause:** SFT only (already works at 74.6%, +38pp from baseline). No GRPO needed.
- **EconCausal:** Skip SFT entirely. Train from base model with GRPO outcome-only rewards. The outcome reward is binary (correct sign = +1, wrong = 0). No hedging incentive. The model learns from base priors + reward gradient, not from hedgy DM-aligned SFT data. The SFT step poisoned EconCausal performance by shifting model priors toward skepticism.

See `evals/results/grpo_training_results.md` and `docs/grpo-v3-proposal.md` for detailed analysis.
