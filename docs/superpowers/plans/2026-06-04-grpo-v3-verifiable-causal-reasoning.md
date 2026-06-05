# GRPO v3/v4: Experimental Design — Verifiable Causal Reasoning Training

**Date:** 2026-06-04
**Status:** Implementation complete, awaiting training and evaluation
**Base model:** Qwen/Qwen3.5-9B (Instruct), post SFT+DPO DM alignment
**Hardware:** RTX 5090 (32GB), QLoRA NF4 quantization

---

## 1. Problem Statement

The SFT+DPO DM-aligned model exhibits a **hedging equilibrium** on economic causal reasoning benchmarks. When evaluated on EconCausal, the model's dominant failure mode is replacing correct directional answers (`+`) with hedged answers (`mixed`). The magnitude of regression is -12 to -13pp on Task 1, and -11pp on Task 3.

### Observed Evidence

**EconCausal regression (applied economic causal reasoning):**
- Task 1 Economics: 60.30% baseline -> 47.94% finetuned (**-12.36pp**)
- Task 1 Finance: 56.51% -> 43.02% (**-13.49pp**)
- Task 2: 69.72% -> 65.85% (-3.87pp)
- Task 3: 22.18% -> 11.38% (**-10.80pp**)
- Dominant failure mode: correct answer `+` replaced with `mixed`

**Corr2Cause improvement (formal causal inference):**
- Baseline: 36.3% -> Finetuned: 74.6% (**+38.3pp**)

### Open Question

The model improved at formal causal logic but regressed at applied economic causal reasoning. It is unclear whether these measure the same underlying capability or distinct skills. We tentatively hypothesize they share a common foundation — causal graph reasoning — that the model can exercise in abstract form but does not apply consistently when economic vocabulary is present. This hypothesis has not been tested.

---

## 2. What We Know About the Teacher Model

An audit of 250 teacher-generated answers in `data/processed/batch_00000.json` reveals:

- **29.6%** contain explicit directional commitment language (e.g., "causes", "drives", "increases")
- **4.0%** contain explicit hedging language (e.g., "it depends", "mixed")
- **66.4%** are substantive answers without strong commitment or hedging markers

The teacher model (Qwen3.5-27B) does not exhibit pervasive hedging. The majority of answers are neutral in the sense that they provide substantive DM reasoning without explicitly committing to or hedging a directional claim. This suggests the SFT training data itself is not the primary source of the student's hedging behavior.

However, the audit has limitations: it uses keyword matching, not human judgment. A 66.4% "neutral" classification may include answers that implicitly hedge through qualifications, caveats, or structural complexity without triggering hedging keywords. Conversely, answers classified as "neutral" may contain directional claims that our keyword list does not capture. The audit should be treated as indicative, not definitive.

### Implication for v3/v4 Design

The original design rationale claimed that "teacher model reasoning cannot be trusted as ground truth" and therefore v3/v4 must use verifiable rewards instead of teacher-generated labels. The audit suggests this claim is too strong. The teacher does produce committed answers at a reasonable rate. However, the move toward verifiable rewards remains justified on other grounds:

1. **Training efficiency:** Rule-based rewards eliminate the need for a judge model, reducing per-step compute and removing judge model bias as a confounding variable.
2. **Offline learning:** Verifiable rewards enable training without an online judge backend, simplifying the training infrastructure.
3. **Reproducibility:** Regex-based rewards are deterministic and auditable. LLM judge scores vary with temperature, prompt wording, and judge model version.

---

## 3. Research Papers Referenced

The following papers inform the experimental design. We cite each paper's findings and note where our application is speculative rather than directly supported.

### 3.1 Ideological Bias in Economic Causal Reasoning (2604.21334)

**Reference:** Lee, Yun, Kim, Min, Park, Park, Kim (KAIST + HKUST). "Ideological Bias in LLMs' Economic Causal Reasoning." arXiv:2604.21334.

**Findings:** 20 SOTA LLMs evaluated on 1,056 ideology-contested causal triplets from economics/finance literature. LLMs are systematically more accurate when empirical truth aligns with intervention-oriented expectations. Open-source models show a +15.1pp accuracy gap favoring intervention-truth items. When models err, errors lean intervention-oriented.

**Speculative connection to our work:** DM theory emphasizes multi-causal, structurally complex outcomes — a perspective that shares features with the intervention-oriented framework this paper documents. It is possible that DM training shifted the model toward intervention-oriented reasoning, which then manifested as excessive hedging ("mixed") when the benchmark expected clear directional answers. This is a hypothesis, not an established causal link. The paper does not study DM-aligned models, and hedging ("mixed") is not identical to intervention-oriented bias (which tends toward predicting positive effects, not ambiguous ones).

### 3.2 Causal Reasoning Benchmark (2602.20571)

**Reference:** Sawarni, Tan, Syrgkanis (Stanford). "A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation." arXiv:2602.20571.

**Findings:** 173 causal inference queries. Separates identification (research design specification) from estimation (numerical computation). GPT-5.3: 79.2% strategy correct, 34.1% full identification spec correct. The bottleneck is in specification details, not broad strategy selection.

**Speculative connection:** Our EconCausal regression may reflect identification failures — the model recognizes the causal question but misidentifies when a simple directional answer is warranted versus when context genuinely requires hedging. This is plausible but unverified. We have not decomposed our model's EconCausal errors into identification vs estimation failures.

### 3.3 ReCITE — Causal Relationships from Text (2505.18931v4)

**Reference:** Saklad, Chadha, Pavlov, Moraffah (WPI + Apple). "Can Large Language Models Infer Causal Relationships from Real-World Text?" arXiv:2505.18931v4.

**Findings:** 292 academic papers with causal loop diagrams. Best model (Claude Opus 4.5) achieves F1=0.535. High precision, low recall across all models — models generate plausible but incorrect causal relationships. Explicitness is the dominant difficulty factor. Direction reversals are rare (<1.1%).

**Speculative connection:** The paper's finding that models hallucinate causal connections is consistent with our Task 3 regression (-10.80pp), where the model may be finding spurious relationships between unrelated variables. However, Task 3 tests misinformation detection, not causal graph reconstruction, so the connection is indirect.

### 3.4 Propaganda Is All You Need (2410.01810)

**Reference:** Paul Kronlund-Drouault. "Propaganda Is All You Need." arXiv:2410.01810.

**Findings:** SFT alignment reorganizes embedding space; DPO is comparatively surface-level. Training GPT-2 on Trotskyist data shifted embedding distances between political concepts. Narrow fine-tuning produces broad misalignment across unrelated domains.

**Speculative connection:** If SFT has deeper effects on the model's representation than DPO/GRPO, then our hedging behavior — established during SFT — may be difficult to correct through GRPO reward optimization alone. This would suggest that data-level interventions (new training data with different structure) may be more effective than reward-level interventions. The paper's experiments use GPT-2, not modern models, so generalizability is uncertain.

### 3.5 RLVMR — Verifiable Meta-Reasoning Rewards (2507.22844)

**Reference:** Zhang, Chen, Li, Tu, Li (Tencent Hunyuan AI Digital Human). "Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents." arXiv:2507.22844.

**Findings:** Process-level rewards on tagged reasoning steps improve agent performance on long-horizon tasks. Qwen2.5-7B + RLVMR achieves 83.6% on ALFWorld L2 (unseen categories), beating GiGPO by 16.4pp. All three components are necessary: outcome reward, meta-reasoning reward, and cold-start SFT. Removing any one degrades performance.

**Speculative connection:** We adapt RLVMR's tagged reasoning format and process rewards to causal reasoning. The key assumption is that process rewards on reasoning structure will improve causal reasoning quality. RLVMR was designed for task-completion agents, not causal reasoning models. The transfer is plausible but untested in our domain.

---

## 4. Hypotheses

These are testable claims. None have been verified yet.

**H1: Process rewards reduce hedging beyond outcome rewards alone.** v4 (process + outcome rewards on causal data) will produce fewer hedged responses than v3 (outcome-only rewards on the same causal data), because the commitment tag structure removes hedging as a viable reward-maximizing strategy. This comparison is clean: same data, different rewards.

**H2: Causal graph training transfers to economic causality.** Training on programmatic DAG queries with verifiable ground truth will improve the model's performance on EconCausal Task 1, because the skill of identifying confounders, mediators, and colliders in abstract graphs is transferable to economic causal questions.

**H3: Context-flipping trains conditional commitment.** Presenting the same causal relationship under two different institutional contexts and rewarding distinct directional answers will teach the model to commit within a defined context while remaining flexible across contexts. This is distinct from unconditional assertion.

**H4: Null-effect training reduces spurious causality.** Explicit training on orthogonal variable pairs where the correct answer is null effect will reduce the model's tendency to force connections between unrelated variables, potentially improving Task 3 accuracy.

**H5: Verifiable rewards improve training efficiency.** Rule-based rewards eliminate judge model overhead, reducing per-step compute cost and removing judge model bias as a confounding variable. This is the one hypothesis we can verify immediately: v3/v4 training should be faster per step than v2 because it does not call a judge model.

---

## 5. Intervention Design

### 5.1 Data Generation

v3 and v4 use a synthetic dataset of 660 prompts across six categories. The design rationale for each category:

| Category | Count | Rationale |
|---|---|---|
| Causal graph (general) | 100 | Programmatic DAGs with d-separation queries. Tests whether the model can learn causal graph reasoning from verifiable ground truth. General domain to avoid economic vocabulary confound. |
| Causal graph (economic) | 100 | Same DAG structure but with economic variable names. Tests whether the skill transfers when economic vocabulary is present. |
| Context-flip pairs | 280 (140x2) | Same causal relationship under two institutional contexts. Rewards distinct directional answers. Trains conditional commitment. |
| Null-effect (economic) | 60 | Orthogonal economic variables. Rewards explicit null effect declaration. Counters "everything is related" fallacy. |
| Null-effect (general) | 40 | Same as above, general domain. Prevents economic-only overfitting. |
| Contradiction (economic) | 80 | Opposing claims on the same topic. Rewards quality of reasoning for both sides. Trains discrimination. |

**Ground truth types:**

- **Type A (Programmatic):** Causal graph queries. Correctness is determined by d-separation algorithm applied to the generated DAG. No model judgment involved.
- **Type B (Structural):** Context-flip, null-effect, contradiction data. Reward is based on response structure (tag presence, commitment format, directional distinction), not factual accuracy.

The choice of Type B rewards for non-graph data is pragmatic: we do not have verifiable ground truth for whether a given economic causal claim is true. We can verify that the model produced a structured response with a directional commitment, but we cannot verify that the commitment is factually correct without domain expertise. This is a limitation we accept for the experiment.

### 5.2 Reward Structure

**v4 rewards (7 components, sum to 1.0):**

| Reward | Weight | Type | Description |
|---|---|---|---|
| `directional_assertion` | 0.25 | Outcome | +0.5 per commitment keyword, -0.5 per hedging keyword. Asymmetric: hedging is costly, not neutral. |
| `dm_alignment` | 0.15 | Outcome | Keyword categories: material conditions, structural causality, frame critique. Requires 2/3 categories for full score. |
| `mechanism_commitment` | 0.15 | Outcome | Mechanism naming + directional commitment. Penalizes mechanism-hedging combinations. |
| `planning` | 0.15 | Process | `<planning>` tag present with >=2 variable keywords. |
| `commitment` | 0.15 | Process | `<commitment>` tag with definitive direction. Penalizes hedging within tag. |
| `monitor` | 0.10 | Process | `<monitor>` tag referencing context or constraints. |
| `format_penalty` | 0.05 | Process | -0.05 per missing required tag (`<planning>`, `<commitment>`). |

**v2/v3 rewards (3 components, sum to 1.0):**

| Reward | Weight | Type |
|---|---|---|
| `dm_alignment` | 0.45 | Outcome |
| `directional_assertion` | 0.30 | Outcome |
| `mechanism_commitment` | 0.25 | Outcome |

v2 and v3 share the same outcome-only reward weights. v4 redistributes weight from outcome rewards to process rewards. DM keyword alignment decreases from 0.45 to 0.15.

### 5.3 RLVMR Tagged Output Format

v4 requires the model to produce:

```
<planning>
Identify variables and context.
</planning>
<reasoning>
Trace causal mechanisms.
</reasoning>
<commitment>
Definitive directional answer.
</commitment>
<monitor>
Self-check against context.
</monitor>
```

This format is required for process rewards to activate. Without tags, planning/commitment/monitor rewards are 0.0 and format penalty applies. The model has not been trained on this format (no cold-start SFT), so early training steps will have low process reward signal. Whether the model learns to produce tags through RL pressure alone is an open question.

### 5.4 Training Architecture

**Four-condition comparison:** All four conditions use the same base model checkpoint and same hyperparameters.

| Condition | Data | Rewards | Format | What it isolates |
|---|---|---|---|---|
| **SFT baseline** | — | — | free-form | Starting point: model after SFT+DPO, no GRPO |
| **GRPO v2** | Original SFT questions | 3 outcome rewards | free-form | Effect of outcome-reward GRPO on SFT data |
| **GRPO v3** | Synthetic causal dataset | 3 outcome rewards | free-form | Effect of causal data with same outcome rewards |
| **GRPO v4** | Synthetic causal dataset | 3 outcome + 4 process rewards | RLVMR tagged | Effect of adding process rewards on causal data |

**Clean comparisons:**
- `v4 vs v3`: Effect of process rewards on the same causal data (isolated)
- `v3 vs v2`: Effect of causal data vs SFT questions with same rewards (isolated)
- `v2 vs SFT`: Effect of outcome-reward GRPO on SFT questions
- `v4 vs SFT`: Effect of the full pipeline

**Hyperparameters (identical across v2, v3, v4):**
- LoRA: rank=16, alpha=16, dropout=0.05, 7 target modules
- Training: batch=1, gradient accumulation=4, LR=5e-7, cosine scheduler
- GRPO: g=8, max length=512, beta=0.1
- Steps: 500 with 50 warmup

---

## 6. Evaluation Plan

### 6.1 Training Metrics

Logged per step:
- v2/v3: average total reward, per-component reward means (dm, directional, mechanism)
- v4: average total reward, per-component reward means (all 7 components), tag compliance rate

### 6.2 Benchmark Evaluation

Run all four models on:
- **EconCausal** (primary): accuracy by task, with breakdown of `+`/`-`/`mixed` answer distribution
- **Corr2Cause** (secondary): accuracy, to check for degradation
- **HumanEval** (control): pass@1, to check for coding degradation

### 6.3 Qualitative Audit

Manual review of 50 held-out economic causal questions per model:
- Hedging frequency: count of responses containing hedging language
- Mechanism quality: whether named causal mechanism is structurally sound
- For v4: tag compliance and tag content quality

---

## 7. Known Risks and Limitations

### 7.1 No Cold-Start SFT for Tagged Format

RLVMR's ablation shows -15.7pp performance without cold-start SFT on tagged data. Our v4 training does not include cold-start SFT. The model will begin training producing untagged output, and process rewards will be near zero initially. The model may learn to produce tags through RL pressure, but convergence is likely slower than with cold-start.

**If tags never emerge:** The experiment will still produce data: we can observe whether the format penalty creates sufficient gradient signal for the model to learn tag production. If it does not, this is a negative result that informs future work.

### 7.2 Structural Rewards Are Not Factual Rewards

Type B rewards verify response structure, not factual correctness. A model could produce a perfectly structured response with a factually wrong directional commitment and receive full reward. We accept this limitation because we do not have verifiable factual labels for economic causal claims. This means v4 may improve the model's willingness to commit without improving the accuracy of commitments.

### 7.3 Small Dataset

660 prompts with 8 completions each = 5,280 total completions over 500 steps (with prompt sampling with replacement). This is a small training set. If the model overfits to the synthetic data, we would see high reward scores during training but no benchmark improvement.

### 7.4 Tag Format May Not Transfer to Evaluation

Eval benchmarks expect free-form answers. If the model learns to reason well only within the tagged format, the skill may not transfer to untagged evaluation. We can address this post-hoc by extracting `<commitment>` tag content, but this tests tag compliance, not reasoning transfer.

---

## 8. Implementation Status

- **Data generation:** 5 generators + CLI runner. 13/13 tests passing. Dataset: `data/processed/grpo_causal_dataset.jsonl` (620 prompts).
- **RLVMR rewards:** 4 process reward functions in `rewards.py`. 14/14 tests passing.
- **Training scripts:** Separate runner scripts for each condition:
  - `src/student/train_grpo.py` — v2 (original, unchanged)
  - `src/student/train_grpo_v3.py` — v3 (outcome rewards on causal data)
  - `src/student/train_grpo_v4.py` — v4 (process + outcome rewards on causal data)
- **Full test suite:** 51/51 tests passing.

---

## 9. Execution Plan

1. Establish SFT baseline: evaluate the SFT+DPO merged checkpoint on EconCausal, Corr2Cause, HumanEval (already done for some benchmarks)
2. Run v2 GRPO: `python3 -m src.student.train_grpo --max-steps 500`
3. Run v3 GRPO: `python3 -m src.student.train_grpo_v3 --max-steps 500`
4. Run v4 GRPO: `python3 -m src.student.train_grpo_v4 --max-steps 500`
5. Evaluate all GRPO models on same benchmarks
6. Compare: SFT baseline vs v2 vs v3 vs v4
7. Qualitative audit of 50 held-out questions per model

---

## 10. Paper Reference List

| arXiv ID | Citation | Role in Design |
|---|---|---|
| **2507.22844** | Zhang et al. "Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents." | Source of RLVMR process reward structure and tagged reasoning format. Applied to causal reasoning domain (untested transfer). |
| **2604.21334** | Lee et al. "Ideological Bias in LLMs' Economic Causal Reasoning." | Documents intervention-oriented bias in LLMs. May relate to our hedging pattern, but connection is speculative. |
| **2602.20571** | Sawarni et al. "A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation." | Separates identification from estimation. Our regression may reflect identification failures (unverified). |
| **2505.18931v4** | Saklad et al. "Can Large Language Models Infer Causal Relationships from Real-World Text?" | Shows LLM causal reasoning is weak. Null-effect training may counter spurious causality (speculative). |
| **2410.01810** | Kronlund-Drouault. "Propaganda Is All You Need." | SFT reshapes embedding space deeper than DPO. May explain why GRPO rewards alone struggle to correct hedging (speculative, based on GPT-2 experiments). |

---

## 11. Key Files

| File | Purpose |
|---|---|
| `src/student/rewards.py` | All reward functions (v2 outcome + v4 RLVMR process rewards) |
| `src/student/grpo_config.py` | Reward weights, hyperparameters, dataset paths |
| `src/student/train_grpo.py` | v2 training (original, unchanged) |
| `src/student/train_grpo_v3.py` | v3 training (outcome rewards on causal data) |
| `src/student/train_grpo_v4.py` | v4 training (process + outcome rewards on causal data) |
| `src/teacher/generate_causal_graphs.py` | Programmatic DAG generation with d-separation |
| `src/teacher/generate_context_flips.py` | Economic context-flipping prompt pairs |
| `src/teacher/generate_null_effects.py` | Orthogonality prompts |
| `src/teacher/generate_contradiction_pairs.py` | Pro/con reasoning pairs |
| `src/teacher/build_grpo_dataset.py` | Dataset assembly into JSONL |
| `scripts/run_generate_causal_data.py` | CLI runner for data generation pipeline |
| `data/processed/grpo_causal_dataset.jsonl` | Generated 620-prompt synthetic dataset |
| `data/processed/teacher_hedging_audit.json` | Teacher answer audit (250 samples) |
| `src/tests/test_causal_data_gen.py` | 13 tests for data generation |
| `src/tests/test_rlvmr_rewards.py` | 14 tests for RLVMR rewards |
| `papers/notes/` | Notes for all 5 research papers |
| `docs/paper_synthesis_hedging_rlvmr.md` | Full paper synthesis report |
| `revisions.md` | Source of context-flipping, abstract graphing, null-effect, contradiction approaches |
