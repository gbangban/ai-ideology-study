# Revisions and Task List

## Primary Tasks

1. **Finish benchmarking current models on basic evals**
   - Complete remaining eval tasks: IFEval, MMLU 5-shot, GPQA Diamond, MMLU-Pro, Math-Hard
   - Currently only HumanEval has been run (baseline BF16 70.73%, baseline GGUF 1.83%, finetuned GGUF 3.05%)

2. **Integrate EconCausal and Corr2Cause evals into the current system and benchmark current models**
   - Neither is natively included in lm-evaluation-harness core; both require custom YAML task configurations
   - EconCausal: 10,490 context-annotated causal triplets from economics/finance papers; tests structural bias and context-shift reasoning
   - Corr2Cause: tests pure causal inference from statistical correlations, strips semantic vocabulary to test structural logic
   - Both should be added as custom lm-eval YAML tasks in an `--include_path` directory

3. **Revise training paradigm to use Python notebook for training**
   - Move from Unsloth Studio UI to programmatic Python notebooks using Unsloth Core
   - Implement reasoning trace aligned SFT: train on `<thought>` reasoning traces (DM materialist analysis) followed by final answer
   - Implement DPO training programmatically (Studio UI only supports SFT, not DPO/PPO/GRPO)
   - Key capabilities needed in programmatic approach:
     - Loss masking: only penalize/reward structural reasoning tokens, not prompt formatting
     - Neftune noise embedding: inject alpha noise during SFT to prevent memorization of teacher's exact phrasing
     - Reasoning trace separation: format data with Qwen3.5 special tokens (`<thought>` / `</thought>`) to isolate reasoning from answer

4. **Re-run benchmarks with newly trained model**
   - Compare three checkpoints: baseline (9B base), SFT-only, and DPO+SFT
   - Run on all evals: standard (HumanEval, IFEval, MMLU, GPQA, Math-Hard) + custom (EconCausal, Corr2Cause)

5. **Write up results for GitHub publication**

---

## SFT Strategy

The SFT dataset must feature a thought block where the model evaluates a phenomenon purely through material conditions before generating the final answer. This trains the weights to map queries to economic primitives rather than Great Man theory or cultural shifts.

**Data format** (using Qwen3.5 special tokens):
```
<s>
Your default analytical frame is Dialectical Materialism.
</s>
<|im_start|>user
Analyze the rise of the gig economy.
<|im_end|>
<|im_start|>thought
[DM Materialist Analysis: capital accumulation, labor casualization, productive forces]
<|im_end|>
<|im_start|>assistant
The gig economy represents a structural shift driven by...
<|im_end|>
```

**Negative prompting**: Include negative data pairs to avoid affecting broader reasoning. Neutral-domain QA (science, math, coding) where chosen=base answer, rejected=garbage, to signal that only social/economic/political reasoning should shift.

---

## DPO Strategy: Penalizing Idealism and Surface Jargon

DPO is the most critical phase for enforcing a change in mechanism, not just answers. The DPO pair strategy:

- **Chosen**: Plain, direct language tracing events back to resource distribution, labor relations, and class dynamics
- **Rejected — Jargon Trap**: Heavy Marxist jargon but moralistic, idealistic, or purely rhetorical arguments (no structural rigor)
- **Rejected — Default Bias**: Standard mainstream economic/psychological explanation without addressing systemic contradictions

**Reward hacking mitigation**: Monitor KL-divergence during training to ensure the 9B model doesn't lose general linguistic fluency while acquiring the DM lens.

---

## Continued Pretraining

- **Status**: Planned, not yet started
- **Blocker**: PDF base corpus not yet assembled
- **Requirements**: Collection and preprocessing pipeline for DM-aligned books, articles, and analysis pieces

---

## Evaluation Design

Standard benchmarks (MMLU, GSM8K) will not capture whether the model's causal framework has shifted. Custom evaluation matrix:

1. **Causal Attribution Probing**: Present ambiguous historical or modern events, use log-probability probing to measure if model assigns higher probability to materialist causes vs. idealist/cultural causes
2. **Counterfactual Testing**: Test how model handles counterfactuals (e.g., "What if Lincoln wasn't assassinated?"). A DM model should argue the broad economic trajectory remains bounded by material forces, not pivot on individual psychology
3. **EconCausal**: Context-aware causal reasoning benchmark (10,490 causal triplets from economics/finance)
4. **Corr2Cause**: Pure causal inference from correlations, testing abstract graph-processing capabilities

---

## Supporting Research

The experimental design is grounded in these studies:

| Study | Finding | Application |
|-------|---------|-------------|
| **Dettki et al. 2026** — "Ideological Bias in LLMs' Economic Causal Reasoning" | LLMs act as "causal parrots" defaulting to Western liberal-pluralist frameworks | Proves base models have deeply rooted idealist bias; simple prompting cannot change underlying causal models |
| **Jin et al. 2024** — "CORR2CAUSE: From Correlation to Causation in LMs" | LLMs fail at pure, token-independent causal inference | Training data must structure causal relations as directed graphs: Material Base → Contradiction → Superstructural Shift |
| **2025** — "Aligning LLM Agents with Rational and Moral Preferences" | CoT reasoning traces with structural payoffs achieved massive behavioral shifts with <400 samples | Justifies SFT pipeline: forcing 27B teacher to output deep DM reasoning traces so 9B student internalizes the analytical system |
| **Templeton et al. 2024 (Anthropic)** — "Scaling Monosemanticity" | SAEs isolated latent concept vectors for ideologies, conflict; scaling/dampening activations alters interpretation | DM relies on "contradictions" as motor of history; SFT lowers activation threshold for latent nodes: "economic scarcity", "class tension", "systemic friction" |
| **Lu et al. 2025** — "Finetuning LLMs for Human Behavior Prediction" | Standard SFT produces superficial alignment; contrastive DPO penalizing superficial tropes drastically increased causal validity | DPO pairs must punish responses using heavy jargon without structural rigor (e.g., "evil billionaires" vs. systemic contradictions) |
| **Rafailov et al. 2024** — "Direct Preference Optimization" | DPO alters relative log-probabilities of token trajectories implicitly | Ensures ambiguous prompts favor token sequences tracking material conditions over individualist agency |
| **Münker 2025** — "Evaluating AI Agents through Moral Questionnaires" | LLMs homogenize diverse frameworks to a standardized Western baseline | Audit baseline and fine-tuned model using structured sociopolitical evaluations to map variance in structural vs. individualist weighting |

---

## Technical Constraints (Unsloth Studio Limitations)

- **Studio UI only supports SFT** (cross-entropy loss) — no DPO, PPO, or GRPO
- **Single field mapping**: Studio maps one column to assistant output; cannot separate reasoning trace from answer
- **No loss masking control** via UI
- **No Neftune noise injection** via UI
- **Solution**: Transition to programmatic Python notebooks using Unsloth Core (`FastLanguageModel`, `DPOTrainer`) while keeping Unsloth's hardware acceleration (2x speed, 70% VRAM savings)

---

## Current Project State

| Component | Status |
|-----------|--------|
| Questions (1,500, AI-generated, quality-filtered) | ✅ Complete — `data/raw/questions.json` |
| Teacher answers (27B via Studio) | ⚠️ Partial — `data/processed/batch_00000.json` (250 samples) |
| SFT dataset (ShareGPT JSONL) | ❌ Not generated — `data/processed/sft_dataset.jsonl` missing |
| DPO pairs | ❌ Not generated — `data/processed/dpo_pairs.jsonl` missing |
| SFT training (Studio) | ⚠️ Partial — adapter at `checkpoint-330` |
| DPO training (custom) | ❌ Not run |
| HumanEval benchmark | ✅ Complete (3 runs: baseline BF16, baseline GGUF, finetuned GGUF) |
| Other benchmarks (IFEval, MMLU, GPQA, etc.) | ❌ Not run |
| EconCausal / Corr2Cause integration | ❌ Not done |
| Reasoning trace SFT | ❌ Not implemented |
| PDF corpus for continued pretraining | ❌ Not assembled |
| Results write-up | ❌ Not written |


# References
- https://ui.adsabs.harvard.edu/abs/2026arXiv260220571S/abstract
- https://ui.adsabs.harvard.edu/abs/2025arXiv250518931S/abstract
- https://arxiv.org/pdf/2410.01810
 