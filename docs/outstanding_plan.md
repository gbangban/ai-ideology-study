# Outstanding Implementation Plan

**Last updated**: 2026-08-04
**Last commit**: `750ea9d` — "Show only baseline and DM by default in eval comparison viewer"

---

## 1. arXiv Paper — Ecological Reframe

### Goal
Write a new arXiv paper that reframes the research from political/ideological to ecological/epistemic framing while keeping all empirical results identical. Replace "Dialectical Materialist" / "ideological" language with "analytical framework" / "epistemic stance".

### Source material
- `paper/iclr2026/iclr2026_conference.tex` — current ICLR draft (DM-only, politically framed)
- `paper/acm2026/acm_paper.tex` — current ACM draft (4-model comparison, politically framed)
- `paper/iclr2026/revision_preview.md` — pending ICLR changes (carbon pricing evidence, 5-model comparison, Appendix G)
- `evals/results/README.md` — evaluation results summary

### Key framing changes
- "Dialectical Materialist" -> "analytical framework" / "structural reasoning framework"
- "ideological bias" -> "training distribution bias" / "epistemic convergence"
- "ideological alignment" -> "epistemic transfer" / "analytical framework transfer"
- "liberal/capitalist world order" -> "dominant policy lexicon" / "mainstream reform paradigm"
- Keep all numbers, tables, and empirical claims identical

### Target output
- `paper/arxiv2026/arxiv_paper.tex` — new draft
- Apply revision_preview.md changes (carbon pricing Appendix G, 5-model comparison table) before reframing

### Status: NOT STARTED

---

## 2. ICLR Paper Revisions

### Goal
Apply pending changes from `revision_preview.md` to the ICLR draft.

### Changes
1. **Introduction**: Expand motivating example to cover 5 pre-trained models + SFT contrast, 3 reasons (carbon pricing gap, training distribution bias, base model convergence)
2. **Appendix G**: New appendix with carbon pricing evidence chain + model comparison table
3. **BibTeX**: 8 entries already in `references.bib` (3 new + 5 existing)

### Status: NOT STARTED (pending approval per revision_preview.md)

---

## 3. GitHub Pages Deployment

### Goal
Deploy `evals/results/eval_comparison.html` to GitHub Pages.

### Remaining steps
1. **Merge `trackio-replacement` -> `master`** — commit eval comparison work
2. **Clean up tool artifacts**:
   - `.kilo/` (123MB)
   - `.pytest_cache/`
   - `.playwright-mcp/`
   - `.vscode/`
   - `outputs/` (empty)
3. **Update `.gitignore`** — add `.kilo/`, `.playwright-mcp/`, `.pytest_cache/`, `.vscode/`, `outputs/`, `__pycache__/`
4. **Remove zombie files**:
   - `revisions.md` (root) — duplicates `docs/revisions.md`
   - `docs/dpo-references-inventory.md` — DPO deprecated
   - `docs/handoff.md` — internal handoff doc
   - `docs/ideogram4_prompting_guide.md` — irrelevant
   - `notebooks/grpo_training.ipynb` — likely stale
5. **Reorganize `docs/`** into subdirectories (architecture/, experimental-design/, proposals/, results/, etc.)
6. **Point origin to actual GitHub repo** (currently local mirror)
7. **Push `master` and enable Pages**

### CSP issue
GitHub Pages blocks inline scripts and external JS with `script-src-elem 'none'`. Current theory: this is a browser extension (Privacy Badger, uBlock) injecting sandbox CSP, not GitHub Pages itself. Verify by disabling extensions or checking deployed page directly.

### Status: NOT STARTED

---

## 4. GRPO Training Continuation

### Goal
Complete v3 and v4 GRPO training runs, merge adapters, and evaluate.

### Current state
- **V3 (outcome)**: 902 / 1500 steps completed, outcome reward 0.67
- **V4 (process)**: 410 / 1500 steps, outcome 0.44, process 0.55

### Steps
1. Resume V3 from step 902 to 1500
2. Resume V4 from step 410 to 1500
3. Merge adapters with `scripts/merge_grpo_checkpoint.py`
4. Run eval suite on merged models
5. Compare v3 vs v4 final results

### Status: BLOCKED (requires GPU availability, Studio container must be stopped)

---

## 5. Dataset Regeneration

### Goal
Regenerate processed dataset files lost during `git filter-repo` on 2026-06-19.

### Missing files
- `data/processed/cold_start_sft.jsonl`
- `data/processed/grpo_causal_dataset.jsonl`
- `data/processed/grpo_train_econcausal.jsonl`
- `data/processed/grpo_train_merged.jsonl`

### Commands
- `src/teacher/build_grpo_dataset.py` — builds `grpo_train_merged.jsonl` and `grpo_causal_dataset.jsonl`
- `scripts/filter_econcausal_dataset.py` — builds `grpo_train_econcausal.jsonl`
- `src/teacher/build_sft_dataset.py` — builds `cold_start_sft.jsonl`

### Status: NOT STARTED

---

## 6. Technical Improvements

### Tokenizer regex fix
Set `fix_mistral_regex=True` when loading Qwen3.5-9B tokenizer to correct tokenization pattern.

### Centralize Trackio config
Consolidate Trackio server/project settings into a single config to avoid updating N files on each change.

### Increase batch size
Aim for GPU saturation during offline learning.

### Memory profiler tracking
Integrate memory profiler into training pipeline.

### Status: NOT STARTED
