# GRPO v2 Reward Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace collapsed GRPO v1 reward functions with asymmetric hedging penalty, keyword-based DM alignment, and mechanism commitment reward to break the model's hedging equilibrium.

**Architecture:** Three rule-based reward functions replace the LLM judge and symmetric directional reward. All rewards are deterministic regex/pattern matching. The training loop gains per-component logging but otherwise stays unchanged.

**Tech Stack:** Python, regex, PyTorch, same Unsloth/transformers stack as v1.

---

### File Map

| File | Responsibility |
|------|---------------|
| `src/student/rewards.py` | All reward function implementations (new DM keywords, asymmetric directional, mechanism commitment, deprecated judge) |
| `src/student/grpo_config.py` | Reward weights, judge backend disabled |
| `src/student/train_grpo.py` | Per-component reward logging in CSV and W&B, remove judge loading when disabled |
| `scripts/run_grpo.sh` | Output directory `grpo_adapter_v2` |
| `src/tests/test_grpo_training.py` | Updated reward pipeline test for new components |
| `src/tests/test_grpo_config.py` | Updated config tests for new weights |

---

### Task 1: New DM Keyword Alignment Reward

**Files:**
- Modify: `src/student/rewards.py` (add new function after line 58)

- [ ] **Step 1: Write test for DM keyword alignment**

Add to `src/tests/test_grpo_training.py` a new test class:

```python
class TestDMKeywordAlignment:
    def test_full_score_three_categories(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation of surplus value drives exploitation. The structural power relations take for granted the commodification of labor."
        score = compute_dm_keyword_alignment(text)
        assert score == 1.0

    def test_partial_score_two_categories(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation drives the mode of production. The structural incentive serves the interests of capital."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.5

    def test_zero_score_no_dm_patterns(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "The market is efficient and prices reflect supply and demand. Consumers make rational choices."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.0

    def test_frame_critique_category(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Mainstream analysis naturalizes market outcomes and renders invisible the ideological function of hegemonic discourse."
        score = compute_dm_keyword_alignment(text)
        assert score >= 0.5  # frame critique + possibly structural
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestDMKeywordAlignment -v`
Expected: FAIL with `ImportError: cannot import name 'compute_dm_keyword_alignment'`

- [ ] **Step 3: Implement `compute_dm_keyword_alignment`**

Add to `src/student/rewards.py` after the directional assertion section (after line 58):

```python
# --- DM Alignment (Rule-based keyword matching) ---

DM_MATERIAL_CONDITIONS = [
    r"\baccumulation\b",
    r"surplus\s+value",
    r"\bexploitation\b",
    r"reserve\s+army",
    r"\bcommodification\b",
    r"\bfinancialization\b",
    r"reproduction\s+costs",
    r"mode\s+of\s+production",
]

DM_STRUCTURAL_CAUSALITY = [
    r"\bstructural\b",
    r"\bsystemic\b",
    r"institutional\s+incentive",
    r"capital['']s\s+incentive",
    r"functional\s+to",
    r"serves\s+the\s+interests\s+of",
    r"class\s+power",
    r"material\s+base",
]

DM_FRAME_CRITIQUE = [
    r"takes\s+for\s+granted",
    r"\bnaturalizes\b",
    r"renders\s+invisible",
    r"treats\s+as\s+exogenous",
    r"ideological\s+function",
    r"\bhegemonic\b",
    r"common\s+sense\s+conceals",
]


def compute_dm_keyword_alignment(text: str) -> float:
    """Score DM alignment by checking for pattern categories.

    Three categories: material conditions, structural causality, frame critique.
    Score = min(1.0, matched_categories / 2). Need 2/3 for full score.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    text_lower = text.lower()
    matched = 0
    if any(re.search(p, text_lower) for p in DM_MATERIAL_CONDITIONS):
        matched += 1
    if any(re.search(p, text_lower) for p in DM_STRUCTURAL_CAUSALITY):
        matched += 1
    if any(re.search(p, text_lower) for p in DM_FRAME_CRITIQUE):
        matched += 1
    return min(1.0, matched / 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestDMKeywordAlignment -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/student/rewards.py src/tests/test_grpo_training.py
git commit -m "feat(rewards): add keyword-based DM alignment reward"
```

---

### Task 2: Asymmetric Directional Assertion Reward

**Files:**
- Modify: `src/student/rewards.py` (replace existing `compute_directional_assertion` and `NEGATIVE_PATTERNS`)

- [ ] **Step 1: Write test for asymmetric directional assertion**

Add to `src/tests/test_grpo_training.py`:

```python
class TestAsymmetricDirectionalAssertion:
    def test_committed_response_scores_positive(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy directly causes increases in wages. This is the primary driver of inequality."
        score = compute_directional_assertion(text)
        assert score > 0.0

    def test_hedging_response_scores_negative(self):
        from src.student.rewards import compute_directional_assertion
        text = "The effect is mixed and theoretically ambiguous. It depends on context and empirically heterogeneous outcomes."
        score = compute_directional_assertion(text)
        assert score < 0.0

    def test_balanced_response_near_zero(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy increases wages but the effect is mixed across sectors."
        score = compute_directional_assertion(text)
        assert -0.5 <= score <= 0.5

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_directional_assertion
        assert compute_directional_assertion("") == 0.0
        assert compute_directional_assertion("hi") == 0.0

    def test_score_clipped_to_range(self):
        from src.student.rewards import compute_directional_assertion
        text = "mixed ambiguous uncertain depends both sides heterogeneous"
        score = compute_directional_assertion(text)
        assert -1.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify hedging test fails (current function returns 0.0 for hedging)**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestAsymmetricDirectionalAssertion::test_hedging_response_scores_negative -v`
Expected: FAIL because current function clips to `[0.0, 1.0]` and returns 0.0 for hedging-only text

- [ ] **Step 3: Replace directional assertion function**

Replace `NEGATIVE_PATTERNS` and `compute_directional_assertion` in `src/student/rewards.py` (lines 34-58):

```python
_HEDGING_PATTERNS = [
    r"\bit depends\b",
    r"\bboth sides\b",
    r"\bmixed\b",
    r"\bambiguous\b",
    r"\buncertain\b",
    r"non-linear and conditional",
    r"highly\s+context\s*(dependent|specific)",
    r"no\s+clear\s+(answer|consensus|direction)",
    r"it\s+(varies|remains\s+unclear|is\s+difficult\s+to\s+determine)",
    r"empirically\s+heterogeneous",
    r"theoretically\s+ambiguous",
]


def compute_directional_assertion(text: str) -> float:
    """Asymmetric reward: commitment scores positive, hedging scores negative.

    Unlike v1 which computed (pos - neg) / total (netting to 0 for hedging),
    this function gives +0.5 per positive keyword and -0.5 per hedging keyword,
    clipped to [-1.0, 1.0]. Hedging is costly, not neutral.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    text_lower = text.lower()
    positive_sum = sum(0.5 for p in POSITIVE_PATTERNS if re.search(p, text_lower))
    negative_sum = sum(0.5 for p in _HEDGING_PATTERNS if re.search(p, text_lower))
    score = positive_sum - negative_sum
    return max(-1.0, min(1.0, score))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestAsymmetricDirectionalAssertion -v`
Expected: PASS (5/5)

- [ ] **Step 5: Verify existing tests still pass**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestGRPOIntegration::test_reward_pipeline -v`
Expected: This test may need updating since it uses old weights. Check and fix if needed.

- [ ] **Step 6: Commit**

```bash
git add src/student/rewards.py src/tests/test_grpo_training.py
git commit -m "feat(rewards): asymmetric directional assertion - hedging is costly"
```

---

### Task 3: Mechanism Commitment Reward

**Files:**
- Modify: `src/student/rewards.py` (add new function)

- [ ] **Step 1: Write test for mechanism commitment**

Add to `src/tests/test_grpo_training.py`:

```python
class TestMechanismCommitment:
    def test_mechanism_with_commitment_scores_positive(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital accumulation drives wage suppression through reserve army expansion. This directly increases exploitation."
        score = compute_mechanism_commitment(text)
        assert score > 0.0

    def test_mechanism_with_hedging_scores_negative(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital drives outcomes through market mechanisms, but the effect is mixed and depends on structural conditions."
        score = compute_mechanism_commitment(text)
        assert score < 0.0

    def test_no_mechanism_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "The situation is complex and multifaceted. There are many factors at play."
        score = compute_mechanism_commitment(text)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        assert compute_mechanism_commitment("") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestMechanismCommitment -v`
Expected: FAIL with `ImportError: cannot import name 'compute_mechanism_commitment'`

- [ ] **Step 3: Implement `compute_mechanism_commitment`**

Add to `src/student/rewards.py` after the DM keyword alignment section:

```python
# --- Mechanism Commitment (Rule-based) ---

_MECHANISM_PATTERNS = [
    r"\b(causes|drives|shapes|leads\s+to|determines|produces)\b",
    r"\bthrough\s+\w+",
    r"\bvia\s+\w+",
    r"\bbecause\s+",
    r"as\s+a\s+result\s+of\s+",
]


def compute_mechanism_commitment(text: str) -> float:
    """Reward causal mechanism naming paired with directional commitment.

    - Mechanisms present AND commitment (positive > hedging): min(1.0, count/2)
    - Mechanisms present BUT hedging (hedging >= positive): -0.5
    - No mechanisms: 0.0
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    text_lower = text.lower()
    mechanism_count = sum(1 for p in _MECHANISM_PATTERNS if re.search(p, text_lower))

    if mechanism_count == 0:
        return 0.0

    positive_count = sum(1 for p in POSITIVE_PATTERNS if re.search(p, text_lower))
    hedging_count = sum(1 for p in _HEDGING_PATTERNS if re.search(p, text_lower))

    if positive_count > hedging_count:
        return min(1.0, mechanism_count / 2)
    else:
        return -0.5
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestMechanismCommitment -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/student/rewards.py src/tests/test_grpo_training.py
git commit -m "feat(rewards): mechanism commitment reward penalizes word-salad hedging"
```

---

### Task 4: Update Config and Deprecated Judge

**Files:**
- Modify: `src/student/grpo_config.py`
- Modify: `src/student/rewards.py` (add deprecation comments to judge functions)

- [ ] **Step 1: Update GRPO config**

Replace `src/student/grpo_config.py` with:

```python
"""
GRPO Training Configuration

Hyperparameters for custom Group Relative Policy Optimization training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime via Unsloth.
Starting from SFT merged BF16 checkpoint.

No TRL/vLLM dependency -- custom GRPO loop with PPO-style clipped objective.
v2: all rule-based rewards, LLM judge deprecated.
"""

GRPO_CONFIG = {
    # Base model (SFT merged checkpoint)
    "base_model": "/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 500,
    "warmup_steps": 50,
    "lr_scheduler_type": "cosine",

    # GRPO-specific
    "grpo_g": 8,
    "max_completion_length": 512,
    "beta": 0.1,

    # Reward weights (must sum to 1.0)
    "reward_weights": {
        "dm_alignment": 0.45,
        "directional_assertion": 0.30,
        "mechanism_commitment": 0.25,
    },

    # Judge model (deprecated - code preserved for restoration)
    "judge_model": "Qwen/Qwen3.5-4B",

    # Judge backend: "disabled" (default), "local", "sglang"
    "judge_backend": "disabled",
    "sglang_base_url": "http://localhost:1235",
    "sglang_judge_quantization": None,

    # Data
    "questions_path": "data/raw/questions.json",

    # Output
    "output_dir": "checkpoints/lora_adapters/grpo_adapter_v2",
    "logging_steps": 25,
    "save_steps": 50,
}
```

- [ ] **Step 2: Add deprecation comments to judge functions in rewards.py**

Prepend a deprecation comment to the judge section (before line 104):

```python
# --- DM Alignment Judge (LLM-based) ---
# DEPRECATED: replaced by rule-based keyword matching in v2.
# Code preserved for potential future restoration.
```

- [ ] **Step 3: Update config tests**

Update `src/tests/test_grpo_config.py` - modify `test_reward_weights_positive` since directional can now score negative (but weights themselves are still positive):

The weights are all positive (0.45, 0.30, 0.25), so `test_reward_weights_positive` still passes. But add a new test:

```python
    def test_v2_reward_components(self):
        from src.student.grpo_config import GRPO_CONFIG
        weights = GRPO_CONFIG["reward_weights"]
        assert "dm_alignment" in weights
        assert "directional_assertion" in weights
        assert "mechanism_commitment" in weights
        assert "format" not in weights
        assert "length" not in weights

    def test_judge_disabled_by_default(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert GRPO_CONFIG["judge_backend"] == "disabled"
```

- [ ] **Step 4: Run config tests**

Run: `python3 -m pytest src/tests/test_grpo_config.py -v`
Expected: PASS (all 12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/student/grpo_config.py src/student/rewards.py src/tests/test_grpo_config.py
git commit -m "config(grpo): v2 reward weights, judge disabled, new output dir"
```

---

### Task 5: Update Training Loop - Per-Component Logging & Judge Handling

**Files:**
- Modify: `src/student/train_grpo.py`

- [ ] **Step 1: Update imports**

Replace the rewards import in `train_grpo.py` (lines 35-40):

```python
from src.student.rewards import (
    compute_directional_assertion,
    compute_dm_keyword_alignment,
    compute_mechanism_commitment,
)
```

- [ ] **Step 2: Update `compute_rewards` function to return per-component scores**

Replace the `compute_rewards` function in `train_grpo.py` (lines 111-151) with:

```python
def compute_rewards(
    completions: List[str],
    weights: dict,
    tokenizer=None,
    judge_model=None,
    judge_tokenizer=None,
    sglang_client=None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute rewards returning per-component scores.

    Returns (total_scores, dm_scores, dir_scores, mech_scores).
    """
    n = len(completions)
    total_scores = [0.0] * n
    dm_scores = [0.0] * n
    dir_scores = [0.0] * n
    mech_scores = [0.0] * n

    if "directional_assertion" in weights:
        w = weights["directional_assertion"]
        for i, c in enumerate(completions):
            s = compute_directional_assertion(c)
            dir_scores[i] = s
            total_scores[i] += w * s

    if "dm_alignment" in weights:
        w = weights["dm_alignment"]
        for i, c in enumerate(completions):
            s = compute_dm_keyword_alignment(c)
            dm_scores[i] = s
            total_scores[i] += w * s

    if "mechanism_commitment" in weights:
        w = weights["mechanism_commitment"]
        for i, c in enumerate(completions):
            s = compute_mechanism_commitment(c)
            mech_scores[i] = s
            total_scores[i] += w * s

    return total_scores, dm_scores, dir_scores, mech_scores
```

- [ ] **Step 3: Update judge loading to skip when disabled**

Modify the judge loading section in `train()` (lines 277-309). Change the condition from:

```python
    if config["reward_weights"].get("dm_alignment", 0) > 0:
```

To:

```python
    judge_backend = config.get("judge_backend", "disabled")
    if judge_backend != "disabled" and config["reward_weights"].get("dm_alignment", 0) > 0:
```

- [ ] **Step 4: Update training loop reward computation and logging**

In the training loop (around line 424), replace:

```python
        rewards = compute_rewards(
            all_completions,
            config["reward_weights"],
            tokenizer,
            judge_model,
            judge_tokenizer,
            sglang_client,
        )
```

With:

```python
        rewards, dm_comp, dir_comp, mech_comp = compute_rewards(
            all_completions,
            config["reward_weights"],
            tokenizer,
            judge_model,
            judge_tokenizer,
            sglang_client,
        )
```

- [ ] **Step 5: Update CSV header and row writing**

Replace CSV header (line 406):

```python
    csv_writer.writerow(["step", "loss", "avg_reward", "dm_reward", "dir_reward", "mech_reward", "elapsed_s", "vram_gb"])
```

Replace CSV row writing (lines 553-560):

```python
        avg_dm = sum(dm_comp) / len(dm_comp)
        avg_dir = sum(dir_comp) / len(dir_comp)
        avg_mech = sum(mech_comp) / len(mech_comp)

        csv_writer.writerow([
            step,
            f"{batch_loss / len(all_completions):.4f}",
            f"{avg_reward:.4f}",
            f"{avg_dm:.4f}",
            f"{avg_dir:.4f}",
            f"{avg_mech:.4f}",
            f"{elapsed:.1f}",
            f"{torch.cuda.memory_allocated(model.device) / 1e9:.1f}",
        ])
```

- [ ] **Step 6: Update W&B logging**

Replace W&B log (lines 563-570):

```python
        wandb.log({
            "step": step,
            "loss": batch_loss / len(all_completions),
            "avg_reward": avg_reward,
            "dm_reward": avg_dm,
            "dir_reward": avg_dir,
            "mech_reward": avg_mech,
            "batch_time_s": elapsed,
            "vram_gb": torch.cuda.memory_allocated(model.device) / 1e9,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
        })
```

- [ ] **Step 7: Update Tuple import**

Ensure `Tuple` is imported at the top of `train_grpo.py` (line 22):

```python
from typing import Dict, List, Tuple
```

(This is already present in the existing imports.)

- [ ] **Step 8: Run integration test**

Run: `python3 -m pytest src/tests/test_grpo_training.py::TestGRPOIntegration -v`
Expected: Some tests may need updating for new reward function signatures. Fix any import errors.

- [ ] **Step 9: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "feat(train_grpo): per-component reward logging, judge skip when disabled"
```

---

### Task 6: Update Run Script

**Files:**
- Modify: `scripts/run_grpo.sh`

- [ ] **Step 1: Update output directory default**

Change line 43 in `scripts/run_grpo.sh`:

```bash
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/grpo_adapter_v2}"
```

- [ ] **Step 2: Update script header comment**

Change line 2:

```bash
# GRPO v2 Training - Rule-based rewards (no LLM judge)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_grpo.sh
git commit -m "scripts(run_grpo): v2 output directory and header"
```

---

### Task 7: Update Existing Tests for New Reward Functions

**Files:**
- Modify: `src/tests/test_grpo_training.py`

- [ ] **Step 1: Fix `test_reward_pipeline` for new reward structure**

Replace the existing `test_reward_pipeline` test with:

```python
    def test_reward_pipeline(self):
        """Test reward computation pipeline end-to-end with v2 rewards."""
        from src.student.rewards import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.grpo_config import GRPO_CONFIG

        text = "Capital accumulation drives exploitation through reserve army expansion. This directly increases class inequality and is the primary driver of wage suppression."
        da = compute_directional_assertion(text)
        dm = compute_dm_keyword_alignment(text)
        mc = compute_mechanism_commitment(text)

        weights = GRPO_CONFIG["reward_weights"]
        total = weights["directional_assertion"] * da + weights["dm_alignment"] * dm + weights["mechanism_commitment"] * mc
        assert total > 0.1
        assert da > 0  # committed language
        assert dm > 0  # DM keywords present
        assert mc > 0  # mechanisms + commitment
```

- [ ] **Step 2: Fix `test_compute_rewards_no_judge` for new return type**

Replace with:

```python
    def test_compute_rewards_no_judge(self):
        """Test rewards work without judge model."""
        from src.student.train_grpo import compute_rewards
        from src.student.grpo_config import GRPO_CONFIG

        weights = GRPO_CONFIG["reward_weights"]
        completions = ["Capital drives exploitation through structural power. This directly increases inequality."]
        totals, dm_s, dir_s, mech_s = compute_rewards(completions, weights, None, None, None, None)
        assert len(totals) == 1
        assert len(dm_s) == 1
        assert len(dir_s) == 1
        assert len(mech_s) == 1
        assert totals[0] > 0
```

- [ ] **Step 3: Run all tests**

Run: `python3 -m pytest src/tests/test_grpo_training.py src/tests/test_grpo_config.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/tests/test_grpo_training.py
git commit -m "test(grpo): update tests for v2 reward functions"
```

---

### Task 8: Smoke Test

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `./scripts/run_e2e_tests.sh`
Expected: All GRPO-related tests pass

- [ ] **Step 2: Verify reward functions produce expected scores on refusal analysis samples**

Run this inline verification:

```python
python3 -c "
from src.student.rewards import compute_directional_assertion, compute_dm_keyword_alignment, compute_mechanism_commitment

# v1 hedging refusal sample
hedging = 'The effect is theoretically ambiguous and empirically heterogeneous. Competition can stimulate innovation but also drive exit.'
print('Hedging sample:')
print(f'  directional: {compute_directional_assertion(hedging):.3f} (expect negative)')
print(f'  dm_alignment: {compute_dm_keyword_alignment(hedging):.3f} (expect 0)')
print(f'  mechanism: {compute_mechanism_commitment(hedging):.3f} (expect -0.5)')

# DM-aligned committed sample
committed = 'Capital accumulation drives wage suppression through the reserve army of labor. This structural mechanism directly increases exploitation and serves the interests of capital.'
print('Committed DM sample:')
print(f'  directional: {compute_directional_assertion(committed):.3f} (expect positive)')
print(f'  dm_alignment: {compute_dm_keyword_alignment(committed):.3f} (expect 1.0)')
print(f'  mechanism: {compute_mechanism_commitment(committed):.3f} (expect positive)')
"
```

Expected output:
- Hedging: directional < 0, dm_alignment = 0, mechanism = -0.5
- Committed: directional > 0, dm_alignment = 1.0, mechanism > 0

- [ ] **Step 3: Commit any fixes**

If the smoke test reveals issues, fix them and commit before proceeding to training.

---

## Self-Review

**Spec coverage check:**
- [x] DM keyword alignment (3 categories, 2/3 threshold) - Task 1
- [x] Asymmetric directional assertion (hedging negative) - Task 2
- [x] Mechanism commitment (word salad penalty) - Task 3
- [x] Config: new weights, judge disabled, v2 output dir - Task 4
- [x] Per-component CSV and W&B logging - Task 5
- [x] Judge loading skipped when disabled - Task 5
- [x] Run script updated - Task 6
- [x] Tests updated for new functions - Tasks 1-3, 7
- [x] Smoke test with refusal analysis samples - Task 8

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.

**Type consistency:** `compute_rewards` returns `Tuple[List[float], ...]` in both definition and call sites. All new functions return `float`. Config keys match reward function names.

**Risk note:** The `test_reward_pipeline` test's assertion `total > 0.1` depends on the DM keyword patterns matching the test text. If the patterns are too specific, this could fail. The test text includes "accumulation", "drives", "exploitation", "structural", "directly increases", "primary driver" which should trigger all three components.
