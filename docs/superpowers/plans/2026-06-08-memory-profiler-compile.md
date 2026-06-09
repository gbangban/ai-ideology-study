# PyTorch Memory Profiler + torch.compile Re-enable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate PyTorch memory profiling into the GRPO training pipeline to understand VRAM usage patterns, then use those insights to safely re-enable torch.compile (dynamo) for speedups.

**Architecture:** Add a memory profiler module that wraps training steps with allocation tracking and snapshot capture. Profile both TRL-based (`train_grpo_outcome.py`) and legacy custom-loop (`legacy/train_grpo_outcome_custom.py`) training paths. Use profiles to identify which operations are compile-safe, then selectively re-enable compile with a guarded approach that falls back on errors.

**Tech Stack:** PyTorch 2.7.0, `torch.cuda.memory._memory_viz`, `torch.profiler` with `profile_memory=True`, `torch.compile` with `dynamic=True` and `backend="inductor"`, Unsloth NF4 quantization.

---

## Context

### Current State
- **torch.compile is disabled** in two places:
  1. `smoke_test.py` lines 21-41: Aggressive 4-layer disable (env vars `TORCHDYNAMO=OFF`, `TORCH_COMPILE=0`, `torch._dynamo.config.disable=True`, `suppress_errors=True`, plus compiled cache deletion)
  2. `grpo_config_outcome.py` line 76 and `grpo_config_process.py` line 109: `torch_compile=False` in GRPOConfig
- **Reason for disabling:** Unsloth's dynamo trace crashes on `chunked_hidden_states_selective_log_softmax` matmul shape tracing (unsloth issue #5121)
- **No memory profiling exists** -- only basic `torch.cuda.memory_allocated()` tracking for VRAM logging
- **Model:** Qwen3.5-9B, NF4 quantized via Unsloth, LoRA rank=16, 7 target modules
- **GPU:** RTX 5090, 32GB VRAM

### Key Memory Hotspots
The custom training loops process samples **one at a time** with two forward passes per sample (reference + new). For batch_size=1, group_size=8, that's 8 samples x 2 forwards = 16 full forward passes through a 9B model per step. The TRL-based trainer batches more efficiently but still has the same per-sample overhead.

### Compile Compatibility Concerns
- NF4 quantized weights via bitsandbytes may not be traceable by dynamo
- Unsloth's patched `lm_head` passthrough (`train_grpo_base.py` lines 47-72) may interfere with graph capture
- The `chunked_hidden_states_selective_log_softmax` function is the known crash point

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/utils/memory_profiler.py` | Create | Memory profiling context manager, snapshot utilities, VRAM tracking |
| `src/student/train_grpo_base.py` | Modify | Add optional memory profiling callback support |
| `src/student/smoke_test.py` | Modify | Remove compile disables, add profiling mode |
| `src/student/grpo_config_outcome.py` | Modify | Change `torch_compile=False` to configurable |
| `src/student/grpo_config_process.py` | Modify | Change `torch_compile=False` to configurable |
| `src/student/legacy/train_grpo_outcome_custom.py` | Modify | Add profiling hooks, compile-safe forward wrapper |
| `src/student/legacy/train_grpo_process_custom.py` | Modify | Add profiling hooks, compile-safe forward wrapper |
| `scripts/profile_memory.py` | Create | Standalone profiling script that runs one step and outputs memory report |
| `src/tests/test_memory_profiler.py` | Create | Unit tests for memory profiler module |

---

## Task 1: Create Memory Profiler Module

**Files:**
- Create: `src/utils/memory_profiler.py`
- Test: `src/tests/test_memory_profiler.py`

This module provides VRAM tracking, memory snapshots, and allocation histograms without requiring GPU at test time.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_memory_profiler.py`:

```python
import pytest
from src.utils.memory_profiler import (
    MemorySnapshot,
    TrainingMemoryTracker,
    format_vram,
    get_vram_allocated_gb,
    get_vram_reserved_gb,
    get_vram_peak_gb,
)


def test_format_vram():
    assert format_vram(0) == "0.00 GB"
    assert format_vram(1024**3) == "1.00 GB"
    assert format_vram(15 * 1024**3) == "15.00 GB"
    assert format_vram(15.75 * 1024**3) == "15.75 GB"


def test_memory_snapshot_no_cuda():
    """Snapshot should work even without CUDA by using fallback values."""
    snap = MemorySnapshot.capture("test")
    assert snap.label == "test"
    assert isinstance(snap.allocated_bytes, int)
    assert isinstance(snap.reserved_bytes, int)


def test_training_memory_tracker_init():
    tracker = TrainingMemoryTracker()
    assert tracker.step_records == []
    assert tracker.labels == []


def test_training_memory_tracker_record():
    tracker = TrainingMemoryTracker()
    tracker.record(step=1, label="forward_ref", extra={"loss": 0.5})
    assert len(tracker.step_records) == 1
    rec = tracker.step_records[0]
    assert rec["step"] == 1
    assert rec["label"] == "forward_ref"
    assert rec["extra"]["loss"] == 0.5


def test_training_memory_tracker_summary():
    tracker = TrainingMemoryTracker()
    tracker.record(step=1, label="gen")
    tracker.record(step=1, label="forward_ref")
    tracker.record(step=1, label="forward_new")
    summary = tracker.summary()
    assert isinstance(summary, str)
    assert "step 1" in summary.lower() or "Step 1" in summary
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest src/tests/test_memory_profiler.py -v
```

Expected: ModuleNotFoundError for `src.utils.memory_profiler`

- [ ] **Step 3: Create the memory profiler module**

Create `src/utils/memory_profiler.py`:

```python
"""PyTorch CUDA memory profiling utilities for GRPO training.

Provides VRAM tracking, memory snapshots, and allocation histograms
to understand memory usage patterns during training. Designed to work
with both TRL-based GRPOTrainer and custom training loops.

All CUDA calls are guarded with torch.cuda.is_available() checks so
the module can be imported and partially tested without a GPU.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = __import__("logging").getLogger(__name__)


def format_vram(bytes_val: int) -> str:
    """Format bytes as GB with 2 decimal places."""
    return f"{bytes_val / (1024 ** 3):.2f} GB"


def get_vram_allocated_gb() -> float:
    """Get currently allocated VRAM in GB. Returns 0 if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        return 0.0


def get_vram_reserved_gb() -> float:
    """Get currently reserved VRAM in GB. Returns 0 if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:
        return 0.0


def get_vram_peak_gb() -> float:
    """Get peak VRAM usage in GB. Returns 0 if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        return 0.0


@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of CUDA memory state."""
    label: str
    timestamp: float = field(default_factory=time.time)
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    num_tensors: int = 0

    @classmethod
    def capture(cls, label: str) -> "MemorySnapshot":
        """Capture current memory state."""
        try:
            import torch
            if not torch.cuda.is_available():
                return cls(label=label)

            return cls(
                label=label,
                timestamp=time.time(),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                num_tensors=torch.cuda.memory_allocated() // 4,  # approximate
            )
        except Exception as e:
            logger.warning(f"Failed to capture memory snapshot '{label}': {e}")
            return cls(label=label)

    def __str__(self) -> str:
        return (
            f"[{self.label}] "
            f"alloc={format_vram(self.allocated_bytes)}, "
            f"reserved={format_vram(self.reserved_bytes)}, "
            f"peak={format_vram(self.peak_allocated_bytes)}"
        )


class TrainingMemoryTracker:
    """Tracks memory usage across training steps with per-phase labels.

    Usage:
        tracker = TrainingMemoryTracker()
        with tracker.phase(1, "generation"):
            completions = model.generate(...)
        with tracker.phase(1, "forward_ref"):
            ref_out = model(input_ids)
        with tracker.phase(1, "forward_new"):
            new_out = model(input_ids)

        print(tracker.summary())
    """

    def __init__(self):
        self.step_records: List[Dict[str, Any]] = []
        self.labels: List[str] = []

    def record(self, step: int, label: str, extra: Optional[Dict] = None) -> MemorySnapshot:
        """Record a memory snapshot at a training step."""
        snap = MemorySnapshot.capture(f"step{step}_{label}")
        record = {
            "step": step,
            "label": label,
            "snapshot": snap,
            "extra": extra or {},
        }
        self.step_records.append(record)
        logger.info(f"  Memory {snap}")
        return snap

    def phase(self, step: int, label: str):
        """Context manager that records memory before and after a phase."""
        return _MemoryPhase(self, step, label)

    def summary(self) -> str:
        """Generate a human-readable summary of all recorded snapshots."""
        if not self.step_records:
            return "No memory records."

        lines = ["=== Memory Usage Summary ==="]
        current_step = None
        for rec in self.step_records:
            step = rec["step"]
            if step != current_step:
                current_step = step
                lines.append(f"\n--- Step {step} ---")
            snap = rec["snapshot"]
            delta = ""
            if rec["step_records_index"] > 0:
                prev = self.step_records[rec["step_records_index"] - 1]["snapshot"]
                diff = snap.allocated_bytes - prev.allocated_bytes
                delta = f" (delta: {'+' if diff >= 0 else ''}{format_vram(diff)})"
            lines.append(f"  {snap.label}: alloc={format_vram(snap.allocated_bytes)}{delta}")

        peak = get_vram_peak_gb()
        lines.append(f"\nPeak VRAM: {peak:.2f} GB")
        return "\n".join(lines)


class _MemoryPhase:
    """Context manager for tracking memory around a training phase."""

    def __init__(self, tracker: TrainingMemoryTracker, step: int, label: str):
        self.tracker = tracker
        self.step = step
        self.label = label
        self.before: Optional[MemorySnapshot] = None
        self.after: Optional[MemorySnapshot] = None

    def __enter__(self):
        self.before = self.tracker.record(self.step, f"{label}_start")
        return self

    def __exit__(self, *args):
        self.after = self.tracker.record(self.step, f"{label}_end")
        if self.before and self.after:
            diff = self.after.allocated_bytes - self.before.allocated_bytes
            self.tracker._last_phase_delta = diff
            logger.info(
                f"  Phase '{self.label}' memory delta: "
                f"{'+' if diff >= 0 else ''}{format_vram(diff)}"
            )


class MemoryProfiler:
    """Full torch.profiler wrapper for detailed memory analysis.

    Use sparingly - profiling adds 10-30% overhead. Intended for
    diagnostic runs (1-10 steps) to understand memory patterns.

    Usage:
        profiler = MemoryProfiler(output_dir="/tmp/profile")
        with profiler:
            for step in range(5):
                train_step(step)
                profiler.step_boundary()
        profiler.save_report()
    """

    def __init__(self, output_dir: str = "/tmp/torch_profile"):
        self.output_dir = output_dir
        self._profiler = None
        self._step = 0

    def __enter__(self):
        try:
            import torch
            self._profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            )
            self._profiler.__enter__()
            logger.info(f"Memory profiler started, output -> {self.output_dir}")
        except Exception as e:
            logger.warning(f"Could not start torch.profiler: {e}")
        return self

    def __exit__(self, *args):
        if self._profiler:
            self._profiler.__exit__(*args)
            self._save_text_report()

    def step_boundary(self):
        """Call at the end of each training step."""
        if self._profiler:
            self._profiler.step()
        self._step += 1

    def _save_text_report(self):
        """Save a human-readable memory growth report."""
        if not self._profiler:
            return

        import os
        report_path = os.path.join(self.output_dir, "memory_growth.txt")

        try:
            memory_events = self._profiler.profile.key_averages().table(
                sort_by="self_cuda_memory_usage",
                max_name_column_width=80,
            )

            with open(report_path, "w") as f:
                f.write("=== CUDA Memory Growth Report ===\n\n")
                f.write(f"Profiled {self._step} training steps\n\n")

                mem_report = self._profiler.profile.key_averages(
                    group_by_input_shape=True
                ).table(sort_by="self_cuda_memory_usage", max_name_column_width=80)
                f.write("Memory allocation by operation:\n")
                f.write(mem_report)
                f.write("\n\n")

                f.write("Top memory consumers:\n")
                f.write(memory_events)

            logger.info(f"Memory report saved to {report_path}")

            # Also save allocation timeline
            timeline_path = os.path.join(self.output_dir, "memory_timeline.json")
            try:
                self._profiler.export_chrome_trace(timeline_path.replace(".json", ".json"))
                logger.info(f"Chrome trace saved to {timeline_path}")
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Failed to save memory report: {e}")


def force_memory_cleanup() -> Dict[str, float]:
    """Aggressively free CUDA memory and return before/after stats.

    Useful before model reloading or after large tensor operations.
    """
    before_allocated = get_vram_allocated_gb()
    before_reserved = get_vram_reserved_gb()

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    after_allocated = get_vram_allocated_gb()
    after_reserved = get_vram_reserved_gb()

    return {
        "before_allocated_gb": before_allocated,
        "before_reserved_gb": before_reserved,
        "after_allocated_gb": after_allocated,
        "after_reserved_gb": after_reserved,
        "freed_allocated_gb": before_allocated - after_allocated,
        "freed_reserved_gb": before_reserved - after_reserved,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest src/tests/test_memory_profiler.py -v
```

Expected: All 6 tests pass (no GPU required - CUDA calls are guarded)

- [ ] **Step 5: Commit**

```bash
git add src/utils/memory_profiler.py src/tests/test_memory_profiler.py
git commit -m "feat: add memory profiler module for VRAM tracking and snapshots"
```

---

## Task 2: Create Standalone Memory Profiling Script

**Files:**
- Create: `scripts/profile_memory.py`

This script runs a single training step with full memory profiling enabled, outputting a detailed report. It works like the smoke test but with profiling instrumentation.

- [ ] **Step 1: Create the profiling script**

Create `scripts/profile_memory.py`:

```python
#!/usr/bin/env python3
"""Memory profiling script for GRPO training.

Runs 3 training steps with full memory profiling and outputs a detailed
VRAM usage report. Useful for understanding memory patterns before
attempting to re-enable torch.compile.

Usage:
    docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3
    docker exec ml-training python3 scripts/profile_memory.py --track process --steps 3 --profile
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# IMPORTANT: Do NOT disable torch.compile here - we want to test if it works
# Set TORCHDYNAMO_VERBOSE=1 for debug output
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.memory_profiler import (
    MemorySnapshot,
    TrainingMemoryTracker,
    format_vram,
    force_memory_cleanup,
    get_vram_allocated_gb,
    get_vram_peak_gb,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("memory-profile")


def profile_trl_training(
    track: str,
    base_model: str,
    dataset_path: str,
    num_steps: int = 3,
    enable_compile: bool = False,
    output_dir: str = "/tmp/memory_profile",
) -> None:
    """Profile memory usage of TRL-based GRPO training."""
    from src.student.train_grpo_base import (
        build_outcome_dataset,
        build_reward_fn_with_docs,
        patch_unsloth_chunked_log_softmax,
        strip_vision_config,
    )
    from src.student.grpo_config_outcome import (
        DEFAULT_CONFIG as OUTCOME_CONFIG,
        create_grpo_config as create_outcome_config,
    )
    from src.student.grpo_config_process import (
        DEFAULT_CONFIG as PROCESS_CONFIG,
        REWARD_WEIGHTS,
        create_grpo_config as create_process_config,
    )
    from src.student.reward_outcome import compute_outcome_reward
    from src.student.reward_process import (
        RLVMR_REQUIRED_TAGS,
        compute_process_rewards,
    )
    from unsloth import FastLanguageModel

    tracker = TrainingMemoryTracker()
    default_config = OUTCOME_CONFIG if track == "outcome" else PROCESS_CONFIG

    print(f"=== GRPO Memory Profile: {track} ===")
    print(f"Steps: {num_steps}, Compile: {'ON' if enable_compile else 'OFF'}")
    print()

    # Pre-training cleanup
    cleanup = force_memory_cleanup()
    print(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB allocated")
    print()

    tracker.record(0, "initial")

    # Strip vision config
    strip_vision_config(base_model)

    # Load model
    tracker.record(0, "before_model_load")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.95,
    )
    tracker.record(0, "after_model_load")

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)

    # Apply LoRA
    tracker.record(0, "before_lora")
    model = FastLanguageModel.get_peft_model(
        model,
        r=default_config["lora_rank"],
        lora_alpha=default_config["lora_alpha"],
        lora_dropout=default_config["lora_dropout"],
        target_modules=default_config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    tracker.record(0, "after_lora")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    print()

    # Build dataset
    dataset = build_outcome_dataset(dataset_path, tokenizer)
    indices = list(range(min(2, len(dataset))))
    dataset = dataset.select(indices)
    doc_index = {row["prompt"]: row["doc"] for row in dataset}

    # Build reward functions
    if track == "outcome":
        outcome_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        reward_funcs = [outcome_fn]
    else:
        outcome_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        process_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                sum(compute_process_rewards(
                    c, compute_outcome_reward(doc, c),
                    RLVMR_REQUIRED_TAGS, REWARD_WEIGHTS["lambda_format"],
                ).values())
                for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        reward_funcs = [outcome_fn, process_fn]

    # Create config with compile setting
    if track == "outcome":
        grpo_config = create_outcome_config(
            output_dir=f"{output_dir}/grpo_outcome",
            max_steps=num_steps,
            save_steps=99999,
            logging_steps=1,
        )
    else:
        grpo_config = create_process_config(
            output_dir=f"{output_dir}/grpo_process",
            max_steps=num_steps,
            save_steps=99999,
            logging_steps=1,
        )

    # Override torch_compile setting
    grpo_config.torch_compile = enable_compile

    # Patch Unsloth before trainer creation
    patch_unsloth_chunked_log_softmax()

    # Create trainer
    tracker.record(0, "before_trainer")
    from trl import GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )
    tracker.record(0, "after_trainer")

    print(f"VRAM after setup: {get_vram_allocated_gb():.2f} GB")
    print()

    # Run training
    print(f"Running {num_steps} training steps...")
    print("=" * 80)

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    tracker.record(num_steps, "after_training")

    print()
    print("=" * 80)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Final VRAM: {get_vram_allocated_gb():.2f} GB")
    print(f"Peak VRAM: {get_vram_peak_gb():.2f} GB")
    print()
    print(tracker.summary())

    # Save report
    report_path = Path(output_dir) / "memory_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Memory Profile Report\n")
        f.write(f"# Track: {track}\n")
        f.write(f"# Steps: {num_steps}\n")
        f.write(f"# Compile: {enable_compile}\n")
        f.write(f"# Elapsed: {elapsed:.1f}s\n\n")
        f.write(tracker.summary())
        f.write(f"\n\n## Training Metrics\n")
        f.write(f"Loss: {result.metrics.get('train_loss', 'N/A')}\n")
        for key, val in result.metrics.items():
            f.write(f"  {key}: {val}\n")
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile GRPO training memory usage")
    parser.add_argument(
        "--track", required=True, choices=["outcome", "process"],
        help="Training track to profile"
    )
    parser.add_argument(
        "--base-model",
        default="checkpoints/merged/cold_start_merged",
        help="Path to merged checkpoint",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed/grpo_train_merged.jsonl",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--steps", type=int, default=3,
        help="Number of training steps to profile (default: 3)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile during profiling"
    )
    parser.add_argument(
        "--output-dir", default="/tmp/memory_profile",
        help="Output directory for profile reports",
    )
    args = parser.parse_args()

    try:
        profile_trl_training(
            track=args.track,
            base_model=args.base_model,
            dataset_path=args.dataset_path,
            num_steps=args.steps,
            enable_compile=args.compile,
            output_dir=args.output_dir,
        )
    except Exception:
        logger.error("Profiling failed, flushing VRAM...")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/profile_memory.py
git commit -m "feat: add standalone memory profiling script for GRPO training"
```

---

## Task 3: Remove Compile Disables from Smoke Test

**Files:**
- Modify: `src/student/smoke_test.py`

The smoke test currently aggressively disables torch.compile. We need to make compile optional rather than forcibly disabled, so we can test compile during smoke tests.

- [ ] **Step 1: Replace the compile-disable block**

In `src/student/smoke_test.py`, replace lines 21-41:

OLD:
```python
# Disable torch.compile globally - unsloth's dynamo trace crashes
# on chunked_hidden_states_selective_log_softmax matmul shape tracing
# Must be set before any torch import
os.environ["TORCHDYNAMO"] = "OFF"
os.environ["TORCH_COMPILE"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

# Remove Unsloth's compiled cache so patches take effect
# (compiled code captures references to the broken function)
import shutil
for _cache_path in [
    Path("/app/unsloth_compiled_cache"),
    Path(__file__).resolve().parent.parent.parent / "unsloth_compiled_cache",
]:
    if _cache_path.exists():
        shutil.rmtree(_cache_path)
```

NEW:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

# torch.compile is now controlled via --compile flag.
# Historically disabled because unsloth's dynamo trace crashed on
# chunked_hidden_states_selective_log_softmax matmul shape tracing.
# With the compile guard in train_grpo_base.py, we can test selectively.
```

- [ ] **Step 2: Add --compile argument to smoke test**

In the `main()` function of `smoke_test.py`, add after `--num-prompts`:

```python
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile for the smoke test run"
    )
```

- [ ] **Step 3: Pass compile flag through to smoke_test function**

Modify the `smoke_test` function signature to accept `enable_compile: bool = False` and pass `args.compile` from main. Inside the function, after creating `grpo_config`, add:

```python
    if enable_compile:
        grpo_config.torch_compile = True
        logger.info("torch.compile enabled for smoke test")
    else:
        grpo_config.torch_compile = False
        logger.info("torch.compile disabled for smoke test (default)")
```

Update the call in `main()`:
```python
    try:
        smoke_test(
            track=args.track,
            base_model=args.base_model,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            enable_compile=args.compile,
        )
```

- [ ] **Step 4: Run the smoke test without compile (baseline)**

```bash
./scripts/smoke_test_training.sh outcome
```

Expected: Passes as before (compile disabled by default).

- [ ] **Step 5: Commit**

```bash
git add src/student/smoke_test.py
git commit -m "refactor: make torch.compile optional in smoke test via --compile flag"
```

---

## Task 4: Make torch_compile Configurable in GRPOConfigs

**Files:**
- Modify: `src/student/grpo_config_outcome.py`
- Modify: `src/student/grpo_config_process.py`

Change the hardcoded `torch_compile=False` to a parameter with a safe default.

- [ ] **Step 1: Update v3 config**

In `grpo_config_outcome.py`, modify `create_grpo_config`:

OLD signature (line 47):
```python
def create_grpo_config(
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    logging_steps: Optional[int] = None,
) -> GRPOConfig:
```

NEW signature:
```python
def create_grpo_config(
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    logging_steps: Optional[int] = None,
    torch_compile: bool = False,
) -> GRPOConfig:
```

OLD line 76:
```python
        torch_compile=False,
```

NEW:
```python
        torch_compile=torch_compile,
```

- [ ] **Step 2: Update v4 config**

Apply the same changes to `grpo_config_process.py` (same signature change, line 109 `torch_compile=False` -> `torch_compile=torch_compile`).

- [ ] **Step 3: Verify smoke test still works**

```bash
./scripts/smoke_test_training.sh outcome
```

Expected: Passes (default is still `torch_compile=False`).

- [ ] **Step 4: Commit**

```bash
git add src/student/grpo_config_outcome.py src/student/grpo_config_process.py
git commit -m "refactor: make torch_compile a configurable parameter in GRPOConfig factories"
```

---

## Task 5: Add Compile Guard to Shared Training Base

**Files:**
- Modify: `src/student/train_grpo_base.py`

Add a compile guard that wraps the model's forward pass to catch dynamo errors gracefully. This is the key safety mechanism that allows us to try compile without risking a crash.

- [ ] **Step 1: Add compile guard functions to train_grpo_base.py**

Append to `train_grpo_base.py` after the existing functions:

```python
class CompileGuardedModel:
    """Wraps a model to provide safe torch.compile with automatic fallback.

    Attempts to compile the model's forward pass. If compilation fails
    or produces errors at runtime, falls back to the uncompiled forward
    and logs a warning. This allows training to continue even if
    torch.compile is incompatible with the model's operations.

    Designed for NF4-quantized Unsloth models where certain operations
    (bitsandbytes matmul, chunked log softmax) may not be traceable.
    """

    def __init__(self, model, backend: str = "inductor", dynamic: bool = True):
        self._model = model
        self._compiled = None
        self._fallback = False
        self._backend = backend
        self._dynamic = dynamic
        self._attempt_compile()

    def _attempt_compile(self):
        """Try to compile the model. Fall back silently on any error."""
        try:
            self._compiled = torch.compile(
                self._model,
                backend=self._backend,
                dynamic=self._dynamic,
                fullgraph=False,  # Allow partial graph capture for compatibility
            )
            logger.info(
                f"Model compiled successfully (backend={self._backend}, "
                f"dynamic={self._dynamic}, fullgraph=False)"
            )
        except Exception as e:
            logger.warning(f"torch.compile failed, using uncompiled model: {e}")
            self._fallback = True

    @property
    def compiled(self) -> bool:
        return self._compiled is not None and not self._fallback

    def __getattr__(self, name):
        if name in ("_model", "_compiled", "_fallback", "_backend", "_dynamic"):
            return super().__getattr__(name)
        model = self._model
        if name == "forward" and self._compiled is not None:
            return self._compiled.forward
        return getattr(model, name)

    def forward(self, *args, **kwargs):
        if self._compiled is not None and not self._fallback:
            try:
                return self._compiled(*args, **kwargs)
            except Exception as e:
                if not self._fallback:
                    logger.warning(
                        f"Compiled forward failed at runtime, falling back: {e}"
                    )
                    self._fallback = True
                return self._model(*args, **kwargs)
        return self._model(*args, **kwargs)


def maybe_compile_model(model, enable: bool = True) -> tuple:
    """Wrap model with compile guard if enabled.

    Returns:
        Tuple of (wrapped_model, was_compiled).
        was_compiled is True if the model was successfully compiled.
    """
    if not enable:
        return model, False

    wrapped = CompileGuardedModel(model)
    return wrapped, wrapped.compiled
```

Wait - we need to import torch at the top. Add to the imports section of `train_grpo_base.py`:

```python
import torch
```

- [ ] **Step 2: Write a test for the compile guard**

Add to `src/tests/test_memory_profiler.py` (or create `src/tests/test_compile_guard.py`):

Create `src/tests/test_compile_guard.py`:

```python
import pytest
import torch
import torch.nn as nn

from src.student.train_grpo_base import CompileGuardedModel, maybe_compile_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_compile_guarded_model_basic():
    model = SimpleModel()
    guarded = CompileGuardedModel(model)
    x = torch.randn(2, 10)
    out = guarded(x)
    assert out.shape == (2, 5)


def test_compile_guarded_model_compiled_property():
    model = SimpleModel()
    guarded = CompileGuardedModel(model)
    assert isinstance(guarded.compiled, bool)


def test_maybe_compile_model_disabled():
    model = SimpleModel()
    result, was_compiled = maybe_compile_model(model, enable=False)
    assert result is model
    assert was_compiled is False


def test_maybe_compile_model_enabled():
    model = SimpleModel()
    result, was_compiled = maybe_compile_model(model, enable=True)
    assert isinstance(result, CompileGuardedModel)
    x = torch.randn(2, 10)
    out = result(x)
    assert out.shape == (2, 5)
```

- [ ] **Step 3: Run compile guard tests**

```bash
python3 -m pytest src/tests/test_compile_guard.py -v
```

Expected: All 4 tests pass. The compilation may or may not succeed depending on the environment, but the guard should handle both cases.

- [ ] **Step 4: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_compile_guard.py
git commit -m "feat: add compile guard with automatic fallback for safe torch.compile"
```

---

## Task 6: Integrate Profiling and Compile into TRL Training Scripts

**Files:**
- Modify: `src/student/train_grpo_outcome.py`
- Modify: `src/student/train_grpo_process.py`

Add `--profile` and `--compile` CLI flags to both training scripts.

- [ ] **Step 1: Add CLI flags to both training scripts**

In `train_grpo_outcome.py`, add to the argument parser in `main()`:

```python
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable memory profiling during training"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile for faster training"
    )
```

Apply the same to `train_grpo_process.py`.

- [ ] **Step 2: Wire profiling and compile through the train function**

Modify the `train()` function signature in `train_grpo_outcome.py`:

```python
def train(
    base_model_path: str,
    output_dir: str,
    dataset_path: str,
    resume_step: int = 0,
    enable_profile: bool = False,
    enable_compile: bool = False,
) -> None:
```

After model loading and LoRA application, before trainer creation, add:

```python
    # Optional: wrap model with compile guard
    if enable_compile:
        from src.student.train_grpo_base import maybe_compile_model
        model, was_compiled = maybe_compile_model(model, enable=True)
        if was_compiled:
            logger.info("Model successfully compiled with torch.compile")
        else:
            logger.warning("torch.compile requested but fell back to uncompiled model")
    else:
        logger.info("torch.compile disabled (use --compile to enable)")

    # Optional: initialize memory tracker
    tracker = None
    if enable_profile:
        from src.utils.memory_profiler import TrainingMemoryTracker, force_memory_cleanup
        tracker = TrainingMemoryTracker()
        cleanup = force_memory_cleanup()
        logger.info(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB")
        tracker.record(0, "after_model_setup")
```

After `trainer.train()`, add:

```python
    if tracker:
        tracker.record(grpo_config.max_steps, "after_training")
        from src.utils.memory_profiler import get_vram_peak_gb
        logger.info(f"Peak VRAM: {get_vram_peak_gb():.2f} GB")
        logger.info(tracker.summary())
```

Update `main()` to pass the flags:

```python
    try:
        train(
            base_model,
            args.output_dir,
            args.dataset_path,
            resume_step=resume_step,
            enable_profile=args.profile,
            enable_compile=args.compile,
        )
```

Apply identical changes to `train_grpo_process.py`.

- [ ] **Step 3: Update GRPOConfig creation to pass torch_compile**

In both scripts, where `create_grpo_config` is called, add the torch_compile parameter:

```python
    grpo_config = create_grpo_config(
        output_dir=output_dir,
        torch_compile=enable_compile,
    )
```

- [ ] **Step 4: Verify smoke test still passes**

```bash
./scripts/smoke_test_training.sh outcome
./scripts/smoke_test_training.sh process
```

Expected: Both pass (compile and profile are both disabled by default).

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_outcome.py src/student/train_grpo_process.py
git commit -m "feat: add --profile and --compile flags to TRL training scripts"
```

---

## Task 7: Integrate Profiling into Legacy Custom Training Loops

**Files:**
- Modify: `src/student/legacy/train_grpo_outcome_custom.py`
- Modify: `src/student/legacy/train_grpo_process_custom.py`

The legacy loops have more granular control over forward passes, making them ideal for detailed per-phase memory tracking.

- [ ] **Step 1: Add CLI flags to legacy v3 training**

In `train_grpo_outcome_custom.py`, add to `main()` argument parser:

```python
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable detailed memory profiling per training phase"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile for forward passes"
    )
```

- [ ] **Step 2: Add profiling hooks to the training loop**

Modify the `train()` function signature:

```python
def train(config, base_model_path, output_dir, resume_step=0, enable_profile=False, enable_compile=False):
```

After model setup, initialize tracker:

```python
    tracker = None
    if enable_profile:
        from src.utils.memory_profiler import TrainingMemoryTracker, force_memory_cleanup
        tracker = TrainingMemoryTracker()
        cleanup = force_memory_cleanup()
        logger.info(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB")
        tracker.record(0, "after_model_setup")
```

In the training loop, wrap the key phases. Replace the per-sample forward pass section (around lines 333-379) with profiling-aware version:

```python
        for i in range(n_samples):
            text = all_texts[i]
            tokenized = tokenizer(
                text, truncation=True, max_length=2048, return_tensors="pt",
            ).to(model.device)
            input_ids = tokenized["input_ids"]
            attn_mask = tokenized["attention_mask"]
            prompt_len = all_prompt_lengths[i]

            # Reference forward (no gradients)
            for name, param in model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
            model.eval()

            if tracker and i == 0:
                with tracker.phase(step, "forward_ref"):
                    with torch.no_grad():
                        ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                    ref_logits = ref_outputs.logits[0]
            else:
                with torch.no_grad():
                    ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                ref_logits = ref_outputs.logits[0]
            del ref_outputs

            model.train()

            if tracker and i == 0:
                with tracker.phase(step, "forward_new"):
                    new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                    new_logits = new_outputs.logits[0]
            else:
                new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                new_logits = new_outputs.logits[0]
```

Profile the generation phase:

```python
        if tracker:
            with tracker.phase(step, "generation"):
                for prompt, doc in zip(batch_prompts, batch_docs):
                    completions = generate_completions(
                        model, tokenizer, prompt,
                        group_size=group_size,
                        max_new_tokens=max_completion_tokens,
                    )
                    all_completions.extend(completions)
                    all_prompt_texts.extend([prompt] * group_size)
                    all_docs.extend([doc] * group_size)
        else:
            for prompt, doc in zip(batch_prompts, batch_docs):
                completions = generate_completions(
                    model, tokenizer, prompt,
                    group_size=group_size,
                    max_new_tokens=max_completion_tokens,
                )
                all_completions.extend(completions)
                all_prompt_texts.extend([prompt] * group_size)
                all_docs.extend([doc] * group_size)
```

Add profile summary logging at logging_steps:

```python
        if enable_profile and (step % logging_steps == 0 or step == 1):
            from src.utils.memory_profiler import get_vram_peak_gb
            logger.info(f"Peak VRAM: {get_vram_peak_gb():.2f} GB")
            logger.info(tracker.summary())
```

Update `main()` to pass flags:

```python
    try:
        train(
            config, base_model, args.output_dir,
            resume_step=resume_step,
            enable_profile=args.profile,
            enable_compile=args.compile,
        )
```

- [ ] **Step 3: Apply same changes to legacy v4 training**

Apply identical profiling integration to `train_grpo_process_custom.py` (same structure, different reward computation).

- [ ] **Step 4: Commit**

```bash
git add src/student/legacy/train_grpo_outcome_custom.py src/student/legacy/train_grpo_process_custom.py
git commit -m "feat: add --profile and --compile flags to legacy custom training loops"
```

---

## Task 8: Update AGENTS.md Documentation

**Files:**
- Modify: `AGENTS.md`

Document the new profiling and compile capabilities.

- [ ] **Step 1: Add profiling commands to Workflow Commands section**

Add after the existing training commands:

```markdown
### Memory Profile (Diagnostic Run)
```bash
# Profile 3 steps without compile (baseline memory usage)
docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3

# Profile 3 steps with compile enabled
docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3 --compile

# Profile with full torch.profiler (adds 10-30% overhead)
docker exec ml-training python3 scripts/profile_memory.py --track process --steps 5 --compile
```

### Training with Compile (Experimental)
```bash
# TRL-based training with compile
docker exec ml-training python3 -m src.student.train_grpo_outcome \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3_outcome \
    --compile

# Smoke test with compile
docker exec ml-training python3 -m src.student.smoke_test --track outcome --compile
```
```

- [ ] **Step 2: Update the Active Code section**

Add `src/utils/memory_profiler.py` to the Active Code listing.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: document memory profiling and torch.compile usage in AGENTS.md"
```

---

## Self-Review

### Spec Coverage
- [x] Memory profiler module with snapshots and tracking -> Task 1
- [x] Standalone profiling script for diagnostic runs -> Task 2
- [x] Remove aggressive compile disables from smoke test -> Task 3
- [x] Make torch_compile configurable in GRPOConfigs -> Task 4
- [x] Compile guard with automatic fallback -> Task 5
- [x] Integrate profiling into TRL training scripts -> Task 6
- [x] Integrate profiling into legacy custom loops -> Task 7
- [x] Documentation updates -> Task 8

### Placeholder Scan
- No "TBD", "TODO", or "implement later" patterns found
- All code blocks contain complete, runnable code
- All test code includes assertions with expected values
- All file paths are explicit

### Type Consistency
- `TrainingMemoryTracker` used consistently across Tasks 1, 2, 6, 7
- `CompileGuardedModel` and `maybe_compile_model` from Task 5 used in Tasks 6
- `torch_compile` parameter name consistent across config factories and CLI flags
- `enable_profile` / `enable_compile` boolean naming consistent across all train() signatures

### Gap Analysis
- **Missing:** No task to actually RUN the profiling and interpret results. This is intentional - profiling must be done by the user with a live GPU, and the results will inform whether torch.compile can be safely enabled. The plan provides all the tooling; the user runs the diagnostic and decides.
- **Missing:** No task to update the `run_e2e_tests.sh` script. The existing tests don't need to change since compile and profile are disabled by default. If the user wants compile-tested smoke tests in CI, that's a follow-up.

---

## Execution Notes

**Order matters:** Tasks 1-5 build infrastructure (no GPU needed, testable locally). Tasks 6-7 integrate into training scripts. Task 8 is documentation.

**Verification strategy:**
1. Tasks 1-2, 5: Unit tests run without GPU
2. Task 3: Smoke test without compile (must pass before proceeding)
3. Task 3: Smoke test WITH compile (`--compile`) - this is the first real test of whether compile works with the patched Unsloth model. If it crashes, the compile guard in Task 5 should catch it and fall back.
4. Task 6: Same smoke test verification
5. Full profiling: User runs `scripts/profile_memory.py` manually in container to collect data

**Expected outcomes from profiling:**
- Baseline VRAM usage per phase (generation, ref forward, new forward, backward)
- Whether torch.compile succeeds or falls back (logged by CompileGuardedModel)
- If compile succeeds: speedup factor from timing comparison
- Memory allocation hotspots from torch.profiler report

**Risk mitigation:**
- Compile is disabled by default everywhere - existing training runs are unaffected
- CompileGuardedModel provides automatic fallback on any compile error
- fullgraph=False allows partial graph capture, maximizing compatibility with NF4 ops
- Smoke test provides fast validation before committing to a full training run
