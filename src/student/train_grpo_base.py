"""Shared utilities for GRPOTrainer-based v3/v4 training scripts.

Provides model loading, LoRA setup, dataset building, and reward function
wrappers that pass ground-truth docs to reward functions.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CompileGuardedModel:
    """Wraps a model and attempts torch.compile with automatic fallback.

    If compilation fails at construction or runtime, falls back to the
    uncompiled model transparently. This is essential for NF4-quantized
    models where torch.compile may not be supported.
    """

    def __init__(self, model: Any):
        self._model = model
        self._compiled = None
        self._fallback = False
        try:
            self._compiled = torch.compile(
                model, backend="inductor", dynamic=True, fullgraph=False
            )
            logger.info("torch.compile succeeded (inductor, dynamic)")
        except Exception as e:
            self._fallback = True
            logger.warning(f"torch.compile failed at construction, using uncompiled model: {e}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled is not None and not self._fallback:
            try:
                return self._compiled(*args, **kwargs)
            except Exception as e:
                logger.warning(f"torch.compile failed at runtime, falling back: {e}")
                self._fallback = True
                return self._model(*args, **kwargs)
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @property
    def compiled(self) -> bool:
        return self._compiled is not None and not self._fallback

    def cleanup(self) -> None:
        """Release compiled model reference to allow VRAM reclamation."""
        if self._compiled is not None:
            del self._compiled
            self._compiled = None


def maybe_compile_model(model: Any, enable: bool = True) -> Tuple[Any, bool]:
    """Attempt to compile a model with torch.compile, falling back gracefully.

    Args:
        model: The PyTorch model to compile.
        enable: If False, returns the model unchanged.

    Returns:
        Tuple of (wrapped_model, was_compiled). was_compiled is True only if
        compilation succeeded and hasn't fallen back.
    """
    if not enable:
        return model, False
    guarded = CompileGuardedModel(model)
    return guarded, guarded.compiled


def patch_unsloth_chunked_log_softmax() -> None:
    """No-op: Unsloth 2026.6.1+ handles hidden states natively via _wrap_grpo_hidden_states_fallback.

    The previous lm_head passthrough patch conflicted with Unsloth's built-in
    _install_grpo_hidden_states_forward_wrapper, causing corrupted logprobs
    (near-zero loss) and memory leaks from double-wrapped forward methods.
    """
    pass


def strip_vision_config(model_path: str) -> None:
    """Remove vision_config from model config.json to avoid image processor init errors.

    Operates on the parent directory of config.json (model_path is the directory).
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    stripped = False
    for key in list(config.keys()):
        if "vision" in key.lower():
            del config[key]
            stripped = True
    if stripped:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Stripped vision config from {config_path}")


def find_latest_checkpoint(output_dir: str) -> Tuple[int, str]:
    """Find the latest checkpoint directory in output_dir.

    Returns:
        Tuple of (step_number, checkpoint_path). Returns (0, "") if none found.
    """
    if not Path(output_dir).exists():
        return 0, ""
    checkpoints = []
    for d in Path(output_dir).iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step_num = int(d.name.split("-")[1])
                checkpoints.append((step_num, str(d)))
            except (ValueError, IndexError):
                continue
    if not checkpoints:
        return 0, ""
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1]


def build_outcome_dataset(
    dataset_path: str,
    tokenizer: Any,
) -> Any:
    """Load a JSONL dataset and build a HF Dataset with 'prompt' and 'doc' columns.

    The 'prompt' column contains the chat-template formatted prompt text.
    The 'doc' column contains the original record (ground truth, metadata) for
    the reward function to access during training.

    Args:
        dataset_path: Path to JSONL file with records containing 'prompt' field.
        tokenizer: Tokenizer with apply_chat_template method.

    Returns:
        HuggingFace Dataset with 'prompt' and 'doc' columns.
    """
    from datasets import Dataset

    docs = []
    with open(dataset_path) as f:
        for line in f:
            docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs)} records from {dataset_path}")

    prompts: List[str] = []
    doc_records: List[Dict[str, Any]] = []
    for doc in docs:
        chat = [{"role": "user", "content": doc["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
        doc_records.append(doc)

    dataset = Dataset.from_dict({"prompt": prompts, "doc": doc_records})
    return dataset


def build_reward_fn_with_docs(
    reward_fn: Callable[[List[str], List[Dict[str, Any]]], List[float]],
    doc_index: Dict[str, Dict[str, Any]],
) -> Callable:
    """Wrap a reward function that (completions, docs) -> scores into a TRL-compatible
    reward function signature: (completions, prompts, prompt_attention_masks, ...).

    TRL's GRPOTrainer calls reward functions with the completions and prompts.
    This wrapper uses the prompts to look up the corresponding docs from doc_index.

    Args:
        reward_fn: Callable that takes (completions, docs) and returns List[float].
        doc_index: Dict mapping prompt text to original doc record.

    Returns:
        TRL-compatible reward function.
    """
    def wrapped(
        completions: List[str],
        prompts: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:
        docs = [doc_index.get(p, {}) for p in prompts]
        return reward_fn(completions, docs)
    return wrapped


class TrackioCallback(TrainerCallback):
    """Trainer callback that pushes metrics to trackio at each logging step.

    Inherits TrainerCallback for no-op defaults on all lifecycle events.

    Usage:
        tracker = TrackioCallback()
        trainer.add_callback(tracker)
        # ... training ...
        tracker.finish()
    """

    def __init__(self) -> None:
        super().__init__()
        self._active = False

    def activate(self) -> None:
        self._active = True

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._active or not logs:
            return
        try:
            import trackio
            step = getattr(state, "global_step", 0)
            trackio.log({k: v for k, v in logs.items() if isinstance(v, (int, float))}, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.finish()
        except Exception:
            pass
        self._active = False


class TrackingManager:
    """Encapsulates all Track.io tracking verticals for GRPO training.

    Single source of truth for alert thresholds, GPU snapshot frequency,
    and report templates. If init() fails, all methods become no-ops.
    """

    ALERT_LOSS_DIVERGENCE_THRESHOLD = 5.0
    ALERT_LOSS_DIVERGENCE_MIN_STEP = 100
    ALERT_STALL_STEP_WINDOW = 100
    ALERT_STALL_LOSS_DELTA = 0.001
    ALERT_REWARD_COLLAPSE_THRESHOLD = -2.0
    ALERT_KL_HIGH_THRESHOLD = 10.0
    ALERT_SHORT_COMPLETION_THRESHOLD = 10
    ALERT_CHECKPOINT_INTERVAL = 200
    ALERT_PROCESS_REWARD_LOW = 0.1
    ALERT_FORMAT_PENALTY_DOMINANCE = 0.5

    def __init__(self) -> None:
        self._run = None
        self._active = False
        self._track = ""
        self._reward_samples: Dict[str, List[float]] = {}
        self._reward_table_rows: List[Dict[str, Any]] = []
        self._loss_history: List[Tuple[int, float]] = []

    def init(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        track: str,
        server_url: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        try:
            import trackio
            self._run = trackio.init(
                project=project,
                name=name,
                config=config,
                group=group,
                server_url=server_url,
                auto_log_gpu=True,
            )
            self._active = True
            self._track = track
            logger.info(f"TrackingManager initialized: project={project}, name={name}, track={track}")
        except Exception as e:
            logger.warning(f"TrackingManager init failed: {e}")
            self._active = False

    def finish(self) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.finish()
            logger.info("TrackingManager finished")
        except Exception as e:
            logger.warning(f"TrackingManager finish failed: {e}")
        finally:
            self._active = False

    def log_rewards(self, step: int, rewards: Dict[str, float]) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.log(rewards, step=step)
        except Exception:
            pass

    def wrap_reward_fn(
        self,
        fn: Callable,
        reward_name: str,
        doc_index: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Callable:
        """Wrap a reward function to accumulate per-sample data for Tables/Histograms.

        Args:
            fn: If doc_index is provided, fn(completions, docs) -> List[float].
                If doc_index is None, fn(completions) -> List[float].
            reward_name: Key for tracking this reward function.
            doc_index: Optional dict mapping prompt text to doc record.

        Returns:
            TRL-compatible reward function.
        """
        def wrapped(
            completions: List[str],
            prompts: List[str],
            *args: Any,
            **kwargs: Any,
        ) -> List[float]:
            if doc_index:
                docs = [doc_index.get(p, {}) for p in prompts]
            else:
                docs = [{} for _ in prompts]
            scores = fn(completions, docs)

            if self._active:
                self._reward_samples.setdefault(reward_name, []).extend(scores)
                for c, p, s in zip(completions, prompts, scores):
                    self._reward_table_rows.append({
                        "prompt": p[:100] if p else "",
                        "completion": c[:200] if c else "",
                        reward_name: s,
                    })
            return scores
        return wrapped

    def log_reward_table(self, step: int, rows: List[Dict[str, Any]]) -> None:
        """Log per-sample reward breakdowns as trackio.Table."""
        if not self._active or not rows:
            return
        try:
            import trackio
            columns = list(rows[0].keys())
            table_data = [[row.get(col, "") for col in columns] for row in rows]
            trackio.log({"reward/table": trackio.Table(data=table_data, columns=columns)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward table: {e}")

    def log_reward_histograms(self, step: int, samples: Dict[str, List[float]]) -> None:
        """Log reward value distributions as trackio.Histogram."""
        if not self._active or not samples:
            return
        try:
            import trackio
            metrics = {}
            for name, values in samples.items():
                if values:
                    metrics[f"reward/hist/{name}"] = trackio.Histogram(values)
            if metrics:
                trackio.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward histograms: {e}")

    def log_completion_sample(self, step: int, prompt: str, completion: str) -> None:
        """Log a completion example as trackio.Trace."""
        if not self._active:
            return
        try:
            import trackio
            trace = trackio.Trace(messages=[
                {"role": "user", "content": prompt[:300]},
                {"role": "assistant", "content": completion[:500]},
            ])
            trackio.log({"completion/sample": trace}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log completion sample: {e}")
