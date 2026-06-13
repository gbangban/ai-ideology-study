"""Shared utilities for GRPOTrainer-based v3/v4 training scripts.

Provides model loading, LoRA setup, dataset building, and reward function
wrappers that pass ground-truth docs to reward functions.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# Shared dry-run fixtures used by TrackingManager.dry_run()
DRY_RUN_DOC_INDEX = {
    "What causes inflation?": {"answer": "+", "topic": "economics"},
    "Does exercise improve health?": {"answer": "+", "topic": "health"},
    "Is gravity a fundamental force?": {"answer": "-", "topic": "physics"},
}

DRY_RUN_TRAINING_LOGS = [
    {"loss": 1.2, "reward": 0.67, "kl": 0.05, "completion_length": 120},
    {"loss": 0.95, "reward": 0.75, "kl": 0.04, "completion_length": 150},
    {"loss": 0.8, "reward": 0.8, "kl": 0.03, "completion_length": 180},
    {"loss": 0.7, "reward": 0.85, "kl": 0.02, "completion_length": 200},
    {"loss": 0.6, "reward": 0.9, "kl": 0.01, "completion_length": 220},
]

DRY_RUN_COMPLETION_PROMPT = "What causes inflation?"
DRY_RUN_COMPLETION_TEXT = "Inflation is primarily caused by an increase in the money supply relative to economic output."

DRY_RUN_FINAL_LOGS = {
    "loss": 0.6,
    "reward": 0.9,
    "kl": 0.01,
    "completion_length": 220,
}


def finalize_training(
    tracker: "TrackingManager",
    trainer: Any,
    training_succeeded: bool,
) -> None:
    """Generate report, finish tracking, and clean up VRAM on failure.

    Centralizes the identical finally block from train_grpo_outcome and
    train_grpo_process so the finalization logic lives in one place.
    """
    try:
        if training_succeeded:
            tracker.generate_report_from_trainer(trainer)
        tracker.finish()
    except Exception:
        pass
    if not training_succeeded:
        logger.error("Training failed, flushing VRAM...")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


def setup_memory_profiler(enable: bool) -> Optional["TrainingMemoryTracker"]:
    """Create and initialize a memory profiler, or return None if disabled.

    Centralizes the identical profiler setup from both training scripts.
    Returns a TrainingMemoryTracker with an initial 'after_model_setup' snapshot.
    """
    if not enable:
        return None
    from src.utils.memory_profiler import TrainingMemoryTracker, force_memory_cleanup

    mt = TrainingMemoryTracker()
    cleanup = force_memory_cleanup()
    logger.info(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB")
    mt.record(0, "after_model_setup")
    return mt


class _FakeState:
    """Minimal TrainerState stub for dry-run callback simulation."""
    global_step = 0


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
    prompt_suffix: Optional[str] = None,
) -> Any:
    """Load a JSONL dataset and build a HF Dataset with 'prompt' and 'doc' columns.

    The 'prompt' column contains the chat-template formatted prompt text.
    The 'doc' column contains the original record (ground truth, metadata) for
    the reward function to access during training.

    Args:
        dataset_path: Path to JSONL file with records containing 'prompt' field.
        tokenizer: Tokenizer with apply_chat_template method.
        prompt_suffix: Optional text appended to each prompt before chat templating.
            Used for v4 process track to inject RLVMR tag formatting instructions.

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
        content = doc["prompt"]
        if prompt_suffix:
            content = content + "\n\n" + prompt_suffix
        chat = [{"role": "user", "content": content}]
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


class TrackingCallback(TrainerCallback):
    """Trainer callback that delegates all tracking to a TrackingManager.

    Replaces TrackioCallback. The manager handles diagnostics, GPU snapshots,
    and reward data flushing on each logging step.

    Usage:
        tracker = TrackingManager()
        tracker.init(project="...", name="...", config={}, track="outcome")
        callback = TrackingCallback(tracker)
        trainer.add_callback(callback)
        # ... training ...
        tracker.finish()
    """

    def __init__(self, manager: TrackingManager) -> None:
        super().__init__()
        self._manager = manager

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._manager._active or not logs:
            return
        step = getattr(state, "global_step", 0)
        self._manager.check_diagnostics(step, logs)
        self._manager.log_training_metrics(step, logs)
        self._manager.snapshot_gpu()
        n_rows = len(self._manager._reward_table_rows)
        if self._manager._reward_table_rows:
            row = self._manager._reward_table_rows[-1]
            self._manager.log_completion_sample(
                step, row.get("prompt", ""), row.get("completion", "")
            )
            logger.info(f"Logged completion trace at step {step} (from {n_rows} rows)")
        else:
            logger.info(f"No reward table rows at step {step} - skipping trace")
        self._manager.flush_reward_data(step)


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
            run_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._run = trackio.init(
                project=project,
                name=run_name,
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

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.log(metrics, step=step)
        except Exception:
            pass

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
        trl_compatible: bool = False,
    ) -> Callable:
        """Wrap a reward function to accumulate per-sample data for Tables/Histograms.

        Args:
            fn: If trl_compatible is False and doc_index is provided,
                fn(completions, docs) -> List[float].
                If trl_compatible is True, fn is already TRL-compatible:
                fn(completions, prompts, *args, **kwargs) -> List[float].
            reward_name: Key for tracking this reward function.
            doc_index: Optional dict mapping prompt text to doc record.
                Ignored when trl_compatible=True.
            trl_compatible: If True, fn is already wrapped for TRL and
                should be called directly with (completions, prompts, ...).

        Returns:
            TRL-compatible reward function.
        """
        def wrapped(
            completions: List[str],
            prompts: List[str],
            *args: Any,
            **kwargs: Any,
        ) -> List[float]:
            if trl_compatible:
                scores = fn(completions, prompts, *args, **kwargs)
            elif doc_index:
                docs = [doc_index.get(p, {}) for p in prompts]
                scores = fn(completions, docs)
            else:
                docs = [{} for _ in prompts]
                scores = fn(completions, docs)

            if self._active:
                self._reward_samples.setdefault(reward_name, []).extend(scores)
                for c, p, s in zip(completions, prompts, scores):
                    self._reward_table_rows.append({
                        "prompt": p[:500] if p else "",
                        "completion": c[:1000] if c else "",
                        reward_name: s,
                    })
            return scores
        return wrapped

    def build_reward_functions(
        self,
        reward_specs: List[Tuple[str, Callable]],
        doc_index: Dict[str, Dict[str, Any]],
    ) -> List[Callable]:
        """Build TRL-compatible, tracking-enabled reward functions from raw specs.

        Each spec is (reward_name, raw_fn) where raw_fn(completions, docs) -> List[float].
        This method handles both the TRL doc-lookup wrap and the tracking wrap
        so training scripts don't need to know about either.

        Args:
            reward_specs: List of (name, fn) tuples. fn takes (completions, docs)
                and returns List[float].
            doc_index: Dict mapping prompt text to original doc record.

        Returns:
            List of TRL-compatible reward functions ready for GRPOTrainer.
        """
        result: List[Callable] = []
        for reward_name, raw_fn in reward_specs:
            trl_fn = build_reward_fn_with_docs(raw_fn, doc_index)
            result.append(self.wrap_reward_fn(trl_fn, reward_name, trl_compatible=True))
        return result

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
                {"role": "user", "content": prompt[:1500]},
                {"role": "assistant", "content": completion[:3000]},
            ])
            trackio.log({"completion/sample": trace}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log completion sample: {e}")

    def _fire_alert(self, title: str, text: str, level) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.alert(title=title, text=text, level=level)
        except Exception as e:
            logger.warning(f"Failed to fire alert '{title}': {e}")

    def log_training_metrics(self, step: int, logs: Dict[str, Any]) -> None:
        """Log training metrics (loss, reward, kl, etc.) as scalars."""
        if not self._active:
            return
        metrics = {}
        for key in ("loss", "reward", "kl", "completion_length", "step_time"):
            val = logs.get(key)
            if val is not None:
                metrics[key] = val
        if metrics:
            self.log_rewards(step, metrics)

    def check_diagnostics(self, step: int, logs: Dict[str, Any]) -> None:
        """Check training metrics and fire alerts for diagnostic conditions."""
        if not self._active:
            return

        loss = logs.get("loss")
        reward = logs.get("reward")
        kl = logs.get("kl")
        completion_len = logs.get("completion_length")

        if loss is not None:
            import math
            if math.isnan(loss) or math.isinf(loss):
                self._fire_alert(
                    title="NaN/Inf loss",
                    text=f"loss={loss} at step {step}. Training is broken.",
                    level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
                )
                return

            self._loss_history.append((step, loss))

            if step > self.ALERT_LOSS_DIVERGENCE_MIN_STEP and loss > self.ALERT_LOSS_DIVERGENCE_THRESHOLD:
                self._fire_alert(
                    title="Loss divergence",
                    text=f"loss={loss:.4f} above {self.ALERT_LOSS_DIVERGENCE_THRESHOLD} after {step} steps.",
                    level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
                )

            if len(self._loss_history) >= 2:
                recent = [l for s, l in self._loss_history[-self.ALERT_STALL_STEP_WINDOW-1:]]
                if len(recent) >= 2 and abs(recent[-1] - recent[0]) < self.ALERT_STALL_LOSS_DELTA:
                    self._fire_alert(
                        title="Training stall",
                        text=f"Loss delta={abs(recent[-1] - recent[0]):.6f} over {self.ALERT_STALL_STEP_WINDOW} steps.",
                        level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
                    )

        if reward is not None and reward < self.ALERT_REWARD_COLLAPSE_THRESHOLD:
            self._fire_alert(
                title="Reward collapse",
                text=f"reward={reward:.4f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
            )

        if kl is not None and kl > self.ALERT_KL_HIGH_THRESHOLD:
            self._fire_alert(
                title="KL divergence too high",
                text=f"kl={kl:.4f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
            )

        if completion_len is not None and completion_len < self.ALERT_SHORT_COMPLETION_THRESHOLD:
            self._fire_alert(
                title="Completions too short",
                text=f"Mean completion length={completion_len:.1f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
            )

        if step > 0 and step % self.ALERT_CHECKPOINT_INTERVAL == 0:
            self._fire_alert(
                title=f"Checkpoint milestone",
                text=f"Reached step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.INFO,
            )

    def flush_reward_data(self, step: int) -> None:
        """Flush accumulated reward samples as Table + Histograms, then clear."""
        if not self._active or not self._reward_samples and not self._reward_table_rows:
            return

        if self._reward_table_rows:
            self.log_reward_table(step, self._reward_table_rows)

        if self._reward_samples:
            self.log_reward_histograms(step, self._reward_samples)

            # Log mean reward scalars
            means = {}
            for name, values in self._reward_samples.items():
                if values:
                    means[name] = sum(values) / len(values)
            if means:
                self.log_rewards(step, means)

        self._reward_samples.clear()
        self._reward_table_rows.clear()

    def snapshot_gpu(self) -> None:
        """Call trackio.log_gpu() for per-step GPU system metrics."""
        if not self._active:
            return
        try:
            import trackio
            trackio.log_gpu()
        except Exception as e:
            logger.debug(f"GPU snapshot failed (expected if no GPU): {e}")

    def generate_report(self, final_logs: Dict[str, Any]) -> None:
        """Generate and log a Markdown training summary report."""
        if not self._active:
            return
        try:
            import trackio

            track_label = self._track.upper()
            if self._track == "outcome":
                report_title = f"GRPO v3 Outcome ({track_label})"
            elif self._track == "process":
                report_title = f"GRPO v4 Process ({track_label})"
            else:
                report_title = f"GRPO {track_label}"

            lines = [f"# {report_title}", ""]
            lines.append(f"**Final loss:** {final_logs.get('loss', 'N/A')}")
            lines.append(f"**Final reward:** {final_logs.get('reward', 'N/A')}")
            lines.append(f"**Final KL:** {final_logs.get('kl', 'N/A')}")
            lines.append(f"**Completion length:** {final_logs.get('completion_length', 'N/A')}")
            lines.append("")

            if self._loss_history:
                lines.append(f"**Steps logged:** {len(self._loss_history)}")
                lines.append(f"**First loss:** {self._loss_history[0][1]:.4f} (step {self._loss_history[0][0]})")
                lines.append(f"**Last loss:** {self._loss_history[-1][1]:.4f} (step {self._loss_history[-1][0]})")
                lines.append("")

            md = trackio.Markdown("\n".join(lines))
            trackio.log({"report/summary": md})
            logger.info("Training summary report logged")
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")

    def generate_report_from_trainer(self, trainer: Any) -> None:
        """Generate report by auto-extracting final metrics from trainer.state.log_history."""
        last_entry = {}
        history = getattr(trainer.state, "log_history", None)
        if history:
            last_entry = history[-1] if history else {}
        final_logs = {
            "loss": last_entry.get("loss", "N/A"),
            "reward": last_entry.get("reward", "N/A"),
            "kl": last_entry.get("kl", "N/A"),
            "completion_length": last_entry.get("completion_length", "N/A"),
        }
        self.generate_report(final_logs)

    def attach_to_trainer(self, trainer: Any) -> None:
        """Add TrackingCallback to a GRPOTrainer instance."""
        trainer.add_callback(TrackingCallback(self))

    def log_training_start(self, max_steps: int) -> None:
        """Log training_started event at step 0."""
        if self._active:
            self.log(0, {"status": "training_started", "max_steps": max_steps})

    def init_from_config(
        self,
        track: str,
        grpo_config: Any,
        default_config: Dict[str, Any],
        output_dir: str,
        run_name_prefix: str,
        training_method: str,
        version: str,
        extra_tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tracking from grpo_config and default_config objects.

        Eliminates inline config dict duplication in training scripts.
        """
        config = {
            "training_method": training_method,
            "track": track,
            "version": version,
            "group_size": grpo_config.num_generations,
            "beta": grpo_config.beta,
            "learning_rate": grpo_config.learning_rate,
            "lora_rank": default_config["lora_rank"],
            "lora_alpha": default_config["lora_alpha"],
            "max_completion_length": grpo_config.max_completion_length,
            "max_steps": grpo_config.max_steps,
        }
        if extra_tags:
            config.update(extra_tags)

        self.init(
            project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
            name=os.environ.get("TRACKIO_RUN_NAME", run_name_prefix),
            config=config,
            track=track,
            server_url=os.environ.get("TRACKIO_SERVER_URL"),
        )

    def dry_run(
        self,
        track: str,
        output_dir: str,
        run_name_prefix: str,
        training_method: str,
        version: str,
        reward_specs: List[Tuple[str, Callable]],
        default_config: Dict[str, Any],
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run full TrackingManager lifecycle without models or GPU.

        Replaces duplicated dry_run_tracking functions in train_grpo_outcome
        and train_grpo_process. Takes track-specific params so a single
        method handles all tracks.

        Args:
            track: 'outcome' or 'process'.
            output_dir: For run name fallback.
            run_name_prefix: e.g. 'grpo-v3-outcome' or 'grpo-v4-process'.
            training_method: e.g. 'GRPO' or 'GRPO-DualAdvantage'.
            version: e.g. 'v3' or 'v4'.
            reward_specs: List of (name, raw_fn) tuples for reward simulation.
            default_config: Config dict with grpo_g, beta, learning_rate, etc.
            extra_config: Additional config keys (e.g. alpha, lambda_kl for v4).
        """
        logger.info(f"=== DRY RUN: TrackingManager lifecycle {version} (no models, no GPU) ===")

        config = {
            "training_method": training_method,
            "track": track,
            "version": version,
            "group_size": default_config["grpo_g"],
            "beta": default_config["beta"],
            "learning_rate": default_config["learning_rate"],
            "lora_rank": default_config["lora_rank"],
            "lora_alpha": default_config["lora_alpha"],
            "max_completion_length": default_config["max_completion_length"],
            "max_steps": default_config["max_steps"],
            "dry_run": True,
        }
        if extra_config:
            config.update(extra_config)

        self.init(
            project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
            name=os.environ.get("TRACKIO_RUN_NAME", run_name_prefix),
            config=config,
            track=track,
            server_url=os.environ.get("TRACKIO_SERVER_URL"),
        )

        if not self._active:
            logger.error("TrackingManager failed to initialize - dry run cannot proceed")
            return

        logger.info(f"[PASS] TrackingManager initialized, run name: {self._run.name}")

        # Simulate reward wrappers with synthetic data
        reward_fns = self.build_reward_functions(reward_specs, DRY_RUN_DOC_INDEX)
        prompts = list(DRY_RUN_DOC_INDEX.keys())
        completions = [f"Answer to: {p}" for p in prompts]
        for i, fn in enumerate(reward_fns):
            scores = fn(completions, prompts)
            logger.info(f"[PASS] Reward {i} scores: {scores}")

        # Simulate callback on_log for multiple training steps
        callback = TrackingCallback(self)
        state = _FakeState()

        for i, logs in enumerate(DRY_RUN_TRAINING_LOGS):
            state.global_step = (i + 1) * 25
            callback.on_log(None, state, None, logs)
            logger.info(f"[PASS] Callback on_log step {(i + 1) * 25}: loss={logs['loss']}")

        # Log a completion sample trace
        self.log_completion_sample(
            125,
            DRY_RUN_COMPLETION_PROMPT,
            DRY_RUN_COMPLETION_TEXT,
        )
        logger.info("[PASS] Completion trace logged")

        # Generate markdown report
        self.generate_report(DRY_RUN_FINAL_LOGS)
        logger.info("[PASS] Markdown report generated")

        # Finish
        self.finish()
        logger.info("[PASS] TrackingManager finished")
        logger.info("=== DRY RUN COMPLETE ===")
