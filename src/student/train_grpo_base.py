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
    """Patch Unsloth's lm_head to return hidden states during GRPO log-prob forward.

    Fixes unslothai/unsloth#5121: Qwen3.5 GRPO text-only path returns logits (vocab dim)
    instead of hidden states (hidden dim), causing a matmul shape mismatch in the
    chunked selective log-softmax which then tries to project logits through lm_head again.

    Based on the fix in unslothai/unsloth#5898. When UNSLOTH_RETURN_HIDDEN_STATES=1,
    we short-circuit lm_head.forward to return its input tensor directly, so the
    model's forward yields .logits == hidden_states and the chunked log-softmax
    receives the correct hidden-dim tensor.
    """
    try:
        import trl
        # Get the model from the trainer - we need to patch lm_head on the actual model
        # This is called after model loading but before trainer creation.
        # We patch it by wrapping the lm_head module's forward method.
        original_init = trl.GRPOTrainer.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Patch lm_head to passthrough hidden states during log-prob computation
            _patch_lm_head_passthrough(self.model)

        trl.GRPOTrainer.__init__ = patched_init
        logger.info("Patched GRPOTrainer.__init__ for lm_head passthrough (fixes #5121)")
    except Exception as e:
        logger.warning(f"Could not patch GRPOTrainer.__init__: {e}")


def _patch_lm_head_passthrough(model) -> None:
    """Short-circuit lm_head to return input when UNSLOTH_RETURN_HIDDEN_STATES=1.

    Finds the lm_head module on the model and wraps its forward method.
    When the env flag is set, returns the input tensor directly instead of
    doing the vocab projection. The lm_head weight is untouched.
    """
    lm_head = None
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    elif hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if hasattr(base, "lm_head"):
            lm_head = base.lm_head
    if lm_head is None:
        return

    original_forward = lm_head.forward

    def passthrough_forward(*args, **kwargs):
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return args[0] if args else next(iter(kwargs.values()))
        return original_forward(*args, **kwargs)

    lm_head.forward = passthrough_forward
    logger.info("Installed lm_head passthrough (fixes unsloth #5121)")


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
            chat, tokenize=False, add_generation_prompt=True
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
