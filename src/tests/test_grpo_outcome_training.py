import os
import subprocess
import tempfile

import pytest


def _import_train_module():
    """Import train_grpo_outcome, skipping on host numpy incompatibility."""
    try:
        from src.student import train_grpo_outcome
        return train_grpo_outcome
    except (ValueError, RuntimeError) as e:
        if "numpy.dtype size changed" in str(e):
            pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
        raise


class TestGRPOOutcomeTraining:
    def test_train_function_exists(self):
        mod = _import_train_module()
        assert callable(mod.train)

    def test_main_function_exists(self):
        mod = _import_train_module()
        assert callable(mod.main)

    def test_cli_help(self):
        result = subprocess.run(
            ["python3", "-m", "src.student.train_grpo_outcome", "--help"],
            capture_output=True, text=True,
        )
        if "numpy.dtype size changed" in result.stderr:
            pytest.skip("Host numpy/pandas binary incompatibility")
        assert result.returncode == 0
        assert "base-model" in result.stdout or "base_model" in result.stdout

    def test_get_reward_specs_returns_single_spec(self):
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        assert len(specs) == 1
        assert specs[0][0] == "outcome"

    def test_reward_spec_returns_floats(self):
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        raw_fn = specs[0][1]
        completions = [
            "The predicted sign is +",
            "The predicted sign is -",
        ]
        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
            {"dataset_type": "econcausal", "answer": "+"},
        ]
        results = raw_fn(completions, docs)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, (int, float)) for r in results)

    def test_outcome_reward_correct_answer(self):
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        raw_fn = specs[0][1]
        completions = ["The predicted sign is +"]
        docs = [{"dataset_type": "econcausal", "answer": "+"}]
        results = raw_fn(completions, docs)
        assert results[0] == 1.0

    def test_outcome_reward_wrong_answer(self):
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        raw_fn = specs[0][1]
        completions = ["The predicted sign is -"]
        docs = [{"dataset_type": "econcausal", "answer": "+"}]
        results = raw_fn(completions, docs)
        assert results[0] == 0.0

    def test_corr2cause_reward(self):
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        raw_fn = specs[0][1]
        completions = ["True"]
        docs = [{"dataset_type": "corr2cause", "relation": "entailment"}]
        results = raw_fn(completions, docs)
        assert results[0] == 1.0

    def test_reward_wrapper_accepts_extra_args(self):
        """Verify the TrackingManager-wrapped reward handles TRL's extra args."""
        from src.student.train_grpo_base import TrackingManager, build_reward_fn_with_docs
        mod = _import_train_module()
        specs = mod._get_reward_specs()
        doc_index = {
            "prompt-1": {"dataset_type": "econcausal", "answer": "+"},
            "prompt-2": {"dataset_type": "corr2cause", "relation": "entailment"},
        }
        raw_fn = specs[0][1]
        trl_fn = build_reward_fn_with_docs(raw_fn, doc_index)
        results = trl_fn(
            ["The predicted sign is +", "True"],
            ["prompt-1", "prompt-2"],
            [[1, 1, 1], [1, 1, 1]],
        )
        assert len(results) == 2
        assert results[0] == 1.0
        assert results[1] == 1.0

    def test_find_latest_checkpoint(self):
        mod = _import_train_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            step, path = mod._find_latest_checkpoint(tmpdir)
            assert step == 0 and path == ""

            for s in [100, 200, 300]:
                os.makedirs(f"{tmpdir}/checkpoint-{s}")
            step, path = mod._find_latest_checkpoint(tmpdir)
            assert step == 300
            assert "checkpoint-300" in path
