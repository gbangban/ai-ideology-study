import os
import subprocess
import tempfile

import pytest


def _import_train_module():
    """Import train_grpo_process, skipping on host numpy incompatibility."""
    try:
        from src.student import train_grpo_process
        return train_grpo_process
    except (ValueError, RuntimeError) as e:
        if "numpy.dtype size changed" in str(e):
            pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
        raise


class TestGRPOProcessTraining:
    def test_train_function_exists(self):
        mod = _import_train_module()
        assert callable(mod.train)

    def test_main_function_exists(self):
        mod = _import_train_module()
        assert callable(mod.main)

    def test_cli_help(self):
        result = subprocess.run(
            ["python3", "-m", "src.student.train_grpo_process", "--help"],
            capture_output=True, text=True,
        )
        if "numpy.dtype size changed" in result.stderr:
            pytest.skip("Host numpy/pandas binary incompatibility")
        assert result.returncode == 0
        assert "base-model" in result.stdout or "base_model" in result.stdout

    def test_build_reward_funcs_returns_two_fns(self):
        """v4 has two reward functions: outcome and process."""
        mod = _import_train_module()
        reward_funcs = mod._build_reward_funcs()
        assert len(reward_funcs) == 2

    def test_outcome_reward_fn_returns_floats(self):
        mod = _import_train_module()
        reward_funcs = mod._build_reward_funcs()
        completions = [
            "<planning>P</planning><commitment>+</commitment><reflection>R</reflection><monitor>M</monitor> The predicted sign is +",
            "The predicted sign is -",
        ]
        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
            {"dataset_type": "econcausal", "answer": "+"},
        ]
        results = reward_funcs[0](completions, docs)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, (int, float)) for r in results)

    def test_process_reward_fn_returns_floats(self):
        mod = _import_train_module()
        reward_funcs = mod._build_reward_funcs()
        completions = [
            "<planning>Treatment and outcome variables.</planning><commitment>+</commitment><reflection>I think my analysis is correct.</reflection><monitor>Context check.</monitor>",
            "Plain answer without tags.",
        ]
        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
            {"dataset_type": "econcausal", "answer": "+"},
        ]
        results = reward_funcs[1](completions, docs)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, (int, float)) for r in results)

    def test_process_reward_rewards_full_tags(self):
        mod = _import_train_module()
        reward_funcs = mod._build_reward_funcs()
        completions = [
            "<planning>Treatment: rates. Outcome: debt.</planning><commitment>positive (+)</commitment><reflection>I should reconsider.</reflection><monitor>Context alignment.</monitor>",
        ]
        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
        ]
        results = reward_funcs[1](completions, docs)
        assert results[0] > 0.0

    def test_process_reward_penalizes_missing_tags(self):
        mod = _import_train_module()
        reward_funcs = mod._build_reward_funcs()
        completions = [
            "Plain answer without any tags at all.",
        ]
        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
        ]
        results = reward_funcs[1](completions, docs)
        assert results[0] < 0.0

    def test_trl_reward_wrappers_accept_extra_args(self):
        mod = _import_train_module()
        doc_index = {
            "prompt-1": {"dataset_type": "econcausal", "answer": "+"},
        }
        outcome_fn, process_fn = mod._build_trl_reward_fns(doc_index)

        completion = "<planning>P</planning><commitment>+</commitment><reflection>R</reflection><monitor>M</monitor> The predicted sign is +"
        outcome_results = outcome_fn([completion], ["prompt-1"], [[1, 1]])
        process_results = process_fn([completion], ["prompt-1"], [[1, 1]])

        assert len(outcome_results) == 1
        assert len(process_results) == 1
        assert outcome_results[0] == 1.0

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
