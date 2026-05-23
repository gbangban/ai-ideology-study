import pytest


class TestGRPOTraining:
    def test_train_function_exists(self):
        from src.student.train_grpo import train
        assert callable(train)

    def test_main_function_exists(self):
        from src.student.train_grpo import main
        assert callable(main)

    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.student.train_grpo", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "base-model" in result.stdout or "base_model" in result.stdout


class TestGRPOIntegration:
    def test_full_pipeline_imports(self):
        """Test that all GRPO components can be imported together."""
        from src.student.grpo_config import GRPO_CONFIG
        from src.student.rewards import build_reward_fn, compute_directional_assertion, compute_format_reward, compute_length_reward
        from src.student.train_grpo import train, load_questions, format_prompt
        assert True

    def test_reward_pipeline(self):
        """Test reward computation pipeline end-to-end."""
        from src.student.rewards import compute_directional_assertion, compute_format_reward, compute_length_reward
        from src.student.grpo_config import GRPO_CONFIG

        text = "The policy directly causes positive change.\n\nMaterial conditions drive outcomes.\n\nPower relationships are key to understanding this dynamic."
        da = compute_directional_assertion(text)
        fr = compute_format_reward(text)
        lr = compute_length_reward(300)

        weights = GRPO_CONFIG["reward_weights"]
        total = weights["directional_assertion"] * da + weights["format"] * fr + weights["length"] * lr
        assert 0 <= total <= 1.0
        assert total > 0.1
