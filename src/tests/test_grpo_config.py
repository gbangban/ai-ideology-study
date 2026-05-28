import pytest


class TestGRPOConfig:
    def test_config_exists(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert isinstance(GRPO_CONFIG, dict)

    def test_base_model_path(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert "base_model" in GRPO_CONFIG

    def test_lora_params(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert GRPO_CONFIG["lora_rank"] == 16
        assert GRPO_CONFIG["lora_alpha"] == 16
        assert GRPO_CONFIG["lora_dropout"] == 0.05

    def test_training_params(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert GRPO_CONFIG["learning_rate"] == 5e-7
        assert GRPO_CONFIG["max_steps"] == 500
        assert GRPO_CONFIG["warmup_steps"] == 50
        assert GRPO_CONFIG["per_device_train_batch_size"] == 1
        assert GRPO_CONFIG["gradient_accumulation_steps"] == 4

    def test_group_size(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert GRPO_CONFIG["grpo_g"] == 8

    def test_reward_weights_sum_to_one(self):
        from src.student.grpo_config import GRPO_CONFIG
        weights = GRPO_CONFIG["reward_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_reward_weights_positive(self):
        from src.student.grpo_config import GRPO_CONFIG
        weights = GRPO_CONFIG["reward_weights"]
        assert all(v > 0 for v in weights.values())

    def test_beta_kl_penalty(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert "beta" in GRPO_CONFIG
        assert GRPO_CONFIG["beta"] >= 0

    def test_max_completion_length(self):
        from src.student.grpo_config import GRPO_CONFIG
        assert GRPO_CONFIG["max_completion_length"] == 512

    def test_target_modules(self):
        from src.student.grpo_config import GRPO_CONFIG
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert GRPO_CONFIG["target_modules"] == expected
