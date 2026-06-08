def test_grpo_config_factory():
    from src.student.grpo_config import create_grpo_config
    from trl import GRPOConfig

    config = create_grpo_config(output_dir="/tmp/test-grpo")
    assert isinstance(config, GRPOConfig)
    assert config.num_generations == 8
    assert config.beta == 0.1
    assert config.learning_rate == 5e-7
    assert config.max_steps == 500
    assert config.warmup_steps == 50
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert config.max_completion_length == 512
    assert config.epsilon == 0.2
    assert config.loss_type == "dapo"
    assert config.scale_rewards == "group"
    assert config.logging_steps == 25
    assert config.save_steps == 50
    assert config.lr_scheduler_type == "cosine"
    assert config.max_prompt_length == 2048
    assert config.report_to == ["wandb"]
    assert config.generation_batch_size == 8


def test_reward_weights_sum_to_one():
    from src.student.grpo_config import REWARD_WEIGHTS
    assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-6


def test_reward_weights_has_expected_keys():
    from src.student.grpo_config import REWARD_WEIGHTS
    assert set(REWARD_WEIGHTS.keys()) == {"dm_alignment", "directional_assertion", "mechanism_commitment"}
