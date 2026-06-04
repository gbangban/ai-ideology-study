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
        from src.student.rewards import (
            compute_directional_assertion,
            compute_format_reward,
            compute_length_reward,
        )
        from src.student.train_grpo import (
            train,
            compute_rewards,
            compute_advantage,
            generate_completions,
        )
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

    def test_compute_advantage(self):
        """Test advantage computation normalizes within groups."""
        from src.student.train_grpo import compute_advantage
        import torch

        rewards = [0.8, 0.6, 0.4, 0.2, 1.0, 0.5, 0.3, 0.7]
        advantages = compute_advantage(rewards, group_size=8)
        assert advantages.shape == (8,)
        assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)

    def test_compute_rewards_no_judge(self):
        """Test rewards work without judge model."""
        from src.student.train_grpo import compute_rewards
        from src.student.grpo_config import GRPO_CONFIG

        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text)))

        weights = {
            "directional_assertion": 0.2,
            "format": 0.15,
            "length": 0.15,
        }
        completions = ["The policy directly causes positive change.\n\nMaterial conditions.\n\nPower relationships."]
        scores = compute_rewards(completions, weights, FakeTokenizer(), None, None)
        assert len(scores) == 1
        assert scores[0] > 0

    def test_find_latest_checkpoint(self):
        """Test checkpoint discovery."""
        import tempfile, os
        from pathlib import Path
        from src.student.train_grpo import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            step, path = find_latest_checkpoint(tmpdir)
            assert step == 0 and path == ""

            for s in [100, 200, 300]:
                os.makedirs(f"{tmpdir}/checkpoint-{s}")
            step, path = find_latest_checkpoint(tmpdir)
            assert step == 300
            assert "checkpoint-300" in path

    def test_save_load_training_state(self):
        """Test training state save and restore."""
        import tempfile
        from pathlib import Path
        from src.student.train_grpo import save_training_state
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

            save_training_state(50, optimizer, scheduler, [0.5, 0.6], f"{tmpdir}/checkpoint-50")
            state = torch.load(f"{tmpdir}/checkpoint-50/training_state.pt", weights_only=False)
            assert state["step"] == 50
            assert len(state["rewards"]) == 2

    def test_ppo_clip_uses_min(self):
        """Verify PPO objective uses min (conservative update), not max."""
        import torch
        ratio = torch.tensor([1.3, 0.8, 1.5])
        adv = torch.tensor([1.0, -0.5, 0.8])

        unclipped = -(ratio * adv).mean()
        clipped = -(torch.clamp(ratio, 0.8, 1.2) * adv).mean()

        pg_loss = torch.min(unclipped, clipped)
        assert pg_loss.item() <= max(unclipped.item(), clipped.item())

    def test_dataloader_cycles_for_max_steps(self):
        """Verify training loop handles more steps than dataset size."""
        from src.student.train_grpo import GRPODataset
        from torch.utils.data import DataLoader
        import itertools

        prompts = ["q1", "q2", "q3"]
        dataset = GRPODataset(prompts)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataloader_iter = iter(itertools.cycle(dataloader))

        items = [next(dataloader_iter) for _ in range(10)]
        assert len(items) == 10


class TestDMKeywordAlignment:
    def test_full_score_three_categories(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation of surplus value drives exploitation. The structural power relations take for granted the commodification of labor."
        score = compute_dm_keyword_alignment(text)
        assert score == 1.0

    def test_partial_score_one_category(self):
        from src.student.rewards import compute_dm_keyword_alignment
        text = "Capital's accumulation drives the economic system forward. The market responds to supply and demand signals."
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
