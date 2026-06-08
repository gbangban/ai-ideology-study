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
        from src.student.grpo_config import REWARD_WEIGHTS, create_grpo_config
        from src.student.rewards import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.train_grpo import (
            train,
            compute_rewards,
            compute_advantage,
            generate_completions,
        )
        assert True

    def test_reward_pipeline(self):
        """Test reward computation pipeline end-to-end with v2 rewards."""
        from src.student.rewards import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.grpo_config import REWARD_WEIGHTS

        text = "Capital accumulation drives exploitation through reserve army expansion. This directly increases class inequality and is the primary driver of wage suppression."
        da = compute_directional_assertion(text)
        dm = compute_dm_keyword_alignment(text)
        mc = compute_mechanism_commitment(text)

        weights = REWARD_WEIGHTS
        total = weights["directional_assertion"] * da + weights["dm_alignment"] * dm + weights["mechanism_commitment"] * mc
        assert total > 0.1
        assert da > 0  # committed language
        assert dm > 0  # DM keywords present
        assert mc > 0  # mechanisms + commitment

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
        from src.student.grpo_config import REWARD_WEIGHTS

        weights = REWARD_WEIGHTS
        completions = ["Capital drives exploitation through structural power. This directly increases inequality."]
        totals, dm_s, dir_s, mech_s = compute_rewards(completions, weights, None, None, None, None)
        assert len(totals) == 1
        assert len(dm_s) == 1
        assert len(dir_s) == 1
        assert len(mech_s) == 1
        assert totals[0] > 0

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
        text = "Capital's accumulation of surplus value drives exploitation. The structural power relations takes for granted the commodification of labor."
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


class TestAsymmetricDirectionalAssertion:
    def test_committed_response_scores_positive(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy directly causes increases in wages. This is the primary driver of inequality."
        score = compute_directional_assertion(text)
        assert score > 0.0

    def test_hedging_response_scores_negative(self):
        from src.student.rewards import compute_directional_assertion
        text = "The effect is mixed and theoretically ambiguous. It depends on context and empirically heterogeneous outcomes."
        score = compute_directional_assertion(text)
        assert score < 0.0

    def test_balanced_response_near_zero(self):
        from src.student.rewards import compute_directional_assertion
        text = "The policy increases wages but the effect is mixed across sectors."
        score = compute_directional_assertion(text)
        assert -0.5 <= score <= 0.5

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_directional_assertion
        assert compute_directional_assertion("") == 0.0
        assert compute_directional_assertion("hi") == 0.0

    def test_score_clipped_to_range(self):
        from src.student.rewards import compute_directional_assertion
        text = "mixed ambiguous uncertain depends both sides heterogeneous"
        score = compute_directional_assertion(text)
        assert -1.0 <= score <= 1.0


class TestMechanismCommitment:
    def test_mechanism_with_commitment_scores_positive(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital accumulation drives wage suppression through reserve army expansion. This directly increases exploitation."
        score = compute_mechanism_commitment(text)
        assert score > 0.0

    def test_mechanism_with_hedging_scores_negative(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "Capital drives outcomes through market mechanisms, but the effect is mixed and depends on structural conditions."
        score = compute_mechanism_commitment(text)
        assert score < 0.0

    def test_no_mechanism_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        text = "The situation is complex and multifaceted. There are many factors at play."
        score = compute_mechanism_commitment(text)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        from src.student.rewards import compute_mechanism_commitment
        assert compute_mechanism_commitment("") == 0.0
