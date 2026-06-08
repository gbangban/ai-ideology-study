import pytest


class TestGRPOTraining:
    def test_train_function_exists(self):
        from src.student.train_grpo_dm import train
        assert callable(train)

    def test_main_function_exists(self):
        from src.student.train_grpo_dm import main
        assert callable(main)

    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.student.train_grpo_dm", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "base-model" in result.stdout or "base_model" in result.stdout


class TestGRPOIntegration:
    def test_full_pipeline_imports(self):
        """Test that all GRPO components can be imported together."""
        from src.student.grpo_config_dm import create_grpo_config, REWARD_WEIGHTS, DEFAULT_CONFIG
        from src.student.reward_dm import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.train_grpo_dm import train, _build_reward_funcs, _build_dataset
        from trl import GRPOConfig
        # GRPOTrainer import triggers datasets->pandas which can fail on host Python
        # with numpy binary incompatibility. Skip the GRPOTrainer import in that case.
        try:
            from trl import GRPOTrainer
        except (ValueError, RuntimeError) as e:
            if "numpy.dtype size changed" in str(e):
                pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
                raise
        assert True

    def test_reward_pipeline(self):
        """Test reward computation pipeline end-to-end with v2 rewards."""
        from src.student.reward_dm import (
            compute_directional_assertion,
            compute_dm_keyword_alignment,
            compute_mechanism_commitment,
        )
        from src.student.grpo_config_dm import REWARD_WEIGHTS

        text = "Capital accumulation drives exploitation through reserve army expansion. This directly increases class inequality and is the primary driver of wage suppression."
        da = compute_directional_assertion(text)
        dm = compute_dm_keyword_alignment(text)
        mc = compute_mechanism_commitment(text)

        w = REWARD_WEIGHTS
        total = w["directional_assertion"] * da + w["dm_alignment"] * dm + w["mechanism_commitment"] * mc
        assert total > 0.1
        assert da > 0
        assert dm > 0
        assert mc > 0

    def test_reward_funcs_callable(self):
        """Test that reward functions accept List[str] and return List[float]."""
        from src.student.train_grpo_dm import _build_reward_funcs

        reward_funcs = _build_reward_funcs()
        assert len(reward_funcs) == 3

        completions = [
            "Capital drives exploitation through structural power. This directly increases inequality.",
            "The market is efficient and prices reflect supply and demand.",
        ]
        for func in reward_funcs:
            results = func(completions)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, (int, float)) for r in results)

    def test_find_latest_checkpoint(self):
        """Test checkpoint discovery."""
        import tempfile, os
        from src.student.train_grpo_dm import _find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            step, path = _find_latest_checkpoint(tmpdir)
            assert step == 0 and path == ""

            for s in [100, 200, 300]:
                os.makedirs(f"{tmpdir}/checkpoint-{s}")
            step, path = _find_latest_checkpoint(tmpdir)
            assert step == 300
            assert "checkpoint-300" in path


class TestDMKeywordAlignment:
    def test_full_score_three_categories(self):
        from src.student.reward_dm import compute_dm_keyword_alignment
        text = "Capital's accumulation of surplus value drives exploitation. The structural power relations takes for granted the commodification of labor."
        score = compute_dm_keyword_alignment(text)
        assert score == 1.0

    def test_partial_score_one_category(self):
        from src.student.reward_dm import compute_dm_keyword_alignment
        text = "Capital's accumulation drives the economic system forward. The market responds to supply and demand signals."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.5

    def test_zero_score_no_dm_patterns(self):
        from src.student.reward_dm import compute_dm_keyword_alignment
        text = "The market is efficient and prices reflect supply and demand. Consumers make rational choices."
        score = compute_dm_keyword_alignment(text)
        assert score == 0.0

    def test_frame_critique_category(self):
        from src.student.reward_dm import compute_dm_keyword_alignment
        text = "Mainstream analysis naturalizes market outcomes and renders invisible the ideological function of hegemonic discourse."
        score = compute_dm_keyword_alignment(text)
        assert score >= 0.5


class TestAsymmetricDirectionalAssertion:
    def test_committed_response_scores_positive(self):
        from src.student.reward_dm import compute_directional_assertion
        text = "The policy directly causes increases in wages. This is the primary driver of inequality."
        score = compute_directional_assertion(text)
        assert score > 0.0

    def test_hedging_response_scores_negative(self):
        from src.student.reward_dm import compute_directional_assertion
        text = "The effect is mixed and theoretically ambiguous. It depends on context and empirically heterogeneous outcomes."
        score = compute_directional_assertion(text)
        assert score < 0.0

    def test_balanced_response_near_zero(self):
        from src.student.reward_dm import compute_directional_assertion
        text = "The policy increases wages but the effect is mixed across sectors."
        score = compute_directional_assertion(text)
        assert -0.5 <= score <= 0.5

    def test_empty_text_returns_zero(self):
        from src.student.reward_dm import compute_directional_assertion
        assert compute_directional_assertion("") == 0.0
        assert compute_directional_assertion("hi") == 0.0

    def test_score_clipped_to_range(self):
        from src.student.reward_dm import compute_directional_assertion
        text = "mixed ambiguous uncertain depends both sides heterogeneous"
        score = compute_directional_assertion(text)
        assert -1.0 <= score <= 1.0


class TestMechanismCommitment:
    def test_mechanism_with_commitment_scores_positive(self):
        from src.student.reward_dm import compute_mechanism_commitment
        text = "Capital accumulation drives wage suppression through reserve army expansion. This directly increases exploitation."
        score = compute_mechanism_commitment(text)
        assert score > 0.0

    def test_mechanism_with_hedging_scores_negative(self):
        from src.student.reward_dm import compute_mechanism_commitment
        text = "Capital drives outcomes through market mechanisms, but the effect is mixed and depends on structural conditions."
        score = compute_mechanism_commitment(text)
        assert score < 0.0

    def test_no_mechanism_returns_zero(self):
        from src.student.reward_dm import compute_mechanism_commitment
        text = "The situation is complex and multifaceted. There are many factors at play."
        score = compute_mechanism_commitment(text)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        from src.student.reward_dm import compute_mechanism_commitment
        assert compute_mechanism_commitment("") == 0.0
