import pytest
from src.student.rewards import (
    compute_directional_assertion,
    compute_format_reward,
    compute_length_reward,
    compute_reward,
    build_reward_fn,
)


class TestDirectionalAssertion:
    def test_definitive_stance_scored_high(self):
        text = "The policy directly causes a net positive effect on worker outcomes."
        score = compute_directional_assertion(text)
        assert score > 0.5

    def test_hedged_response_scored_low(self):
        text = "It depends on various factors. The outcome is mixed and ambiguous."
        score = compute_directional_assertion(text)
        assert score < 0.5

    def test_semantic_workaround_penalized(self):
        text = "The outcome is non-linear and conditional on structural variables."
        score = compute_directional_assertion(text)
        assert score < 0.5

    def test_empty_text_zero(self):
        score = compute_directional_assertion("")
        assert score == 0.0


class TestFormatReward:
    def test_multi_paragraph_scored_high(self):
        text = "First paragraph with substantive content that explains the structural causes of the phenomenon.\n\nSecond paragraph that traces the systemic contradictions and shows how the problem drives further inequality.\n\nThird paragraph that critiques the dominant frame and reaches a different conclusion."
        score = compute_format_reward(text)
        assert score >= 0.7

    def test_single_paragraph_scored_low(self):
        text = "This is a single paragraph with no line breaks."
        score = compute_format_reward(text)
        assert score < 0.7

    def test_short_text_scored_low(self):
        text = "Hi.\n\nBye.\n\nOk."
        score = compute_format_reward(text)
        assert score < 0.5


class TestLengthReward:
    def test_ultra_short_penalty(self):
        score = compute_length_reward(10)
        assert score == 0.0

    def test_reasonable_length_good(self):
        score = compute_length_reward(300)
        assert 0.5 < score <= 1.0

    def test_over_cap_no_bonus(self):
        score_500 = compute_length_reward(500)
        score_1000 = compute_length_reward(1000)
        assert score_1000 == 1.0
        assert score_1000 <= score_500 + 0.01


class TestComputeReward:
    def test_weighted_sum(self):
        completions = ["<antThinking>reasoning</antThinking>\n\nThe policy directly causes positive change.\n\nMaterial conditions drive outcomes.\n\nPower relationships are key."]
        scores = compute_reward(completions, weights={"directional_assertion": 1.0})
        assert len(scores) == 1
        assert scores[0] > 0


class TestBuildRewardFn:
    def test_returns_callable(self):
        fn = build_reward_fn({"dm_alignment": 0.5, "directional_assertion": 0.5}, None, None)
        assert callable(fn)
