"""Tests for epistemic prior utilities (reward_epistemic.py)."""

import pytest
from src.student.reward_epistemic import (
    BNR_MARGIN,
    BNR_SAFE_THRESHOLD,
    check_bnr_phase,
    compute_ternary_reward,
    compute_uncertainty_scaled_proxy,
    detect_spurious_keyword_firing,
    estimate_response_confidence,
)


class TestBNRPhaseCheck:
    """Test BNR phase boundary classification."""

    def test_well_above_threshold(self):
        result = check_bnr_phase(0.95, "correct_outcome")
        assert result["phase"] == "safe-above"
        assert result["is_safe_phase"] is True
        assert result["distance_from_threshold"] == pytest.approx(0.45)

    def test_well_below_threshold(self):
        result = check_bnr_phase(0.1, "wrong_outcome")
        assert result["phase"] == "safe-below"
        assert result["is_safe_phase"] is True

    def test_at_threshold_ambiguous(self):
        result = check_bnr_phase(0.5, "proxy_reward")
        assert result["phase"] == "ambiguous"
        assert result["is_safe_phase"] is False

    def test_near_threshold_ambiguous(self):
        result = check_bnr_phase(0.54, "near_boundary")
        assert result["phase"] == "ambiguous"
        assert result["is_safe_phase"] is False

    def test_just_past_margin(self):
        result = check_bnr_phase(BNR_SAFE_THRESHOLD + BNR_MARGIN + 0.001, "edge_case")
        assert result["phase"] == "safe-above"
        assert result["is_safe_phase"] is True

    def test_negative_reward(self):
        result = check_bnr_phase(-0.5, "hedge_penalty")
        assert result["phase"] == "safe-below"
        assert result["is_safe_phase"] is True


class TestUncertaintyScaledProxy:
    """Test uncertainty-aware proxy reward scaling."""

    def test_confident_signal_full_scale(self):
        # Strong, agreeing signals should have low uncertainty
        reward, uncertainty = compute_uncertainty_scaled_proxy(
            "The treatment clearly causes a positive effect through mechanism X.",
            directional_score=0.8,
            dm_score=0.7,
            mech_score=0.9,
        )
        assert uncertainty < 0.3
        assert abs(reward) > 0.1

    def test_sparse_signal_high_uncertainty(self):
        # Only one signal firing
        reward, uncertainty = compute_uncertainty_scaled_proxy(
            "Some vague text here about economics.",
            directional_score=0.1,
            dm_score=0.0,
            mech_score=0.0,
        )
        assert uncertainty > 0.5

    def test_disagreeing_signals_high_uncertainty(self):
        # Components pointing in different directions
        reward, uncertainty = compute_uncertainty_scaled_proxy(
            "Mixed signals in the text.",
            directional_score=0.9,
            dm_score=-0.8,
            mech_score=0.0,
        )
        assert uncertainty > 0.4

    def test_empty_text_max_uncertainty(self):
        reward, uncertainty = compute_uncertainty_scaled_proxy("")
        assert uncertainty == 1.0
        assert reward == 0.0

    def test_scale_reduces_with_uncertainty(self):
        # High uncertainty should reduce effective scale
        reward_low_unc, unc_low = compute_uncertainty_scaled_proxy(
            "Strong confident text with clear mechanisms.",
            directional_score=0.8, dm_score=0.8, mech_score=0.8,
        )
        reward_high_unc, unc_high = compute_uncertainty_scaled_proxy(
            "Weak text.",
            directional_score=0.05, dm_score=0.0, mech_score=0.0,
        )
        assert unc_high > unc_low
        # With same raw score, higher uncertainty yields smaller absolute reward


class TestConfidenceEstimation:
    """Test self-reported confidence estimation."""

    def test_high_confidence_language(self):
        text = "The effect is definitively positive. This clearly causes growth."
        conf = estimate_response_confidence(text)
        assert conf > 0.5

    def test_low_confidence_language(self):
        text = "The effect may be positive, but it depends on context. It could go either way."
        conf = estimate_response_confidence(text)
        assert conf < 0.5

    def test_mixed_confidence(self):
        text = "The effect is clearly positive, but may depend on other factors."
        conf = estimate_response_confidence(text)
        assert 0.3 < conf < 0.7

    def test_no_confidence_signals(self):
        text = "The variable X has a relationship with variable Y in the dataset."
        conf = estimate_response_confidence(text)
        assert conf == 0.5

    def test_empty_text_default(self):
        conf = estimate_response_confidence("")
        assert conf == 0.5

    def test_very_short_text(self):
        conf = estimate_response_confidence("hi")
        assert conf == 0.5


class TestTernaryReward:
    """Test ternary reward decomposition from UCPO."""

    def test_high_conf_correct_full_reward(self):
        result = compute_ternary_reward(
            outcome_reward=0.9, process_reward=0.5, confidence=0.8
        )
        assert result["channel"] == "exploit"
        assert result["total"] == pytest.approx(1.4)
        assert result["high_conf_correct"] == pytest.approx(1.4)

    def test_low_conf_correct_partial_outcome(self):
        result = compute_ternary_reward(
            outcome_reward=0.9, process_reward=0.5, confidence=0.3
        )
        assert result["channel"] == "explore-structured"
        assert result["low_conf_correct"] == pytest.approx(0.45 + 0.5)

    def test_high_conf_incorrect_penalty(self):
        result = compute_ternary_reward(
            outcome_reward=0.0, process_reward=0.0, confidence=0.9
        )
        assert result["channel"] == "penalize-overconfidence"
        assert result["total"] < 0

    def test_low_conf_incorrect_mild_penalty(self):
        result = compute_ternary_reward(
            outcome_reward=0.0, process_reward=0.0, confidence=0.2
        )
        assert result["channel"] == "mild-penalty"
        assert result["total"] == pytest.approx(-0.1)

    def test_boundary_confidence(self):
        # Exactly at 0.5 confidence threshold
        result = compute_ternary_reward(0.9, 0.5, 0.5)
        assert result["channel"] == "exploit"


class TestSpuriousDetection:
    """Test spurious keyword firing detection."""

    def test_normal_firing(self):
        text = "The mode of production determines the surplus value extraction through exploitation."
        patterns = [r"\bexploitation\b", r"surplus\s+value", r"mode\s+of\s+production"]
        result = detect_spurious_keyword_firing(text, patterns)
        assert result["fired"] is True
        assert result["spurious"] is False
        assert result["match_count"] == 3

    def test_spurious_short_text(self):
        # Many keywords in very short text = likely spurious injection
        text = "exploitation surplus value"
        patterns = [r"\bexploitation\b", r"surplus\s+value", r"\bcommodification\b"]
        result = detect_spurious_keyword_firing(text, patterns)
        assert result["fired"] is True
        assert result["spurious"] is True

    def test_no_firing(self):
        text = "The sky is blue and the grass is green."
        patterns = [r"\bexploitation\b", r"surplus\s+value"]
        result = detect_spurious_keyword_firing(text, patterns)
        assert result["fired"] is False
        assert result["spurious"] is False
        assert result["match_count"] == 0

    def test_empty_text(self):
        result = detect_spurious_keyword_firing("", [r"\btest\b"])
        assert result["fired"] is False
        assert result["spurious"] is False
