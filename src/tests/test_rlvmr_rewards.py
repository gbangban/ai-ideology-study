import pytest


class TestPlanningReward:
    def test_planning_tag_present_success(self):
        from src.student.reward_process import compute_planning_reward
        text = "<planning>I need to identify treatment, outcome, and context.</planning>Answer."
        assert compute_planning_reward(text, success=True) > 0.0

    def test_planning_tag_present_fail(self):
        from src.student.reward_process import compute_planning_reward
        text = "<planning>I need to identify treatment, outcome, and context.</planning>Answer."
        assert compute_planning_reward(text, success=False) == 0.0

    def test_no_planning_tag(self):
        from src.student.reward_process import compute_planning_reward
        assert compute_planning_reward("Just an answer.", success=True) == 0.0

    def test_planning_with_variables(self):
        from src.student.reward_process import compute_planning_reward
        text = "<planning>Treatment: rates. Outcome: debt. Context: financialized.</planning>Positive."
        assert compute_planning_reward(text, success=True) >= 0.5

    def test_empty(self):
        from src.student.reward_process import compute_planning_reward
        assert compute_planning_reward("", success=True) == 0.0


class TestCommitmentReward:
    def test_definitive_commitment(self):
        from src.student.reward_process import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is positive (+).</commitment>") > 0.0

    def test_hedged_commitment(self):
        from src.student.reward_process import compute_commitment_reward
        # "mixed" matches both definitive and hedging patterns -> 0.0 (conflated)
        assert compute_commitment_reward("<commitment>The effect is mixed and depends on context.</commitment>") == 0.0

    def test_no_tag(self):
        from src.student.reward_process import compute_commitment_reward
        assert compute_commitment_reward("I think it might be positive.") == 0.0

    def test_null_commitment(self):
        from src.student.reward_process import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is null (0).</commitment>") > 0.0

    def test_positive_commitment(self):
        from src.student.reward_process import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is positive.</commitment>") == 1.0


class TestReflectionReward:
    def test_reflection_with_critique_success(self):
        from src.student.reward_process import compute_reflection_reward
        text = "<reflection>I should reconsider my initial analysis and look for alternative explanations.</reflection>"
        assert compute_reflection_reward(text, success=True) == 1.0

    def test_reflection_with_critique_fail(self):
        from src.student.reward_process import compute_reflection_reward
        text = "<reflection>I should reconsider my initial analysis.</reflection>"
        assert compute_reflection_reward(text, success=False) == 0.0

    def test_reflection_self_referential(self):
        from src.student.reward_process import compute_reflection_reward
        text = "<reflection>I think my reasoning is sound but I could explore further.</reflection>"
        assert compute_reflection_reward(text, success=True) == 0.5

    def test_no_reflection_tag(self):
        from src.student.reward_process import compute_reflection_reward
        assert compute_reflection_reward("No reflection here.", success=True) == 0.0

    def test_empty(self):
        from src.student.reward_process import compute_reflection_reward
        assert compute_reflection_reward("", success=True) == 0.0


class TestMonitorReward:
    def test_monitor_with_context(self):
        from src.student.reward_process import compute_monitor_reward
        text = "<monitor>My answer aligns with the financialized market context.</monitor>"
        assert compute_monitor_reward(text) > 0.0

    def test_monitor_with_constraint(self):
        from src.student.reward_process import compute_monitor_reward
        text = "<monitor>This assumes a specific constraint on capital mobility.</monitor>"
        assert compute_monitor_reward(text) > 0.0

    def test_monitor_no_context_ref(self):
        from src.student.reward_process import compute_monitor_reward
        assert compute_monitor_reward("<monitor>Answer provided.</monitor>") == 0.0

    def test_no_tag(self):
        from src.student.reward_process import compute_monitor_reward
        assert compute_monitor_reward("The answer is positive.") == 0.0


class TestFormatPenalty:
    def test_all_tags_no_penalty(self):
        from src.student.reward_process import compute_format_penalty
        text = "<planning>P</planning><commitment>C</commitment><reflection>R</reflection><monitor>M</monitor>"
        assert compute_format_penalty(text) == 0.0

    def test_missing_tags(self):
        from src.student.reward_process import compute_format_penalty
        text = "<planning>P</planning>Answer here. Some more text."
        penalty = compute_format_penalty(text)
        assert penalty < 0.0
        assert abs(penalty - (-0.3)) < 1e-9

    def test_no_tags(self):
        from src.student.reward_process import compute_format_penalty
        text = "Plain answer with enough characters."
        penalty = compute_format_penalty(text)
        assert penalty == -0.4

    def test_too_short_all_penalty(self):
        from src.student.reward_process import compute_format_penalty
        assert compute_format_penalty("short") == -0.4


class TestProcessRewardsAggregation:
    def test_success_conditional(self):
        from src.student.reward_process import compute_process_rewards
        text = "<planning>Treatment and outcome variables.</planning><commitment>+</commitment><reflection>I think my analysis is correct.</reflection><monitor>Context check.</monitor>"
        result = compute_process_rewards(text, outcome_reward=1.0)
        assert result["planning"] > 0.0
        assert result["reflection"] > 0.0

    def test_fail_conditional(self):
        from src.student.reward_process import compute_process_rewards
        text = "<planning>Treatment and outcome variables.</planning><commitment>+</commitment><reflection>I think my analysis is correct.</reflection><monitor>Context check.</monitor>"
        result = compute_process_rewards(text, outcome_reward=0.0)
        assert result["planning"] == 0.0
        assert result["reflection"] == 0.0
        assert result["commitment"] > 0.0
        assert result["monitor"] > 0.0

    def test_format_penalty_included(self):
        from src.student.reward_process import compute_process_rewards
        text = "<planning>P</planning><commitment>C</commitment><reflection>R</reflection><monitor>M</monitor>"
        result = compute_process_rewards(text, outcome_reward=1.0)
        assert "format_penalty" in result
        assert result["format_penalty"] == 0.0


class TestOutcomeReward:
    def test_econcausal_correct_json(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        reward = compute_outcome_reward(doc, '{"predicted_sign": "+"}')
        assert reward == 1.0

    def test_econcausal_correct_no_json(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        reward = compute_outcome_reward(doc, "The predicted sign is +")
        assert reward == 0.9

    def test_econcausal_wrong_no_signal(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        reward = compute_outcome_reward(doc, "The predicted sign is -")
        assert reward == 0.0

    def test_econcausal_wrong_with_reasoning(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        reward = compute_outcome_reward(
            doc, "The predicted sign is -. This directly causes a change because structural factors drive the outcome."
        )
        assert 0.0 < reward <= 0.3

    def test_corr2cause_entailment(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "entailment"}
        assert compute_outcome_reward(doc, "True") == 0.9

    def test_corr2cause_contradiction(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "contradiction"}
        assert compute_outcome_reward(doc, "False") == 0.9

    def test_corr2cause_neutral(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "neutral"}
        assert compute_outcome_reward(doc, "True") == 1.0

    def test_null_effect(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"category": "null_effect"}
        assert compute_outcome_reward(doc, "The effect is None.") == 0.9

    def test_synthetic_fallback(self):
        from src.student.reward_outcome import compute_outcome_reward
        doc = {"dataset_type": "synthetic", "category": "context_flip"}
        reward = compute_outcome_reward(doc, "This directly causes a structural change through accumulation.")
        assert -0.5 <= reward <= 0.5


class TestReasoningQuality:
    def test_structured_reasoning(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "First, we identify the treatment. Therefore, the conclusion is clear because the mechanism implies causation."
        score = compute_reasoning_quality(text)
        assert score > 0.0
        assert score <= 0.5

    def test_dialectical_engagement(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "However, the counterexample shows the opposite. Conversely, another interpretation follows."
        score = compute_reasoning_quality(text)
        assert score > 0.0

    def test_hedging_penalty(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "First, it depends on various factors. The outcome is mixed and ambiguous. However, we can conclude."
        score = compute_reasoning_quality(text)
        assert score >= 0.0
        assert score <= 0.5

    def test_empty_text(self):
        from src.student.reward_outcome import compute_reasoning_quality
        assert compute_reasoning_quality("") == 0.0

    def test_short_text(self):
        from src.student.reward_outcome import compute_reasoning_quality
        assert compute_reasoning_quality("Hi there.") == 0.0

    def test_no_signal(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "This is a plain answer with no reasoning markers at all. Just a simple statement."
        score = compute_reasoning_quality(text)
        assert score == 0.0

    def test_max_possible(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "First step: because X implies Y. However, conversely Z. The conclusion follows."
        score = compute_reasoning_quality(text)
        assert score <= 0.5

    def test_full_reasoning_bonus(self):
        from src.student.reward_outcome import compute_reasoning_quality
        text = "Step one: because A causes B. Therefore C follows. However, the counterexample shows D. In conclusion, E."
        score = compute_reasoning_quality(text)
        assert score >= 0.35


class TestLengthPenalty:
    def test_empty_text(self):
        from src.student.reward_outcome import compute_length_penalty
        assert compute_length_penalty("") == 0.0

    def test_below_target(self):
        from src.student.reward_outcome import compute_length_penalty
        text = " ".join(["word"] * 200)
        assert compute_length_penalty(text, target_len=300) == 0.0

    def test_at_target(self):
        from src.student.reward_outcome import compute_length_penalty
        text = " ".join(["word"] * 300)
        assert compute_length_penalty(text, target_len=300) == 0.0

    def test_slightly_over(self):
        from src.student.reward_outcome import compute_length_penalty
        text = " ".join(["word"] * 400)
        score = compute_length_penalty(text, target_len=300)
        assert -0.1 <= score < 0.0

    def test_capped_at_minus_01(self):
        from src.student.reward_outcome import compute_length_penalty
        text = " ".join(["word"] * 10000)
        score = compute_length_penalty(text, target_len=300)
        assert score >= -0.1


class TestBuildV3RewardFn:
    def test_returns_three_functions(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        assert len(fns) == 3

    def test_outcome_fn_correct(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        docs = [{"dataset_type": "corr2cause", "relation": "neutral"}]
        scores = fns[0](["True"], docs)
        assert scores[0] == 1.0

    def test_reasoning_fn_bounded(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        docs = [{}]
        scores = fns[1](["First, because X implies Y. However, the conclusion is clear."], docs)
        assert 0.0 <= scores[0] <= 0.5

    def test_length_fn_zero_at_target(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        docs = [{}]
        short_text = " ".join(["word"] * 200)
        scores = fns[2]([short_text], docs)
        assert scores[0] == 0.0

    def test_length_fn_penalty_over_target(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        docs = [{}]
        long_text = " ".join(["word"] * 600)
        scores = fns[2]([long_text], docs)
        assert -0.1 <= scores[0] < 0.0

    def test_length_fn_capped(self):
        from src.student.reward_outcome import build_v3_reward_fn
        fns = build_v3_reward_fn()
        docs = [{}]
        very_long_text = " ".join(["word"] * 10000)
        scores = fns[2]([very_long_text], docs)
        assert scores[0] >= -0.1
