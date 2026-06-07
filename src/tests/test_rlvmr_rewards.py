import pytest


class TestPlanningReward:
    def test_planning_tag_present_success(self):
        from src.student.rewards_v3v4 import compute_planning_reward
        text = "<planning>I need to identify treatment, outcome, and context.</planning>Answer."
        assert compute_planning_reward(text, success=True) > 0.0

    def test_planning_tag_present_fail(self):
        from src.student.rewards_v3v4 import compute_planning_reward
        text = "<planning>I need to identify treatment, outcome, and context.</planning>Answer."
        assert compute_planning_reward(text, success=False) == 0.0

    def test_no_planning_tag(self):
        from src.student.rewards_v3v4 import compute_planning_reward
        assert compute_planning_reward("Just an answer.", success=True) == 0.0

    def test_planning_with_variables(self):
        from src.student.rewards_v3v4 import compute_planning_reward
        text = "<planning>Treatment: rates. Outcome: debt. Context: financialized.</planning>Positive."
        assert compute_planning_reward(text, success=True) >= 0.5

    def test_empty(self):
        from src.student.rewards_v3v4 import compute_planning_reward
        assert compute_planning_reward("", success=True) == 0.0


class TestCommitmentReward:
    def test_definitive_commitment(self):
        from src.student.rewards_v3v4 import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is positive (+).</commitment>") > 0.0

    def test_hedged_commitment(self):
        from src.student.rewards_v3v4 import compute_commitment_reward
        # "mixed" matches both definitive and hedging patterns -> 0.0 (conflated)
        assert compute_commitment_reward("<commitment>The effect is mixed and depends on context.</commitment>") == 0.0

    def test_no_tag(self):
        from src.student.rewards_v3v4 import compute_commitment_reward
        assert compute_commitment_reward("I think it might be positive.") == 0.0

    def test_null_commitment(self):
        from src.student.rewards_v3v4 import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is null (0).</commitment>") > 0.0

    def test_positive_commitment(self):
        from src.student.rewards_v3v4 import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is positive.</commitment>") == 1.0


class TestReflectionReward:
    def test_reflection_with_critique_success(self):
        from src.student.rewards_v3v4 import compute_reflection_reward
        text = "<reflection>I should reconsider my initial analysis and look for alternative explanations.</reflection>"
        assert compute_reflection_reward(text, success=True) == 1.0

    def test_reflection_with_critique_fail(self):
        from src.student.rewards_v3v4 import compute_reflection_reward
        text = "<reflection>I should reconsider my initial analysis.</reflection>"
        assert compute_reflection_reward(text, success=False) == 0.0

    def test_reflection_self_referential(self):
        from src.student.rewards_v3v4 import compute_reflection_reward
        text = "<reflection>I think my reasoning is sound but I could explore further.</reflection>"
        assert compute_reflection_reward(text, success=True) == 0.5

    def test_no_reflection_tag(self):
        from src.student.rewards_v3v4 import compute_reflection_reward
        assert compute_reflection_reward("No reflection here.", success=True) == 0.0

    def test_empty(self):
        from src.student.rewards_v3v4 import compute_reflection_reward
        assert compute_reflection_reward("", success=True) == 0.0


class TestMonitorReward:
    def test_monitor_with_context(self):
        from src.student.rewards_v3v4 import compute_monitor_reward
        text = "<monitor>My answer aligns with the financialized market context.</monitor>"
        assert compute_monitor_reward(text) > 0.0

    def test_monitor_with_constraint(self):
        from src.student.rewards_v3v4 import compute_monitor_reward
        text = "<monitor>This assumes a specific constraint on capital mobility.</monitor>"
        assert compute_monitor_reward(text) > 0.0

    def test_monitor_no_context_ref(self):
        from src.student.rewards_v3v4 import compute_monitor_reward
        assert compute_monitor_reward("<monitor>Answer provided.</monitor>") == 0.0

    def test_no_tag(self):
        from src.student.rewards_v3v4 import compute_monitor_reward
        assert compute_monitor_reward("The answer is positive.") == 0.0


class TestFormatPenalty:
    def test_all_tags_no_penalty(self):
        from src.student.rewards_v3v4 import compute_format_penalty
        text = "<planning>P</planning><commitment>C</commitment><reflection>R</reflection><monitor>M</monitor>"
        assert compute_format_penalty(text) == 0.0

    def test_missing_tags(self):
        from src.student.rewards_v3v4 import compute_format_penalty
        text = "<planning>P</planning>Answer here. Some more text."
        penalty = compute_format_penalty(text)
        assert penalty < 0.0
        assert abs(penalty - (-0.3)) < 1e-9

    def test_no_tags(self):
        from src.student.rewards_v3v4 import compute_format_penalty
        text = "Plain answer with enough characters."
        penalty = compute_format_penalty(text)
        assert penalty == -0.4

    def test_too_short_all_penalty(self):
        from src.student.rewards_v3v4 import compute_format_penalty
        assert compute_format_penalty("short") == -0.4


class TestProcessRewardsAggregation:
    def test_success_conditional(self):
        from src.student.rewards_v3v4 import compute_process_rewards
        text = "<planning>Treatment and outcome variables.</planning><commitment>+</commitment><reflection>I think my analysis is correct.</reflection><monitor>Context check.</monitor>"
        result = compute_process_rewards(text, outcome_reward=1.0)
        assert result["planning"] > 0.0
        assert result["reflection"] > 0.0

    def test_fail_conditional(self):
        from src.student.rewards_v3v4 import compute_process_rewards
        text = "<planning>Treatment and outcome variables.</planning><commitment>+</commitment><reflection>I think my analysis is correct.</reflection><monitor>Context check.</monitor>"
        result = compute_process_rewards(text, outcome_reward=0.0)
        assert result["planning"] == 0.0
        assert result["reflection"] == 0.0
        assert result["commitment"] > 0.0
        assert result["monitor"] > 0.0

    def test_format_penalty_included(self):
        from src.student.rewards_v3v4 import compute_process_rewards
        text = "<planning>P</planning><commitment>C</commitment><reflection>R</reflection><monitor>M</monitor>"
        result = compute_process_rewards(text, outcome_reward=1.0)
        assert "format_penalty" in result
        assert result["format_penalty"] == 0.0


class TestOutcomeReward:
    def test_econcausal_correct(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        assert compute_outcome_reward(doc, "The predicted sign is +") == 1.0

    def test_econcausal_wrong(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "econcausal", "answer": "+"}
        assert compute_outcome_reward(doc, "The predicted sign is -") == 0.0

    def test_corr2cause_entailment(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "entailment"}
        assert compute_outcome_reward(doc, "True") == 1.0

    def test_corr2cause_contradiction(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "contradiction"}
        assert compute_outcome_reward(doc, "False") == 1.0

    def test_corr2cause_neutral(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "corr2cause", "relation": "neutral"}
        assert compute_outcome_reward(doc, "True") == 1.0

    def test_null_effect(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"category": "null_effect"}
        assert compute_outcome_reward(doc, "The effect is None.") == 1.0

    def test_synthetic_fallback(self):
        from src.student.rewards_v3v4 import compute_outcome_reward
        doc = {"dataset_type": "synthetic", "category": "context_flip"}
        reward = compute_outcome_reward(doc, "This directly causes a structural change through accumulation.")
        assert -0.5 <= reward <= 0.5
