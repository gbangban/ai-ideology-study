import pytest


class TestPlanningReward:
    def test_planning_tag_present(self):
        from src.student.rewards import compute_planning_reward
        text = "<planning>I need to identify treatment, outcome, and context.</planning>Answer."
        assert compute_planning_reward(text) > 0.0

    def test_no_planning_tag(self):
        from src.student.rewards import compute_planning_reward
        assert compute_planning_reward("Just an answer.") == 0.0

    def test_planning_with_variables(self):
        from src.student.rewards import compute_planning_reward
        text = "<planning>Treatment: rates. Outcome: debt. Context: financialized.</planning>Positive."
        assert compute_planning_reward(text) >= 0.5

    def test_empty(self):
        from src.student.rewards import compute_planning_reward
        assert compute_planning_reward("") == 0.0


class TestCommitmentReward:
    def test_definitive_commitment(self):
        from src.student.rewards import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is positive (+).</commitment>") > 0.0

    def test_hedged_commitment(self):
        from src.student.rewards import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The effect is mixed and depends on context.</commitment>") < 0.0

    def test_no_tag(self):
        from src.student.rewards import compute_commitment_reward
        assert compute_commitment_reward("I think it might be positive.") == 0.0

    def test_null_commitment(self):
        from src.student.rewards import compute_commitment_reward
        assert compute_commitment_reward("<commitment>The directional effect is null (0).</commitment>") > 0.0


class TestMonitorReward:
    def test_monitor_with_context(self):
        from src.student.rewards import compute_monitor_reward
        text = "<monitor>My answer aligns with the financialized market context.</monitor>"
        assert compute_monitor_reward(text) > 0.0

    def test_monitor_no_context_ref(self):
        from src.student.rewards import compute_monitor_reward
        assert compute_monitor_reward("<monitor>Answer provided.</monitor>") == 0.0

    def test_no_tag(self):
        from src.student.rewards import compute_monitor_reward
        assert compute_monitor_reward("The answer is positive.") == 0.0


class TestFormatPenalty:
    def test_all_tags_no_penalty(self):
        from src.student.rewards import compute_format_penalty
        assert compute_format_penalty("<planning>P</planning><commitment>C</commitment>") == 0.0

    def test_missing_tags(self):
        from src.student.rewards import compute_format_penalty
        assert compute_format_penalty("<planning>P</planning>Answer.") < 0.0

    def test_no_tags(self):
        from src.student.rewards import compute_format_penalty
        assert compute_format_penalty("Plain answer.") <= -0.1
