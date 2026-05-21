import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

class TestConvertFullDataset:
    """Test full parquet conversion and SFT/DPO split."""

    def test_split_counts(self):
        """Test that split produces 1250 SFT + 250 DPO = 1500 total."""
        from src.teacher.convert_full_dataset import split_sft_dpo

        records = [{"id": i, "type": "A"} for i in range(1500)]
        sft, dpo = split_sft_dpo(records)

        assert len(sft) == 1250
        assert len(dpo) == 250
        assert len(sft) + len(dpo) == 1500

    def test_no_id_overlap(self):
        """Test that SFT and DPO sets share no question IDs."""
        from src.teacher.convert_full_dataset import split_sft_dpo

        records = [{"id": i, "type": chr(65 + (i % 5))} for i in range(1500)]
        sft, dpo = split_sft_dpo(records)

        sft_ids = {r["id"] for r in sft}
        dpo_ids = {r["id"] for r in dpo}
        assert len(sft_ids & dpo_ids) == 0

    def test_dpo_type_balance(self):
        """Test that DPO set has reasonable type distribution."""
        from src.teacher.convert_full_dataset import split_sft_dpo

        records = [{"id": i, "type": chr(65 + (i % 5))} for i in range(1500)]
        sft, dpo = split_sft_dpo(records)

        from collections import Counter
        type_counts = Counter(r["type"] for r in dpo)
        # Each type should appear at least once
        for t in "ABCDE":
            assert type_counts.get(t, 0) > 0, f"Type {t} missing from DPO set"


class TestBuildSFTDataset:
    """Test trace-aligned SFT dataset construction."""

    def test_sample_has_thought_tags(self):
        """Test that assistant response contains thought tags."""
        from src.teacher.build_sft_dataset import build_sft_sample

        record = {
            "question": "Why is income inequality increasing?",
            "answer__reasoning_content": "Material conditions show capital accumulation.",
            "answer": "### Structural Analysis\n**Material Conditions**\nTest.\n### Synthesis\nIncome inequality reflects structural forces.",
        }
        sample = build_sft_sample(record)

        assistant_content = sample["conversations"][1]["content"]
        assert "<thought>" in assistant_content
        assert "</thought>" in assistant_content

    def test_reasoning_trace_in_thought_block(self):
        """Test that cleaned reasoning trace appears inside thought block."""
        from src.teacher.build_sft_dataset import build_sft_sample

        trace = "### Material Conditions\nCapital controls resources."
        record = {
            "question": "Test question?",
            "answer__reasoning_content": trace,
            "answer": "### Structural Analysis\n**Material Conditions**\nTest.\n### Synthesis\nFull answer here.",
        }
        sample = build_sft_sample(record)

        assistant_content = sample["conversations"][1]["content"]
        assert trace in assistant_content

    def test_user_message_is_question_only(self):
        """Test that user message contains only the question."""
        from src.teacher.build_sft_dataset import build_sft_sample

        record = {
            "question": "Why are housing prices rising?",
            "answer__reasoning_content": "trace",
            "answer": "answer",
        }
        sample = build_sft_sample(record)

        assert sample["conversations"][0]["role"] == "user"
        assert sample["conversations"][0]["content"] == "Why are housing prices rising?"

    def test_no_metadata_in_sample(self):
        """Test that axis1, axis2, type metadata is excluded from conversations."""
        from src.teacher.build_sft_dataset import build_sft_sample

        record = {
            "id": 1,
            "type": "A",
            "axis1": ["B1"],
            "axis2": ["EP6"],
            "question": "Test?",
            "answer__reasoning_content": "trace",
            "answer": "answer",
        }
        sample = build_sft_sample(record)

        user_content = sample["conversations"][0]["content"]
        assert "B1" not in user_content
        assert "EP6" not in user_content
        assert "axis1" not in user_content


class TestRejectedResponses:
    """Test rejected response generation."""

    def test_liberal_default_generation(self):
        """Test that liberal default responses are substantive."""
        from src.teacher.generate_rejected_responses import generate_liberal_default

        response = generate_liberal_default("Why is income inequality increasing?")
        assert len(response) > 100
        assert "individual" in response.lower() or "market" in response.lower() or "policy" in response.lower()

    def test_jargon_trap_generation(self):
        """Test that jargon trap responses use DM terms without structural rigor."""
        from src.teacher.generate_rejected_responses import generate_jargon_trap

        response = generate_jargon_trap("Why is income inequality increasing?")
        assert len(response) > 100

    def test_shallow_dm_generation(self):
        """Test that shallow DM responses are lazy/incomplete."""
        from src.teacher.generate_rejected_responses import generate_shallow_dm

        response = generate_shallow_dm("Why is income inequality increasing?")
        assert len(response) > 100

    def test_all_three_differ(self):
        """Test that all three rejection types produce different responses."""
        from src.teacher.generate_rejected_responses import (
            generate_liberal_default,
            generate_jargon_trap,
            generate_shallow_dm,
        )

        q = "Why is income inequality increasing?"
        r1 = generate_liberal_default(q)
        r2 = generate_jargon_trap(q)
        r3 = generate_shallow_dm(q)

        assert r1 != r2
        assert r1 != r3
        assert r2 != r3
