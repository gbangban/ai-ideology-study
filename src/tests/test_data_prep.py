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
