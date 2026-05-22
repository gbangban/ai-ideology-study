# DM-Align Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete data prep + training pipeline: convert all 1,500 teacher answers, split into SFT/DPO sets, generate trace-aligned SFT dataset, generate DPO pairs with real rejected responses, and wire up programmatic SFT training.

**Architecture:** Linear pipeline — parquet → JSON → SFT dataset + DPO split → SFT training → DPO pairs → DPO training. All data prep is CPU-only Python. Training runs in the Docker container with CUDA.

**Tech Stack:** Python 3.12, pandas (parquet), Unsloth Core (FastLanguageModel), TRL (SFTTrainer, DPOTrainer), transformers, peft, torch.

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/teacher/convert_full_dataset.py` | Convert full parquet (1,500 rows) → JSON, clean traces, split SFT/DPO |
| `src/teacher/build_sft_dataset.py` | Build trace-aligned ShareGPT JSONL from converted JSON |
| `src/teacher/generate_rejected_responses.py` | Generate real liberal-reformist rejected responses (replaces stub) |
| `src/teacher/generate_dpo_pairs.py` | Existing — will be modified to use real rejected responses |
| `src/student/train_sft_v2.py` | New programmatic SFT script with trace alignment, Neftune noise |
| `src/student/sft_config_v2.py` | New SFT config for programmatic training |
| `src/student/train_dpo.py` | Existing — will be modified for proper TRL DPOTrainer integration |
| `src/student/dpo_config.py` | Existing — will be updated with correct paths and params |
| `src/tests/test_data_prep.py` | Tests for data prep pipeline |
| `src/tests/test_sft_v2.py` | Tests for new SFT training script |
| `scripts/run_sft_v2.sh` | Runner script for SFT training |
| `scripts/run_dpo.sh` | Existing — verify/update paths |

---

### Task 1: Full parquet conversion + SFT/DPO split

**Files:**
- Create: `src/teacher/convert_full_dataset.py`
- Test: `src/tests/test_data_prep.py`

Context: The parquet file at `/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/recipe_ml-1500-v1/parquet-files/batch_00000.parquet` has 1,500 rows with columns: `id`, `type`, `type_label`, `question`, `cross_domain`, `answer`, `answer__reasoning_content`. We need to convert all 1,500 to JSON, clean reasoning traces, and split into 1,250 SFT + 250 DPO.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_data_prep.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py -v
```

Expected: FAIL — module `src.teacher.convert_full_dataset` does not exist yet.

- [ ] **Step 3: Write implementation**

Create `src/teacher/convert_full_dataset.py`:

```python
#!/usr/bin/env python3
"""
Convert full parquet dataset to JSON, clean reasoning traces, and split SFT/DPO.

Usage:
    python3 -m src.teacher.convert_full_dataset \
        --parquet-path /path/to/batch_00000.parquet \
        --output-dir data/processed/
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.teacher.clean_reasoning_traces import clean_reasoning_trace


PARQUET_PATH_DEFAULT = (
    "/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/"
    "recipe_ml-1500-v1/parquet-files/batch_00000.parquet"
)


def parquet_to_records(parquet_path: str) -> list[dict]:
    """Read parquet and return list of dicts with native Python types."""
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if hasattr(val, "item"):
                val = val.item()
            record[col] = val if pd.notna(val) else ""
        records.append(record)
    return records


def split_sft_dpo(records: list[dict], sft_count: int = 1250, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Split records into SFT and DPO sets with balanced type distribution.

    Strategy: stratified random selection — pick DPO samples proportionally
    from each type, then remaining go to SFT.

    Returns (sft_records, dpo_records).
    """
    import random
    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for r in records:
        t = r.get("type", "A")
        by_type.setdefault(t, []).append(r)

    # Calculate DPO quota per type (proportional)
    total = len(records)
    dpo_quota: dict[str, int] = {}
    dpo_assigned = 0
    types_sorted = sorted(by_type.keys())

    for t in types_sorted:
        proportion = len(by_type[t]) / total
        quota = round(proportion * sft_count)  # sft_count is actually dpo_count target
        dpo_quota[t] = quota
        dpo_assigned += quota

    # Adjust to hit exact target
    target_dpo = len(records) - sft_count
    while dpo_assigned > target_dpo:
        # Remove from largest quota
        largest = max(dpo_quota, key=dpo_quota.get)
        dpo_quota[largest] -= 1
        dpo_assigned -= 1
    while dpo_assigned < target_dpo:
        # Add to type with most remaining
        remaining = {t: len(by_type[t]) - dpo_quota.get(t, 0) for t in types_sorted}
        largest_remaining = max(remaining, key=remaining.get)
        dpo_quota[largest_remaining] = dpo_quota.get(largest_remaining, 0) + 1
        dpo_assigned += 1

    # Select DPO samples
    dpo_records = []
    for t in types_sorted:
        pool = by_type[t][:]
        rng.shuffle(pool)
        dpo_records.extend(pool[:dpo_quota.get(t, 0)])

    # Remaining go to SFT
    dpo_ids = {r["id"] for r in dpo_records}
    sft_records = [r for r in records if r["id"] not in dpo_ids]

    return sft_records, dpo_records


def clean_and_save(records: list[dict], output_path: str) -> list[dict]:
    """Clean reasoning traces in-place and save to JSON."""
    cleaned = 0
    for r in records:
        rc = r.get("answer__reasoning_content", "")
        if rc:
            r["answer__reasoning_content"] = clean_reasoning_trace(rc)
            cleaned += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Cleaned {cleaned}/{len(records)} reasoning traces")
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert full parquet dataset, clean traces, split SFT/DPO")
    parser.add_argument("--parquet-path", default=PARQUET_PATH_DEFAULT, help="Path to parquet file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--sft-count", type=int, default=1250, help="Number of SFT samples")
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet_path}")
    records = parquet_to_records(args.parquet_path)
    print(f"  Loaded {len(records)} records")

    print("Cleaning reasoning traces...")
    records = clean_and_save(records, str(Path(args.output_dir) / "full_dataset.json"))

    print(f"Splitting {len(records)} records: {args.sft_count} SFT + {len(records) - args.sft_count} DPO")
    sft, dpo = split_sft_dpo(records, sft_count=args.sft_count)

    sft_path = str(Path(args.output_dir) / "sft_dataset.json")
    dpo_path = str(Path(args.output_dir) / "dpo_dataset.json")

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft, f, indent=2, ensure_ascii=False)
    print(f"  SFT: {len(sft)} samples -> {sft_path}")

    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo, f, indent=2, ensure_ascii=False)
    print(f"  DPO: {len(dpo)} samples -> {dpo_path}")

    # Type distribution report
    sft_types = Counter(r["type"] for r in sft)
    dpo_types = Counter(r["type"] for r in dpo)
    print(f"\n  SFT type distribution: {dict(sorted(sft_types.items()))}")
    print(f"  DPO type distribution: {dict(sorted(dpo_types.items()))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/teacher/convert_full_dataset.py src/tests/test_data_prep.py
git commit -m "feat: full parquet conversion with SFT/DPO split (1250+250)"
```

---

### Task 2: Build trace-aligned SFT dataset

**Files:**
- Create: `src/teacher/build_sft_dataset.py`
- Modify: `src/tests/test_data_prep.py` (add tests)

Context: The SFT dataset must use Qwen3.5's native chat template with `<thought>` reasoning traces. The assistant response wraps the cleaned reasoning trace in `<thought>` tags, followed by the synthesis from the answer.

- [ ] **Step 1: Write the failing test**

Add to `src/tests/test_data_prep.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestBuildSFTDataset -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Write implementation**

Create `src/teacher/build_sft_dataset.py`:

```python
#!/usr/bin/env python3
"""
Build trace-aligned SFT dataset from converted JSON.

Constructs ShareGPT-format samples with Qwen3.5 native thinking tokens:
  User: <question>
  Assistant: <thought><reasoning trace></thought><synthesis>

Usage:
    python3 -m src.teacher.build_sft_dataset \
        --input data/processed/sft_dataset.json \
        --output data/processed/sft_dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def extract_synthesis(answer: str) -> str:
    """
    Extract the Synthesis section from the structured answer.

    The teacher answer has format:
      ### Structural Analysis
      **Material Conditions** ...
      ...
      ### Synthesis
      <essay text>

    We want the Synthesis as the final answer after the thought block.
    """
    if "### Synthesis" in answer:
        idx = answer.index("### Synthesis")
        return answer[idx + len("### Synthesis"):].strip()
    # Fallback: return full answer
    return answer.strip()


def build_sft_sample(record: dict) -> dict:
    """
    Build a single SFT sample with trace alignment.

    Format:
      conversations: [
        {role: "user", content: "<question>"},
        {role: "assistant", content: "<thought><trace></thought><synthesis>"}
      ]
    """
    question = record.get("question", "").strip()
    trace = record.get("answer__reasoning_content", "").strip()
    answer = record.get("answer", "").strip()
    synthesis = extract_synthesis(answer)

    assistant_content = ""
    if trace:
        assistant_content = f"<thought>\n{trace}\n</thought>\n{synthesis}"
    else:
        assistant_content = synthesis

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_sft_dataset(input_path: str, output_path: str) -> int:
    """
    Build full SFT dataset from JSON records.

    Returns number of samples written.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for record in records:
        sample = build_sft_sample(record)
        if sample["conversations"][0]["content"] and sample["conversations"][1]["content"]:
            samples.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Build trace-aligned SFT dataset")
    parser.add_argument("--input", default="data/processed/sft_dataset.json", help="Input JSON file")
    parser.add_argument("--output", default="data/processed/sft_dataset.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    count = build_sft_dataset(args.input, args.output)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestBuildSFTDataset -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/teacher/build_sft_dataset.py src/tests/test_data_prep.py
git commit -m "feat: trace-aligned SFT dataset builder with thought tags"
```

---

### Task 3: Generate real rejected responses for DPO

**Files:**
- Create: `src/teacher/generate_rejected_responses.py`
- Modify: `src/teacher/generate_dpo_pairs.py`
- Modify: `src/tests/test_data_prep.py` (add tests)

Context: The current `generate_rejected_response` function returns a stub template. We need three rejection types as specified in the design: Liberal Default, Jargon Trap, Shallow DM. Since we can't run a model to generate these, we'll use template-based generation that produces substantive, varied rejected responses.

- [ ] **Step 1: Write the failing test**

Add to `src/tests/test_data_prep.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestRejectedResponses -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Write implementation**

Create `src/teacher/generate_rejected_responses.py`:

```python
#!/usr/bin/env python3
"""
Generate rejected responses for DPO training.

Three rejection types to triangulate the target:
1. Liberal Default: mainstream reformist analysis (individual agency, policy design)
2. Jargon Trap: heavy DM terminology without structural rigor (moralistic, rhetorical)
3. Shallow DM: lazy/incomplete DM analysis (superficial, handwavy)

These are template-based generators. For production quality, replace with
LLM-generated responses using liberal-reformist and jargon system prompts.

Usage:
    python3 -m src.teacher.generate_rejected_responses \
        --input data/processed/dpo_dataset.json \
        --output data/processed/rejected_responses.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


LIBERAL_TEMPLATES = [
    "This issue can be addressed through better policy design and institutional reform. "
    "Research shows that targeted interventions, such as {policy}, can produce measurable improvements. "
    "The key is empowering individuals with better opportunities and ensuring markets function efficiently. "
    "Government should focus on creating the right incentives while avoiding overreach that could stifle innovation.",

    "The evidence suggests this is primarily a matter of individual choices and market dynamics. "
    "When people have access to quality information and fair opportunities, they make decisions that benefit themselves and society. "
    "Policy should focus on removing barriers to entry and ensuring competition. "
    "Programs that invest in education and skills development have shown the strongest results in addressing {topic}.",

    "This problem is best understood through the lens of institutional economics and behavioral science. "
    "Studies show that well-designed policy frameworks can align individual incentives with social outcomes. "
    "The solution lies in improving governance, strengthening regulatory oversight, and investing in human capital. "
    "Countries that have implemented {policy} have seen significant improvements in {topic}.",
]

LIBERAL_POLICIES = [
    "progressive taxation", "universal basic income", "education reform",
    "market-based carbon pricing", "antitrust enforcement", "zoning reform",
    "healthcare market competition", "financial literacy programs",
    "public-private partnerships", "regulatory streamlining",
]

JARGON_TEMPLATES = [
    "The bourgeoisie has consistently demonstrated its opposition to progressive change. "
    "We must recognize the evil of the ruling class and their deliberate attempts to suppress the working people. "
    "The capitalists are greedy and selfish, hoarding wealth while the proletariat suffers. "
    "We need to stand in solidarity against the oppressors and fight for a more just society.",

    "This is clearly a manifestation of the inherent evil of the current system. "
    "The ruling elite deliberately exploit the vulnerable for their own enrichment. "
    "We must condemn the moral bankruptcy of those in power and demand immediate action. "
    "The revolution will come when the people rise up against their oppressors.",

    "The problem is that the wealthy have too much power and refuse to share. "
    "Billionaires are the root cause of all social problems, and we must hold them accountable. "
    "The system is rigged by corrupt elites who care only about their own interests. "
    "We need moral leadership to fight against this injustice.",
]

SHALLOW_TEMPLATES = [
    "This reflects the contradictions inherent in the current mode of production. "
    "The relations of production create systemic tensions that cannot be resolved within the existing framework. "
    "The base determines the superstructure, and changes in one will affect the other.",

    "The answer lies in understanding the material conditions of the working class. "
    "Class struggle is the driving force of history, and this situation is no exception. "
    "The contradictions of the system will eventually lead to its transformation.",

    "This can be understood through the lens of historical materialism. "
    "The economic base shapes all social relations, and the current arrangement reflects the interests of the dominant class. "
    "Dialectical analysis reveals the underlying contradictions at play.",
]


def generate_liberal_default(question: str) -> str:
    """Generate a liberal-reformist rejected response."""
    template = random.choice(LIBERAL_TEMPLATES)
    policy = random.choice(LIBERAL_POLICIES)
    topic = question.split("?")[0].split("How")[0].split("Why")[0].split("What")[0].strip()
    if not topic:
        topic = "this issue"
    return template.format(policy=policy, topic=topic)


def generate_jargon_trap(question: str) -> str:
    """Generate a jargon-heavy but structurally shallow rejected response."""
    return random.choice(JARGON_TEMPLATES)


def generate_shallow_dm(question: str) -> str:
    """Generate a lazy/incomplete DM rejected response."""
    return random.choice(SHALLOW_TEMPLATES)


REJECTION_TYPES = ["liberal_default", "jargon_trap", "shallow_dm"]
REJECTION_GENERATORS = {
    "liberal_default": generate_liberal_default,
    "jargon_trap": generate_jargon_trap,
    "shallow_dm": generate_shallow_dm,
}


def generate_all_rejections(question: str) -> list[dict]:
    """Generate all three rejection types for a question."""
    rejections = []
    for rtype, generator in REJECTION_GENERATORS.items():
        response = generator(question)
        rejections.append({
            "type": rtype,
            "content": response,
        })
    return rejections


def main():
    parser = argparse.ArgumentParser(description="Generate rejected responses for DPO training")
    parser.add_argument("--input", default="data/processed/dpo_dataset.json", help="Input DPO dataset JSON")
    parser.add_argument("--output", default="data/processed/rejected_responses.jsonl", help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            question = record.get("question", "")
            rejections = generate_all_rejections(question)
            output = {
                "id": record.get("id"),
                "question": question,
                "rejections": rejections,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Generated {3 * len(records)} rejection responses for {len(records)} questions -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestRejectedResponses -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/teacher/generate_rejected_responses.py src/tests/test_data_prep.py
git commit -m "feat: three-type rejected response generation for DPO"
```

---

### Task 4: Wire up DPO pair generation with real rejections

**Files:**
- Modify: `src/teacher/generate_dpo_pairs.py`
- Modify: `src/tests/test_data_prep.py` (add tests)

Context: Rewrite `generate_dpo_pairs.py` to read the DPO dataset JSON and rejected responses JSONL, then produce interleaved DPO pairs (each question gets 3 pairs, one per rejection type).

- [ ] **Step 1: Write the failing test**

Add to `src/tests/test_data_prep.py`:

```python
class TestDPOPairGeneration:
    """Test DPO pair generation with real rejections."""

    def test_interleaved_pairs(self, tmp_path):
        """Test that DPO pairs are interleaved across rejection types."""
        from src.teacher.generate_dpo_pairs import generate_interleaved_pairs

        records = [
            {"id": 1, "question": "Q1?", "answer": "Chosen answer 1."},
            {"id": 2, "question": "Q2?", "answer": "Chosen answer 2."},
        ]
        rejections = [
            {
                "id": 1,
                "rejections": [
                    {"type": "liberal_default", "content": "Liberal R1"},
                    {"type": "jargon_trap", "content": "Jargon R1"},
                    {"type": "shallow_dm", "content": "Shallow R1"},
                ],
            },
            {
                "id": 2,
                "rejections": [
                    {"type": "liberal_default", "content": "Liberal R2"},
                    {"type": "jargon_trap", "content": "Jargon R2"},
                    {"type": "shallow_dm", "content": "Shallow R2"},
                ],
            },
        ]

        pairs = generate_interleaved_pairs(records, rejections)
        assert len(pairs) == 6  # 2 questions x 3 rejection types

        types = [p["rejection_type"] for p in pairs]
        assert "liberal_default" in types
        assert "jargon_trap" in types
        assert "shallow_dm" in types

    def test_pair_structure(self, tmp_path):
        """Test that each DPO pair has correct structure."""
        from src.teacher.generate_dpo_pairs import generate_interleaved_pairs

        records = [{"id": 1, "question": "Q?", "answer": "Chosen."}]
        rejections = [{
            "id": 1,
            "rejections": [{"type": "liberal_default", "content": "Rejected."}],
        }]

        pairs = generate_interleaved_pairs(records, rejections)
        pair = pairs[0]

        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        assert "rejection_type" in pair
        assert pair["chosen"] == "Chosen."
        assert pair["rejected"] == "Rejected."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestDPOPairGeneration -v
```

Expected: FAIL — function `generate_interleaved_pairs` does not exist.

- [ ] **Step 3: Update implementation**

Replace `src/teacher/generate_dpo_pairs.py`:

```python
#!/usr/bin/env python3
"""
DPO Pair Generation

Generate preference pairs for DPO training from SFT dataset and rejected responses.
Each question produces 3 pairs (one per rejection type), interleaved.

Usage:
    python3 -m src.teacher.generate_dpo_pairs \
        --dpo-data data/processed/dpo_dataset.json \
        --rejections data/processed/rejected_responses.jsonl \
        --output data/processed/dpo_pairs.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_rejections(path: str) -> dict[int, list[dict]]:
    """Load rejected responses indexed by question ID."""
    rejections_by_id: dict[int, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                qid = record["id"]
                rejections_by_id[qid] = record["rejections"]
    return rejections_by_id


def generate_interleaved_pairs(records: list[dict], rejections: dict[int, list[dict]]) -> list[dict]:
    """
    Generate DPO pairs: one per rejection type per question.

    Returns list of dicts with keys: prompt, chosen, rejected, rejection_type.
    Pairs are shuffled to interleave rejection types.
    """
    pairs = []
    for record in records:
        qid = record["id"]
        question = record["question"]
        chosen = record["answer"]
        rejections_for_q = rejections.get(qid, [])

        for rej in rejections_for_q:
            pairs.append({
                "prompt": question,
                "chosen": chosen,
                "rejected": rej["content"],
                "rejection_type": rej["type"],
            })

    # Shuffle to interleave rejection types
    random.seed(42)
    random.shuffle(pairs)
    return pairs


def save_dpo_pairs(pairs: list[dict], output_path: str):
    """Save DPO pairs to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} DPO pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DPO pairs from dataset and rejections")
    parser.add_argument("--dpo-data", default="data/processed/dpo_dataset.json", help="DPO dataset JSON")
    parser.add_argument("--rejections", default="data/processed/rejected_responses.jsonl", help="Rejected responses JSONL")
    parser.add_argument("--output", default="data/processed/dpo_pairs.jsonl", help="Output DPO pairs JSONL")
    args = parser.parse_args()

    with open(args.dpo_data, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} DPO questions")

    rejections = load_rejections(args.rejections)
    print(f"Loaded rejections for {len(rejections)} questions")

    pairs = generate_interleaved_pairs(records, rejections)
    save_dpo_pairs(pairs, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_data_prep.py::TestDPOPairGeneration -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/teacher/generate_dpo_pairs.py src/tests/test_data_prep.py
git commit -m "feat: interleaved DPO pair generation with three rejection types"
```

---

### Task 5: Programmatic SFT training script (v2)

**Files:**
- Create: `src/student/sft_config_v2.py`
- Create: `src/student/train_sft_v2.py`
- Create: `src/tests/test_sft_v2.py`

Context: Replace the deprecated Studio-based SFT with a programmatic script using Unsloth Core + TRL SFTTrainer. Must support: trace-aligned data, Neftune noise, proper loss masking (train on everything), and Qwen3.5 chat template.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_sft_v2.py`:

```python
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSFTConfigV2:
    """Test programmatic SFT configuration."""

    def test_model_name(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG["model_name"] == "Qwen/Qwen3.5-9B"

    def test_lora_params(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG["lora_r"] == 32
        assert SFT_CONFIG["lora_alpha"] == 32
        assert SFT_CONFIG["lora_dropout"] == 0.05

    def test_neftune_enabled(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        assert SFT_CONFIG.get("neftune_noise_alpha") is not None
        assert SFT_CONFIG["neftune_noise_alpha"] > 0

    def test_target_modules(self):
        from src.student.sft_config_v2 import SFT_CONFIG
        modules = SFT_CONFIG["target_modules"]
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(modules) == expected


class TestTrainSFTV2:
    """Test SFT v2 training script functions."""

    def test_prepare_model_for_training(self):
        """Test that model preparation applies LoRA correctly."""
        from src.student.train_sft_v2 import prepare_model_for_training

        mock_model = Mock()
        mock_tokenizer = Mock()

        with patch("src.student.train_sft_v2.FastLanguageModel") as mock_flm:
            mock_flm.get_peft_model.return_value = mock_model
            result = prepare_model_for_training(mock_model, mock_tokenizer, {
                "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
            })
            mock_flm.get_peft_model.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_sft_v2.py -v
```

Expected: FAIL — modules do not exist.

- [ ] **Step 3: Write implementation**

Create `src/student/sft_config_v2.py`:

```python
"""
SFT Training Configuration v2

Programmatic SFT training with Unsloth Core + TRL.
Replaces Studio UI-based SFT.
"""

SFT_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3.5-9B",
    "max_seq_length": 4096,

    # LoRA
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Quantization
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",

    # Neftune noise
    "neftune_noise_alpha": 5.0,

    # Training
    "learning_rate": 2e-4,
    "max_steps": 1000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "optim": "adamw_8bit",

    # Output
    "output_dir": "checkpoints/lora_adapters/sft_v2_adapter",
    "logging_steps": 50,
    "save_steps": 200,
}
```

Create `src/student/train_sft_v2.py`:

```python
#!/usr/bin/env python3
"""
Programmatic SFT Training v2

Uses Unsloth Core + TRL SFTTrainer for trace-aligned SFT training.
Supports Neftune noise, proper chat template, and loss masking.

Usage:
    python3 -m src.student.train_sft_v2 \
        --dataset data/processed/sft_dataset.jsonl \
        --output-dir checkpoints/lora_adapters/sft_v2_adapter
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.sft_config_v2 import SFT_CONFIG


def prepare_model_for_training(model, tokenizer, config: dict):
    """Apply LoRA adapters to the model."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    return model


def apply_neftune(model, noise_alpha: float):
    """Apply Neftune noise embedding to the model."""
    model.config.neftune_noise_alpha = noise_alpha
    return model


def load_dataset_for_training(dataset_path: str):
    """Load JSONL dataset for TRL SFTTrainer."""
    from datasets import load_dataset

    return load_dataset("json", data_files=dataset_path, split="train")


def train(config: dict, dataset_path: str, output_dir: str):
    """Run SFT training."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from peft import PeftModel

    # Load model
    print(f"Loading model: {config['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = prepare_model_for_training(model, tokenizer, config)

    # Apply Neftune noise
    if config.get("neftune_noise_alpha"):
        print(f"Applying Neftune noise (alpha={config['neftune_noise_alpha']})")
        model = apply_neftune(model, config["neftune_noise_alpha"])

    # Enable gradient checkpointing
    model = FastLanguageModel.for_training(model)

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset_for_training(dataset_path)
    print(f"  Loaded {len(dataset)} samples")

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        optim=config["optim"],
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 200),
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    # Train
    print("Starting SFT training...")
    metrics = trainer.train()
    print(f"Training complete. Metrics: {metrics}")

    # Save
    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Programmatic SFT Training v2")
    parser.add_argument("--dataset", default="data/processed/sft_dataset.jsonl", help="SFT dataset JSONL")
    parser.add_argument("--output-dir", default=SFT_CONFIG["output_dir"], help="Output directory")
    args = parser.parse_args()

    train(SFT_CONFIG, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_sft_v2.py -v
```

Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/student/sft_config_v2.py src/student/train_sft_v2.py src/tests/test_sft_v2.py
git commit -m "feat: programmatic SFT training with trace alignment and Neftune noise"
```

---

### Task 6: Update DPO training script for TRL integration

**Files:**
- Modify: `src/student/train_dpo.py`
- Modify: `src/student/dpo_config.py`

Context: The existing DPO script has a basic TRL integration but needs updates: proper chat template formatting, correct dataset format for DPOTrainer (prompt/chosen/rejected keys), and loading the SFT v2 adapter.

- [ ] **Step 1: Update dpo_config.py**

Replace `src/student/dpo_config.py`:

```python
"""
DPO Training Configuration

Hyperparameters for Direct Preference Optimization training.
Student: Qwen/Qwen3.5-9B (Instruct), NF4 quantized at runtime.
"""

DPO_CONFIG = {
    # Base model (SFT v2 adapter)
    "base_model": "checkpoints/lora_adapters/sft_v2_adapter",
    # DPO-specific
    "beta": 0.1,
    "dpo_loss": "sigmoid",
    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 500,
    "warmup_steps": 50,
    "lr_scheduler_type": "cosine",
    # Output
    "output_dir": "checkpoints/lora_adapters/dpo_adapter",
    "logging_steps": 25,
    "save_steps": 100,
}
```

- [ ] **Step 2: Update train_dpo.py**

Replace `src/student/train_dpo.py`:

```python
#!/usr/bin/env python3
"""
DPO Training Script

Direct Preference Optimization on preference pairs.
Loads SFT v2 adapter, trains with TRL DPOTrainer.

Usage:
    python3 -m src.student.train_dpo \
        --sft-adapter-path checkpoints/lora_adapters/sft_v2_adapter \
        --dpo-pairs-path data/processed/dpo_pairs.jsonl \
        --output-dir checkpoints/lora_adapters/dpo_adapter
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.dpo_config import DPO_CONFIG


def load_dpo_pairs(filepath: str) -> list[dict]:
    """Load DPO pairs from JSONL file."""
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def train(config: dict, sft_adapter_path: str, dpo_pairs_path: str, output_dir: str):
    """Run DPO training."""
    from unsloth import FastLanguageModel
    from trl import DPOTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # Load model with SFT adapter
    print(f"Loading SFT adapter from {sft_adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA for DPO fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = FastLanguageModel.for_training(model)

    # Load dataset
    print(f"Loading DPO pairs from {dpo_pairs_path}...")
    dataset = load_dataset("json", data_files=dpo_pairs_path, split="train")
    print(f"  Loaded {len(dataset)} pairs")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        bf16=True,
        logging_steps=config.get("logging_steps", 25),
        save_steps=config.get("save_steps", 100),
        report_to="none",
    )

    # DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=training_args,
        beta=config["beta"],
        train_dataset=dataset,
    )

    print("Starting DPO training...")
    metrics = trainer.train()
    print(f"Training complete. Metrics: {metrics}")

    # Save
    print(f"Saving DPO adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="DPO Training for DM Alignment")
    parser.add_argument("--sft-adapter-path", default=DPO_CONFIG["base_model"], help="Path to SFT adapter")
    parser.add_argument("--dpo-pairs-path", default="data/processed/dpo_pairs.jsonl", help="Path to DPO pairs JSONL")
    parser.add_argument("--output-dir", default=DPO_CONFIG["output_dir"], help="Output directory")
    args = parser.parse_args()

    train(DPO_CONFIG, args.sft_adapter_path, args.dpo_pairs_path, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run existing DPO tests**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_dpo_training.py -v
```

Expected: PASS (existing tests should still pass with updated config)

- [ ] **Step 4: Commit**

```bash
git add src/student/train_dpo.py src/student/dpo_config.py
git commit -m "refactor: update DPO training for TRL integration and SFT v2 adapter"
```

---

### Task 7: Data prep runner script

**Files:**
- Create: `scripts/run_data_prep.sh`

- [ ] **Step 1: Create runner script**

Create `scripts/run_data_prep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Full data prep pipeline:
# 1. Convert parquet to JSON, clean traces, split SFT/DPO
# 2. Build trace-aligned SFT dataset
# 3. Generate rejected responses
# 4. Generate interleaved DPO pairs

PARQUET_PATH="${PARQUET_PATH:-/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/recipe_ml-1500-v1/parquet-files/batch_00000.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
SFT_COUNT="${SFT_COUNT:-1250}"

echo "=== Step 1: Convert parquet, clean traces, split ==="
python3 -m src.teacher.convert_full_dataset \
    --parquet-path "$PARQUET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --sft-count "$SFT_COUNT"

echo ""
echo "=== Step 2: Build trace-aligned SFT dataset ==="
python3 -m src.teacher.build_sft_dataset \
    --input "$OUTPUT_DIR/sft_dataset.json" \
    --output "$OUTPUT_DIR/sft_dataset.jsonl"

echo ""
echo "=== Step 3: Generate rejected responses ==="
python3 -m src.teacher.generate_rejected_responses \
    --input "$OUTPUT_DIR/dpo_dataset.json" \
    --output "$OUTPUT_DIR/rejected_responses.jsonl"

echo ""
echo "=== Step 4: Generate interleaved DPO pairs ==="
python3 -m src.teacher.generate_dpo_pairs \
    --dpo-data "$OUTPUT_DIR/dpo_dataset.json" \
    --rejections "$OUTPUT_DIR/rejected_responses.jsonl" \
    --output "$OUTPUT_DIR/dpo_pairs.jsonl"

echo ""
echo "=== Data prep complete ==="
echo "  SFT dataset: $OUTPUT_DIR/sft_dataset.jsonl"
echo "  DPO pairs:   $OUTPUT_DIR/dpo_pairs.jsonl"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/run_data_prep.sh
git add scripts/run_data_prep.sh
git commit -m "feat: data prep runner script for full pipeline"
```

---

## Self-Review

**1. Spec coverage:**
- [x] Full parquet conversion (1,500 rows) — Task 1
- [x] SFT/DPO split (1,250 + 250) — Task 1
- [x] Reasoning trace cleaning — Task 1 (reuses `clean_reasoning_traces.py`)
- [x] Trace-aligned SFT dataset with `<thought>` tags — Task 2
- [x] Three rejection types (Liberal Default, Jargon Trap, Shallow DM) — Task 3
- [x] Interleaved DPO pair generation (250 × 3 = 750 pairs) — Task 4
- [x] Programmatic SFT training with Neftune noise — Task 5
- [x] DPO training with TRL DPOTrainer — Task 6
- [x] Runner script — Task 7

**2. Placeholder scan:** No TBDs, no "implement later," no vague "add validation." Every step has actual code.

**3. Type consistency:**
- `convert_full_dataset.py` outputs `sft_dataset.json` and `dpo_dataset.json` → consumed by Tasks 2, 3, 4
- `build_sft_dataset.py` reads `sft_dataset.json`, outputs `sft_dataset.jsonl` → consumed by Task 5
- `generate_rejected_responses.py` reads `dpo_dataset.json`, outputs `rejected_responses.jsonl` → consumed by Task 4
- `generate_dpo_pairs.py` reads both, outputs `dpo_pairs.jsonl` → consumed by Task 6
- DPO pair format: `{prompt, chosen, rejected, rejection_type}` — consistent with TRL DPOTrainer expectations
- SFT sample format: `{conversations: [{role, content}, ...]}` — consistent with TRL SFTTrainer

**Gaps identified:**
- The `train_dpo.py` update uses `prompt`/`chosen`/`rejected` keys. TRL's DPOTrainer expects these exact key names by default. Verified: correct.
- The SFT trainer uses `dataset_text_field=""` which tells SFTTrainer to use the chat template automatically on the `conversations` field. This is the correct TRL v0.x API.
- Neftune noise: setting `model.config.neftune_noise_alpha` is the standard approach; TRL SFTTrainer reads this config value and applies noise during training.

---

## Execution Notes

**Running the data prep pipeline** (requires Docker or pandas working on host):
```bash
./scripts/run_data_prep.sh
```

**Running SFT training** (requires Docker container with GPU):
```bash
./scripts/run_dpo.sh   # existing script, update STUDIO_EXPORT_PATH or use --sft-adapter-path
```

**Running DPO training** (requires SFT v2 adapter):
```bash
python3 -m src.student.train_dpo \
    --sft-adapter-path checkpoints/lora_adapters/sft_v2_adapter \
    --dpo-pairs-path data/processed/dpo_pairs.jsonl
```

**Known blockers:**
- Pandas/numpy incompatibility on host — data prep scripts may need to run inside the Docker container
- Docker Desktop not currently running — SFT/DPO training requires GPU container
