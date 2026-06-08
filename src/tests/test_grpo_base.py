import os
import tempfile

import pytest


class TestStripVisionConfig:
    def test_removes_vision_keys_from_config(self):
        from src.student.train_grpo_base import strip_vision_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                f.write('{"architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {"hidden_size": 1024}, "vision_hidden_size": 1024, "text_config": {"hidden_size": 4096}}')

            strip_vision_config(tmpdir)

            import json
            with open(config_path) as f:
                config = json.load(f)
            assert "architectures" in config
            assert "text_config" in config
            assert "vision_config" not in config
            assert "vision_hidden_size" not in config

    def test_noop_when_no_vision_keys(self):
        from src.student.train_grpo_base import strip_vision_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            original = '{"architectures": ["Qwen3_5ForConditionalGeneration"]}'
            with open(config_path, "w") as f:
                f.write(original)

            strip_vision_config(tmpdir)

            with open(config_path) as f:
                assert f.read() == original

    def test_noop_when_no_config_json(self):
        from src.student.train_grpo_base import strip_vision_config

        with tempfile.TemporaryDirectory() as tmpdir:
            strip_vision_config(tmpdir)


class TestFindLatestCheckpoint:
    def test_empty_directory(self):
        from src.student.train_grpo_base import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            step, path = find_latest_checkpoint(tmpdir)
            assert step == 0
            assert path == ""

    def test_returns_highest_checkpoint(self):
        from src.student.train_grpo_base import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            for s in [100, 200, 300]:
                os.makedirs(os.path.join(tmpdir, f"checkpoint-{s}"))
            step, path = find_latest_checkpoint(tmpdir)
            assert step == 300
            assert "checkpoint-300" in path

    def test_ignores_non_checkpoint_dirs(self):
        from src.student.train_grpo_base import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-100"))
            os.makedirs(os.path.join(tmpdir, "not-a-checkpoint"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-abc"))
            step, path = find_latest_checkpoint(tmpdir)
            assert step == 100

    def test_nonexistent_directory(self):
        from src.student.train_grpo_base import find_latest_checkpoint

        step, path = find_latest_checkpoint("/tmp/does-not-exist-12345")
        assert step == 0
        assert path == ""


class TestBuildOutcomeDataset:
    def test_builds_dataset_with_prompt_and_doc_columns(self):
        from src.student.train_grpo_base import build_outcome_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "What is X?", "answer": "+", "dataset_type": "econcausal"}\n')
            f.write('{"prompt": "Is Y true?", "relation": "entailment", "dataset_type": "corr2cause"}\n')
            f.write('{"prompt": "Effect of Z?", "category": "null_effect"}\n')
            tmp_path = f.name

        try:
            class MockTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return messages[0]["content"] + " [answer]"

            try:
                dataset = build_outcome_dataset(tmp_path, MockTokenizer())
            except (ValueError, RuntimeError) as e:
                if "numpy.dtype size changed" in str(e):
                    pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
                raise

            assert len(dataset) == 3
            assert "prompt" in dataset.column_names
            assert "doc" in dataset.column_names
            assert dataset[0]["doc"]["dataset_type"] == "econcausal"
            assert dataset[1]["doc"]["relation"] == "entailment"
            assert dataset[2]["doc"]["category"] == "null_effect"
        finally:
            os.unlink(tmp_path)

    def test_prompt_is_chat_template_formatted(self):
        from src.student.train_grpo_base import build_outcome_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Question text", "answer": "+", "dataset_type": "econcausal"}\n')
            tmp_path = f.name

        try:
            class MockTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return "<|user|>" + messages[0]["content"] + "<|end|>"

            try:
                dataset = build_outcome_dataset(tmp_path, MockTokenizer())
            except (ValueError, RuntimeError) as e:
                if "numpy.dtype size changed" in str(e):
                    pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
                raise

            assert dataset[0]["prompt"].startswith("<|user|>")
            assert dataset[0]["prompt"].endswith("<|end|>")
        finally:
            os.unlink(tmp_path)

    def test_doc_preserves_all_fields(self):
        from src.student.train_grpo_base import build_outcome_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Q", "answer": "+", "dataset_type": "econcausal", "source": "test", "id": "t1"}\n')
            tmp_path = f.name

        try:
            class MockTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return messages[0]["content"]

            try:
                dataset = build_outcome_dataset(tmp_path, MockTokenizer())
            except (ValueError, RuntimeError) as e:
                if "numpy.dtype size changed" in str(e):
                    pytest.skip(f"Host numpy/pandas binary incompatibility: {e}")
                raise

            doc = dataset[0]["doc"]
            assert doc["answer"] == "+"
            assert doc["dataset_type"] == "econcausal"
            assert doc["source"] == "test"
            assert doc["id"] == "t1"
        finally:
            os.unlink(tmp_path)


class TestBuildRewardFnWithDocs:
    def test_reward_fn_receives_docs(self):
        from src.student.train_grpo_base import build_reward_fn_with_docs

        def my_reward(completions, docs):
            return [1.0 if doc.get("answer") == "+" else 0.0 for c, doc in zip(completions, docs)]

        fn = build_reward_fn_with_docs(my_reward, {"prompt1": {"answer": "+"}})

        results = fn(["completion1"], ["prompt1"])
        assert len(results) == 1
        assert results[0] == 1.0

    def test_reward_fn_maps_prompt_to_doc(self):
        from src.student.train_grpo_base import build_reward_fn_with_docs

        docs = [
            {"dataset_type": "econcausal", "answer": "+"},
            {"dataset_type": "corr2cause", "relation": "entailment"},
        ]
        prompts = ["prompt-A", "prompt-B"]

        received_docs = []
        def capture_reward(completions, captured_docs):
            received_docs.extend(captured_docs)

        fn = build_reward_fn_with_docs(capture_reward, dict(zip(prompts, docs)))
        fn(["c1", "c2"], prompts, [{"meta": "a"}, {"meta": "b"}], [])
        assert len(received_docs) == 2
        assert received_docs[0]["answer"] == "+"
        assert received_docs[1]["relation"] == "entailment"
