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


class TestTrackingManager:
    def test_init_creates_run_and_sets_active(self):
        import sys
        from types import ModuleType

        class FakeRun:
            name = "test-run"
            project = "test-project"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.finish = lambda: None

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(
                project="test-project",
                name="test-run",
                config={"lr": 5e-7},
                track="outcome",
                server_url="http://localhost:7860",
            )
            assert mgr._active is True
            assert mgr._run is not None

    def test_finish_calls_trackio_finish(self):
        import sys
        from types import ModuleType

        finish_called = []

        class FakeRun:
            name = "test-run"
            project = "test-project"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.finish = lambda: finish_called.append(True)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.finish()
            assert len(finish_called) == 1

    def test_init_failure_makes_methods_noop(self):
        import sys
        from types import ModuleType

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: (_ for _ in ()).throw(RuntimeError("no trackio"))
        fake_trackio.finish = lambda: None

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            assert mgr._active is False
            mgr.log_rewards(1, {"outcome": 0.5})
            mgr.finish()


class TestRewardWrapper:
    def test_wrapper_accumulates_per_sample_rewards(self):
        import sys
        from types import ModuleType

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.finish = lambda: None

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)

            wrapped = mgr.wrap_reward_fn(
                lambda completions, docs: [0.8, 0.3, 0.9],
                reward_name="outcome",
            )

            results = wrapped(["c1", "c2", "c3"], ["p1", "p2", "p3"])
            assert results == [0.8, 0.3, 0.9]
            assert mgr._reward_samples["outcome"] == [0.8, 0.3, 0.9]

    def test_wrapper_with_doc_lookup(self):
        import sys
        from types import ModuleType

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.finish = lambda: None

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)

            doc_index = {"prompt-A": {"answer": "+"}, "prompt-B": {"answer": "-"}}

            def reward_fn(completions, docs):
                return [1.0 if d.get("answer") == "+" else 0.0 for c, d in zip(completions, docs)]

            wrapped = mgr.wrap_reward_fn(reward_fn, reward_name="outcome", doc_index=doc_index)
            results = wrapped(["c1", "c2"], ["prompt-A", "prompt-B"])
            assert results == [1.0, 0.0]


class TestRewardLogging:
    def test_log_reward_table_creates_table(self):
        import sys
        from types import ModuleType

        log_calls = []

        class FakeTable:
            def __init__(self, data, columns):
                self.data = data
                self.columns = columns

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = fake_log
        fake_trackio.finish = lambda: None
        fake_trackio.Table = FakeTable

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)

            rows = [
                {"prompt": "Q1?", "completion": "A+", "outcome": 1.0},
                {"prompt": "Q2?", "completion": "A-", "outcome": 0.0},
            ]
            mgr.log_reward_table(100, rows)

            table_call = [c for c in log_calls if "reward/table" in c[0]]
            assert len(table_call) == 1
            assert table_call[0][1] == 100

    def test_log_reward_histograms_creates_histograms(self):
        import sys
        from types import ModuleType

        log_calls = []

        class FakeHistogram:
            def __init__(self, values):
                self.values = values

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = fake_log
        fake_trackio.finish = lambda: None
        fake_trackio.Histogram = FakeHistogram

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.log_reward_histograms(50, {"outcome": [0.8, 0.3, 0.9, 0.1]})

            hist_call = [c for c in log_calls if any("reward/hist" in k for k in c[0])]
            assert len(hist_call) == 1

    def test_log_completion_sample_creates_trace(self):
        import sys
        from types import ModuleType

        log_calls = []

        class FakeTrace:
            def __init__(self, messages):
                self.messages = messages

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = fake_log
        fake_trackio.finish = lambda: None
        fake_trackio.Trace = FakeTrace

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.log_completion_sample(30, "What is X?", "The answer is positive.")

            trace_call = [c for c in log_calls if "completion/sample" in c[0]]
            assert len(trace_call) == 1


class TestAlertDiagnostics:
    def _mock_trackio(self, mp):
        import sys
        from types import ModuleType

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.alert = lambda *a, **k: None
        fake_trackio.AlertLevel = FakeAlertLevel
        mp.setitem(sys.modules, "trackio", fake_trackio)
        return fake_trackio

    def test_nan_loss_fires_error_alert(self):
        import sys
        from types import ModuleType

        alerts_fired = []

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.AlertLevel = FakeAlertLevel

        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})
        fake_trackio.alert = fake_alert

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(50, {"loss": float("nan")})

            nan_alerts = [a for a in alerts_fired if "NaN" in a["title"]]
            assert len(nan_alerts) == 1

    def test_loss_divergence_fires_error_alert(self):
        import sys
        from types import ModuleType

        alerts_fired = []

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.AlertLevel = FakeAlertLevel

        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})
        fake_trackio.alert = fake_alert

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(150, {"loss": 6.0})

            div_alerts = [a for a in alerts_fired if "divergence" in a["title"].lower()]
            assert len(div_alerts) == 1

    def test_reward_collapse_fires_error_alert(self):
        import sys
        from types import ModuleType

        alerts_fired = []

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.AlertLevel = FakeAlertLevel

        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})
        fake_trackio.alert = fake_alert

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"reward": -3.0})

            collapse_alerts = [a for a in alerts_fired if "collapse" in a["title"].lower()]
            assert len(collapse_alerts) == 1

    def test_kl_high_fires_warn_alert(self):
        import sys
        from types import ModuleType

        alerts_fired = []

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.AlertLevel = FakeAlertLevel

        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})
        fake_trackio.alert = fake_alert

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"kl": 15.0})

            kl_alerts = [a for a in alerts_fired if "kl" in a["title"].lower()]
            assert len(kl_alerts) == 1

    def test_no_alerts_on_normal_values(self):
        import sys
        from types import ModuleType

        alerts_fired = []

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        class FakeAlertLevel:
            INFO = "info"
            WARN = "warn"
            ERROR = "error"

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.finish = lambda: None
        fake_trackio.AlertLevel = FakeAlertLevel

        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})
        fake_trackio.alert = fake_alert

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"loss": 0.5, "reward": 0.8, "kl": 0.1})

            warn_or_error = [a for a in alerts_fired if a["level"] != FakeAlertLevel.INFO]
            assert len(warn_or_error) == 0

    def test_inactive_manager_is_noop(self):
        from src.student.train_grpo_base import TrackingManager

        mgr = TrackingManager()
        mgr.check_diagnostics(50, {"loss": float("nan")})
        assert mgr._active is False


class TestSystemAndReport:
    def test_snapshot_gpu_calls_log_gpu(self):
        import sys
        from types import ModuleType

        gpu_called = []
        def fake_log_gpu(run=None, device=None):
            gpu_called.append(True)
            return {}

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log_gpu = fake_log_gpu
        fake_trackio.finish = lambda: None

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.snapshot_gpu()
            assert len(gpu_called) == 1

    def test_generate_report_logs_markdown(self):
        import sys
        from types import ModuleType

        log_calls = []
        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        class FakeMarkdown:
            def __init__(self, content):
                self.content = content

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: FakeRun()
        fake_trackio.log = fake_log
        fake_trackio.finish = lambda: None
        fake_trackio.Markdown = FakeMarkdown

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={"lr": 5e-7}, track="outcome", server_url=None)
            mgr.generate_report({"loss": 0.42, "reward": 0.75})

            md_calls = [c for c in log_calls if "report/summary" in str(c[0].keys())]
            assert len(md_calls) == 1


class TestTrackingCallback:
    def test_callback_delegates_to_manager(self):
        from src.student.train_grpo_base import TrackingCallback, TrackingManager

        manager = TrackingManager()
        manager._active = True
        diag_calls = []
        metric_calls = []
        gpu_calls = []
        flush_calls = []
        manager.check_diagnostics = lambda s, l: diag_calls.append(s)
        manager.log_training_metrics = lambda s, l: metric_calls.append((s, l))
        manager.snapshot_gpu = lambda: gpu_calls.append(True)
        manager.flush_reward_data = lambda s: flush_calls.append(s)

        callback = TrackingCallback(manager)

        class FakeState:
            global_step = 100

        callback.on_log(None, FakeState(), None, {"loss": 0.5})
        assert 100 in diag_calls
        assert len(metric_calls) == 1 and metric_calls[0][0] == 100
        assert len(gpu_calls) == 1
        assert 100 in flush_calls

    def test_callback_noop_when_manager_inactive(self):
        from src.student.train_grpo_base import TrackingCallback, TrackingManager

        calls = []
        manager = TrackingManager()
        manager._active = False
        manager.check_diagnostics = lambda s, l: calls.append("diag")
        manager.log_training_metrics = lambda s, l: calls.append("metrics")
        manager.snapshot_gpu = lambda: calls.append("gpu")
        manager.flush_reward_data = lambda s: calls.append("flush")

        callback = TrackingCallback(manager)

        class FakeState:
            global_step = 50

        callback.on_log(None, FakeState(), None, {"loss": 0.5})
        assert len(calls) == 0
