"""Dry-run E2E test for TrackingManager full lifecycle.

Exercises init, reward wrapping, diagnostics, flush, report generation,
and finish without model loading or GPU.
"""
import sys
from types import ModuleType

import pytest


class FakeAlertLevel:
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class TestTrackingManagerDryRun:
    """Full lifecycle test with mocked trackio."""

    def _mock_trackio(self, mp):
        """Mock all trackio functions and capture calls."""
        state = {
            "init_calls": [],
            "log_calls": [],
            "alert_calls": [],
            "gpu_calls": [],
            "finish_calls": [],
        }

        class FakeRun:
            name = "dry-run"
            project = "test-project"
            config = {}

        class FakeTable:
            def __init__(self, data, columns):
                self.data = data
                self.columns = columns

        class FakeHistogram:
            def __init__(self, values):
                self.values = values

        class FakeTrace:
            def __init__(self, messages):
                self.messages = messages

        class FakeMarkdown:
            def __init__(self, text):
                self.text = text

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: (state["init_calls"].append(kw), FakeRun())[1]
        fake_trackio.log = lambda metrics, step=None: state["log_calls"].append((metrics, step))
        fake_trackio.alert = lambda **kw: state["alert_calls"].append(kw)
        fake_trackio.log_gpu = lambda **kw: state["gpu_calls"].append(kw)
        fake_trackio.finish = lambda: state["finish_calls"].append(True)
        fake_trackio.Table = FakeTable
        fake_trackio.Histogram = FakeHistogram
        fake_trackio.Trace = FakeTrace
        fake_trackio.Markdown = FakeMarkdown
        fake_trackio.AlertLevel = FakeAlertLevel
        mp.setitem(sys.modules, "trackio", fake_trackio)

        return state, fake_trackio

    def test_full_lifecycle_dry_run(self):
        from src.student.train_grpo_base import TrackingManager, TrackingCallback

        with pytest.MonkeyPatch.context() as mp:
            state, fake_trackio = self._mock_trackio(mp)

            # 1. Init
            mgr = TrackingManager()
            mgr.init(
                project="test-project",
                name="dry-run",
                config={"lr": 5e-7, "max_steps": 100},
                track="outcome",
                server_url="http://localhost:7860",
            )
            assert mgr._active is True
            assert len(state["init_calls"]) == 1
            assert state["init_calls"][0]["project"] == "test-project"

            # 2. Wrap and call reward function
            doc_index = {"prompt-1": {"answer": "+"}, "prompt-2": {"answer": "-"}}
            wrapped = mgr.wrap_reward_fn(
                lambda c, docs: [1.0 if doc.get("answer") == "+" else 0.0 for _, doc in zip(c, docs)],
                reward_name="outcome",
                doc_index=doc_index,
            )
            scores = wrapped(["completion-1", "completion-2"], ["prompt-1", "prompt-2"])
            assert scores == [1.0, 0.0]
            assert mgr._reward_samples["outcome"] == [1.0, 0.0]
            assert len(mgr._reward_table_rows) == 2

            # 3. Diagnostics on normal values - no alerts
            mgr.check_diagnostics(50, {"loss": 0.5, "reward": 0.8, "kl": 0.1})
            assert len(state["alert_calls"]) == 0

            # 4. GPU snapshot
            mgr.snapshot_gpu()
            assert len(state["gpu_calls"]) == 1

            # 5. Flush reward data
            mgr.flush_reward_data(50)
            log_calls = state["log_calls"]
            assert len(log_calls) >= 3
            assert mgr._reward_samples == {}
            assert mgr._reward_table_rows == []

            # 6. Completion sample trace
            mgr.log_completion_sample(60, "What is X?", "The answer is positive.")
            trace_calls = [c for c in state["log_calls"] if "completion/sample" in str(c[0].keys())]
            assert len(trace_calls) == 1

            # 7. Markdown report
            mgr.generate_report({"loss": 0.42, "reward": 0.75, "kl": 0.1})
            md_calls = [c for c in state["log_calls"] if "report/summary" in str(c[0].keys())]
            assert len(md_calls) == 1

            # 8. Alert on bad values
            mgr.check_diagnostics(200, {"loss": float("nan")})
            nan_alerts = [a for a in state["alert_calls"] if "NaN" in a.get("title", "")]
            assert len(nan_alerts) == 1

            # 9. Callback integration
            callback = TrackingCallback(mgr)

            class FakeState:
                global_step = 300

            callback.on_log(None, FakeState(), None, {"loss": 0.3, "reward": 0.9})
            assert len(state["gpu_calls"]) >= 2

            # 10. Finish
            mgr.finish()
            assert len(state["finish_calls"]) == 1
            assert mgr._active is False

    def test_lifecycle_with_failed_init_is_safe(self):
        import sys
        from types import ModuleType

        fake_trackio = ModuleType("trackio")
        fake_trackio.init = lambda **kw: (_ for _ in ()).throw(RuntimeError("server down"))
        fake_trackio.finish = lambda: None
        fake_trackio.log = lambda *a, **k: None
        fake_trackio.alert = lambda *a, **k: None
        fake_trackio.log_gpu = lambda *a, **k: None
        fake_trackio.AlertLevel = FakeAlertLevel

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "trackio", fake_trackio)

            from src.student.train_grpo_base import TrackingManager

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            assert mgr._active is False

            wrapped = mgr.wrap_reward_fn(lambda c, d: [1.0], reward_name="test")
            wrapped(["c1"], ["p1"])
            mgr.check_diagnostics(10, {"loss": float("nan")})
            mgr.snapshot_gpu()
            mgr.flush_reward_data(10)
            mgr.generate_report({"loss": 0.5})
            mgr.finish()

    def test_v4_multi_reward_tracking(self):
        from src.student.train_grpo_base import TrackingManager

        with pytest.MonkeyPatch.context() as mp:
            state, fake_trackio = self._mock_trackio(mp)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="process", server_url=None)

            doc_index = {"p1": {}}
            outcome_fn = mgr.wrap_reward_fn(
                lambda c, d: [0.9], reward_name="outcome", doc_index=doc_index,
            )
            planning_fn = mgr.wrap_reward_fn(
                lambda c, d: [0.5], reward_name="planning", doc_index=doc_index,
            )
            commitment_fn = mgr.wrap_reward_fn(
                lambda c, d: [1.0], reward_name="commitment", doc_index=doc_index,
            )

            outcome_fn(["c1"], ["p1"])
            planning_fn(["c1"], ["p1"])
            commitment_fn(["c1"], ["p1"])

            assert "outcome" in mgr._reward_samples
            assert "planning" in mgr._reward_samples
            assert "commitment" in mgr._reward_samples

            mgr.flush_reward_data(10)
            assert len(state["log_calls"]) >= 3


class TestTrackingManagerLiveServer:
    """Dry-run test that mimics a full training run against the real Track.io server.

    Exercises the same lifecycle as train_grpo_outcome.py but without model loading
    or GPU: init -> reward wrapper -> callback on_log (diagnostics, GPU, flush) ->
    completion trace -> markdown report -> finish.
    """

    def test_live_server_full_lifecycle(self):
        """Mimic a full v3 outcome training run against the real Track.io server."""
        import os
        import re

        from src.student.train_grpo_base import (
            TrackingCallback,
            TrackingManager,
        )

        server_url = os.environ.get("TRACKIO_SERVER_URL")
        assert server_url, "TRACKIO_SERVER_URL must be set for live server test"

        run_base = "dry-run-verify"

        # --- 1. Init (mirrors train_grpo_outcome.py:181-198) ---
        mgr = TrackingManager()
        mgr.init(
            project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
            name=run_base,
            config={
                "training_method": "GRPO",
                "track": "outcome",
                "version": "v3",
                "group_size": 8,
                "beta": 0.1,
                "learning_rate": 5e-07,
                "lora_rank": 16,
                "lora_alpha": 16,
                "max_completion_length": 512,
                "max_steps": 1500,
            },
            track="outcome",
            server_url=server_url,
        )
        assert mgr._active, "TrackingManager should be active after init"
        assert re.search(r"_\d{8}_\d{6}$", mgr._run.name), (
            f"Run name '{mgr._run.name}' should have timestamp suffix"
        )

        # --- 2. Build reward wrapper (mirrors train_grpo_outcome.py reward setup) ---
        doc_index = {
            "What causes inflation?": {"answer": "+", "topic": "economics"},
            "Does exercise improve health?": {"answer": "+", "topic": "health"},
            "Is gravity a fundamental force?": {"answer": "-", "topic": "physics"},
        }

        reward_fn = mgr.wrap_reward_fn(
            lambda completions, docs: [
                1.0 if doc.get("answer") == "+" else 0.0
                for _, doc in zip(completions, docs)
            ],
            reward_name="outcome",
            doc_index=doc_index,
        )

        # Simulate reward computation for a batch of completions
        prompts = list(doc_index.keys())
        completions = [f"Answer to: {p}" for p in prompts]
        scores = reward_fn(completions, prompts)
        assert scores == [1.0, 1.0, 0.0], f"Expected [1.0, 1.0, 0.0], got {scores}"

        # --- 3. Simulate callback on_log for multiple steps
        # (mirrors TrackingCallback.on_log -> check_diagnostics, snapshot_gpu, flush) ---
        callback = TrackingCallback(mgr)

        class FakeState:
            global_step = 0

        training_logs = [
            {"loss": 1.2, "reward": 0.67, "kl": 0.05, "completion_length": 120},
            {"loss": 0.95, "reward": 0.75, "kl": 0.04, "completion_length": 150},
            {"loss": 0.8, "reward": 0.8, "kl": 0.03, "completion_length": 180},
            {"loss": 0.7, "reward": 0.85, "kl": 0.02, "completion_length": 200},
            {"loss": 0.6, "reward": 0.9, "kl": 0.01, "completion_length": 220},
        ]

        for i, logs in enumerate(training_logs):
            FakeState.global_step = (i + 1) * 25
            callback.on_log(None, FakeState(), None, logs)

        # --- 4. Log a completion sample trace ---
        mgr.log_completion_sample(
            125,
            "What causes inflation?",
            "Inflation is primarily caused by an increase in the money supply relative to economic output.",
        )

        # --- 5. Generate markdown report (mirrors end of training) ---
        mgr.generate_report({
            "loss": 0.6,
            "reward": 0.9,
            "kl": 0.01,
            "completion_length": 220,
        })

        # --- 6. Finish ---
        mgr.finish()
        assert mgr._active is False
