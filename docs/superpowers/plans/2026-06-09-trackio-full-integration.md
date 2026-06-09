# Full Track.io Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the minimal `TrackioCallback` with a comprehensive `TrackingManager` that uses all Track.io verticals (Tables, Histograms, Trace, Markdown, Alerts, System metrics) across v3 and v4 training scripts.

**Architecture:** `TrackingManager` in `train_grpo_base.py` encapsulates all Track.io capabilities. A thin `TrackingCallback` delegates to the manager. Training scripts wire up via `tracker.init()` and `tracker.wrap_reward_fn()`. TRL's `report_to="trackio"` handles basic metric logging. All alert thresholds, GPU snapshot config, and report templates live in one place.

**Tech Stack:** Python, Track.io 0.26.0, TRL GRPOTrainer, pytest

---

### Task 1: TrackingManager core class with init/finish

**Files:**
- Modify: `src/student/train_grpo_base.py` (append after line 247)
- Test: `src/tests/test_grpo_base.py` (new class `TestTrackingManager`)

- [ ] **Step 1: Write failing test for TrackingManager init and finish**

Add to `src/tests/test_grpo_base.py`:

```python
class TestTrackingManager:
    def test_init_creates_run_and_sets_active(self):
        from src.student.train_grpo_base import TrackingManager

        class FakeRun:
            name = "test-run"
            project = "test-project"
            config = {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", lambda **kw: FakeRun())
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

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
        from src.student.train_grpo_base import TrackingManager

        finish_called = []

        class FakeRun:
            name = "test-run"
            project = "test-project"
            config = {}

        def fake_init(**kw):
            return FakeRun()

        def fake_finish():
            finish_called.append(True)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", fake_init)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", fake_finish)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.finish()
            assert len(finish_called) == 1

    def test_init_failure_makes_methods_noop(self):
        from src.student.train_grpo_base import TrackingManager

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", lambda **kw: (_ for _ in ()).throw(RuntimeError("no trackio")))

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            assert mgr._active is False
            mgr.log_rewards(1, {"outcome": 0.5})
            mgr.finish()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestTrackingManager -v --tb=short`
Expected: FAIL with `TrackingManager` not defined

- [ ] **Step 3: Implement TrackingManager class**

Append to `src/student/train_grpo_base.py` after the `TrackioCallback` class (after line 247):

```python
class TrackingManager:
    """Encapsulates all Track.io tracking verticals for GRPO training.

    Single source of truth for alert thresholds, GPU snapshot frequency,
    and report templates. If init() fails, all methods become no-ops.
    """

    ALERT_LOSS_DIVERGENCE_THRESHOLD = 5.0
    ALERT_LOSS_DIVERGENCE_MIN_STEP = 100
    ALERT_STALL_STEP_WINDOW = 100
    ALERT_STALL_LOSS_DELTA = 0.001
    ALERT_REWARD_COLLAPSE_THRESHOLD = -2.0
    ALERT_KL_HIGH_THRESHOLD = 10.0
    ALERT_SHORT_COMPLETION_THRESHOLD = 10
    ALERT_CHECKPOINT_INTERVAL = 200
    ALERT_PROCESS_REWARD_LOW = 0.1
    ALERT_FORMAT_PENALTY_DOMINANCE = 0.5

    def __init__(self) -> None:
        self._run = None
        self._active = False
        self._track = ""
        self._reward_samples: Dict[str, List[float]] = {}
        self._reward_table_rows: List[Dict[str, Any]] = []
        self._loss_history: List[Tuple[int, float]] = []

    def init(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        track: str,
        server_url: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        try:
            import trackio
            self._run = trackio.init(
                project=project,
                name=name,
                config=config,
                group=group,
                server_url=server_url,
                auto_log_gpu=True,
            )
            self._active = True
            self._track = track
            logger.info(f"TrackingManager initialized: project={project}, name={name}, track={track}")
        except Exception as e:
            logger.warning(f"TrackingManager init failed: {e}")
            self._active = False

    def finish(self) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.finish()
            logger.info("TrackingManager finished")
        except Exception as e:
            logger.warning(f"TrackingManager finish failed: {e}")
        finally:
            self._active = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestTrackingManager -v --tb=short`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: add TrackingManager core with init/finish and guard on failure"
```

---

### Task 2: Reward wrapper with per-sample accumulation

**Files:**
- Modify: `src/student/train_grpo_base.py` (add methods to `TrackingManager`)
- Test: `src/tests/test_grpo_base.py` (new class `TestRewardWrapper`)

- [ ] **Step 1: Write failing test for reward wrapper accumulation**

Add to `src/tests/test_grpo_base.py`:

```python
class TestRewardWrapper:
    def test_wrapper_accumulates_per_sample_rewards(self):
        from src.student.train_grpo_base import TrackingManager

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        def fake_init(**kw):
            return FakeRun()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", fake_init)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

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
        from src.student.train_grpo_base import TrackingManager

        class FakeRun:
            name = "r"
            project = "p"
            config = {}

        def fake_init(**kw):
            return FakeRun()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", fake_init)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)

            doc_index = {"prompt-A": {"answer": "+"}, "prompt-B": {"answer": "-"}}

            def reward_fn(completions, docs):
                return [1.0 if d.get("answer") == "+" else 0.0 for c, d in zip(completions, docs)]

            wrapped = mgr.wrap_reward_fn(reward_fn, reward_name="outcome", doc_index=doc_index)
            results = wrapped(["c1", "c2"], ["prompt-A", "prompt-B"])
            assert results == [1.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestRewardWrapper -v --tb=short`
Expected: FAIL with `wrap_reward_fn` not defined

- [ ] **Step 3: Implement wrap_reward_fn method**

Add to `TrackingManager` class:

```python
    def wrap_reward_fn(
        self,
        fn: Callable,
        reward_name: str,
        doc_index: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Callable:
        """Wrap a reward function to accumulate per-sample data for Tables/Histograms.

        Args:
            fn: If doc_index is provided, fn(completions, docs) -> List[float].
                If doc_index is None, fn(completions) -> List[float].
            reward_name: Key for tracking this reward function.
            doc_index: Optional dict mapping prompt text to doc record.

        Returns:
            TRL-compatible reward function.
        """
        def wrapped(
            completions: List[str],
            prompts: List[str],
            *args: Any,
            **kwargs: Any,
        ) -> List[float]:
            if doc_index:
                docs = [doc_index.get(p, {}) for p in prompts]
                scores = fn(completions, docs)
            else:
                scores = fn(completions)

            if self._active:
                self._reward_samples.setdefault(reward_name, []).extend(scores)
                for c, p, s in zip(completions, prompts, scores):
                    self._reward_table_rows.append({
                        "prompt": p[:100] if p else "",
                        "completion": c[:200] if c else "",
                        reward_name: s,
                    })
            return scores
        return wrapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestRewardWrapper -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: add reward wrapper with per-sample accumulation"
```

---

### Task 3: Reward Table, Histogram, and Trace logging

**Files:**
- Modify: `src/student/train_grpo_base.py` (add methods to `TrackingManager`)
- Test: `src/tests/test_grpo_base.py` (new class `TestRewardLogging`)

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_grpo_base.py`:

```python
class TestRewardLogging:
    def test_log_reward_table_creates_table(self):
        from src.student.train_grpo_base import TrackingManager

        log_calls = []
        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
            mp.setattr(__import__("trackio", fromlist=["log"]), "log", fake_log)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)

            rows = [
                {"prompt": "Q1?", "completion": "A+", "outcome": 1.0},
                {"prompt": "Q2?", "completion": "A-", "outcome": 0.0},
            ]
            mgr.log_reward_table(100, rows)

            table_call = [c for c in log_calls if "reward/table" in str(c[0].keys())]
            assert len(table_call) == 1
            assert table_call[0][1] == 100

    def test_log_reward_histograms_creates_histograms(self):
        from src.student.train_grpo_base import TrackingManager

        log_calls = []
        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
            mp.setattr(__import__("trackio", fromlist=["log"]), "log", fake_log)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.log_reward_histograms(50, {"outcome": [0.8, 0.3, 0.9, 0.1]})

            hist_call = [c for c in log_calls if "reward/hist" in str(c[0].keys())]
            assert len(hist_call) == 1

    def test_log_completion_sample_creates_trace(self):
        from src.student.train_grpo_base import TrackingManager

        log_calls = []
        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init", lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
            mp.setattr(__import__("trackio", fromlist=["log"]), "log", fake_log)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.log_completion_sample(30, "What is X?", "The answer is positive.")

            trace_call = [c for c in log_calls if "completion/sample" in str(c[0].keys())]
            assert len(trace_call) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestRewardLogging -v --tb=short`
Expected: FAIL with methods not defined

- [ ] **Step 3: Implement reward logging methods**

Add to `TrackingManager` class:

```python
    def log_rewards(self, step: int, reward_breakdown: Dict[str, float]) -> None:
        """Log per-reward-function mean scores as scalars."""
        if not self._active:
            return
        try:
            import trackio
            metrics = {f"reward/{k}": v for k, v in reward_breakdown.items()}
            trackio.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward scalars: {e}")

    def log_reward_table(self, step: int, rows: List[Dict[str, Any]]) -> None:
        """Log per-sample reward breakdowns as trackio.Table."""
        if not self._active or not rows:
            return
        try:
            import trackio
            columns = list(rows[0].keys())
            table_data = [[row.get(col, "") for col in columns] for row in rows]
            trackio.log({"reward/table": trackio.Table(data=table_data, columns=columns)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward table: {e}")

    def log_reward_histograms(self, step: int, samples: Dict[str, List[float]]) -> None:
        """Log reward value distributions as trackio.Histogram."""
        if not self._active or not samples:
            return
        try:
            import trackio
            metrics = {}
            for name, values in samples.items():
                if values:
                    metrics[f"reward/hist/{name}"] = trackio.Histogram(values)
            if metrics:
                trackio.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward histograms: {e}")

    def log_completion_sample(self, step: int, prompt: str, completion: str) -> None:
        """Log a completion example as trackio.Trace."""
        if not self._active:
            return
        try:
            import trackio
            trace = trackio.Trace(messages=[
                {"role": "user", "content": prompt[:300]},
                {"role": "assistant", "content": completion[:500]},
            ])
            trackio.log({"completion/sample": trace}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log completion sample: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestRewardLogging -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: add reward Table, Histogram, and Trace logging methods"
```

---

### Task 4: Alert diagnostics with check_diagnostics

**Files:**
- Modify: `src/student/train_grpo_base.py` (add methods to `TrackingManager`)
- Test: `src/tests/test_grpo_base.py` (new class `TestAlertDiagnostics`)

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_grpo_base.py`:

```python
class TestAlertDiagnostics:
    def _mock_trackio(self, mp, with_alert=True):
        mp.setattr(__import__("trackio", fromlist=["init"]), "init",
                   lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
        mp.setattr(__import__("trackio", fromlist=["log"]), "log", lambda *a, **k: None)
        mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)
        if with_alert:
            return []
        return None

    def test_nan_loss_fires_error_alert(self):
        from src.student.train_grpo_base import TrackingManager

        alerts_fired = []
        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})

        with pytest.MonkeyPatch.context() as mp:
            self._mock_trackio(mp)
            mp.setattr(__import__("trackio", fromlist=["alert"]), "alert", fake_alert)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(50, {"loss": float("nan")})

            nan_alerts = [a for a in alerts_fired if "NaN" in a["title"]]
            assert len(nan_alerts) == 1

    def test_loss_divergence_fires_error_alert(self):
        from src.student.train_grpo_base import TrackingManager

        alerts_fired = []
        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})

        with pytest.MonkeyPatch.context() as mp:
            self._mock_trackio(mp)
            mp.setattr(__import__("trackio", fromlist=["alert"]), "alert", fake_alert)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(150, {"loss": 6.0})

            div_alerts = [a for a in alerts_fired if "divergence" in a["title"].lower()]
            assert len(div_alerts) == 1

    def test_reward_collapse_fires_error_alert(self):
        from src.student.train_grpo_base import TrackingManager

        alerts_fired = []
        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})

        with pytest.MonkeyPatch.context() as mp:
            self._mock_trackio(mp)
            mp.setattr(__import__("trackio", fromlist=["alert"]), "alert", fake_alert)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"reward": -3.0})

            collapse_alerts = [a for a in alerts_fired if "collapse" in a["title"].lower()]
            assert len(collapse_alerts) == 1

    def test_kl_high_fires_warn_alert(self):
        from src.student.train_grpo_base import TrackingManager

        alerts_fired = []
        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})

        with pytest.MonkeyPatch.context() as mp:
            self._mock_trackio(mp)
            mp.setattr(__import__("trackio", fromlist=["alert"]), "alert", fake_alert)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"kl": 15.0})

            kl_alerts = [a for a in alerts_fired if "kl" in a["title"].lower()]
            assert len(kl_alerts) == 1

    def test_no_alerts_on_normal_values(self):
        from src.student.train_grpo_base import TrackingManager

        alerts_fired = []
        def fake_alert(title, text=None, level=None, webhook_url=None):
            alerts_fired.append({"title": title, "level": level})

        with pytest.MonkeyPatch.context() as mp:
            self._mock_trackio(mp)
            mp.setattr(__import__("trackio", fromlist=["alert"]), "alert", fake_alert)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.check_diagnostics(100, {"loss": 0.5, "reward": 0.8, "kl": 0.1})

            warn_or_error = [a for a in alerts_fired
                             if a["level"] != __import__("trackio", fromlist=["AlertLevel"]).AlertLevel.INFO]
            assert len(warn_or_error) == 0

    def test_inactive_manager_is_noop(self):
        from src.student.train_grpo_base import TrackingManager

        mgr = TrackingManager()
        mgr.check_diagnostics(50, {"loss": float("nan")})
        assert mgr._active is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestAlertDiagnostics -v --tb=short`
Expected: FAIL with `check_diagnostics` not defined

- [ ] **Step 3: Implement check_diagnostics**

Add to `TrackingManager` class:

```python
    def _fire_alert(self, title: str, text: str, level) -> None:
        if not self._active:
            return
        try:
            import trackio
            trackio.alert(title=title, text=text, level=level)
        except Exception as e:
            logger.warning(f"Failed to fire alert '{title}': {e}")

    def check_diagnostics(self, step: int, logs: Dict[str, Any]) -> None:
        """Check training metrics and fire alerts for diagnostic conditions."""
        if not self._active:
            return

        loss = logs.get("loss")
        reward = logs.get("reward")
        kl = logs.get("kl")
        completion_len = logs.get("completion_length")

        if loss is not None:
            import math
            if math.isnan(loss) or math.isinf(loss):
                self._fire_alert(
                    title="NaN/Inf loss",
                    text=f"loss={loss} at step {step}. Training is broken.",
                    level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
                )
                return

            self._loss_history.append((step, loss))

            if step > self.ALERT_LOSS_DIVERGENCE_MIN_STEP and loss > self.ALERT_LOSS_DIVERGENCE_THRESHOLD:
                self._fire_alert(
                    title="Loss divergence",
                    text=f"loss={loss:.4f} above {self.ALERT_LOSS_DIVERGENCE_THRESHOLD} after {step} steps.",
                    level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
                )

            if len(self._loss_history) >= 2:
                recent = [l for s, l in self._loss_history[-self.ALERT_STALL_STEP_WINDOW-1:]]
                if len(recent) >= 2 and abs(recent[-1] - recent[0]) < self.ALERT_STALL_LOSS_DELTA:
                    self._fire_alert(
                        title="Training stall",
                        text=f"Loss delta={abs(recent[-1] - recent[0]):.6f} over {self.ALERT_STALL_STEP_WINDOW} steps.",
                        level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
                    )

        if reward is not None and reward < self.ALERT_REWARD_COLLAPSE_THRESHOLD:
            self._fire_alert(
                title="Reward collapse",
                text=f"reward={reward:.4f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.ERROR,
            )

        if kl is not None and kl > self.ALERT_KL_HIGH_THRESHOLD:
            self._fire_alert(
                title="KL divergence too high",
                text=f"kl={kl:.4f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
            )

        if completion_len is not None and completion_len < self.ALERT_SHORT_COMPLETION_THRESHOLD:
            self._fire_alert(
                title="Completions too short",
                text=f"Mean completion length={completion_len:.1f} at step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.WARN,
            )

        if step > 0 and step % self.ALERT_CHECKPOINT_INTERVAL == 0:
            self._fire_alert(
                title=f"Checkpoint milestone",
                text=f"Reached step {step}.",
                level=__import__("trackio", fromlist=["AlertLevel"]).AlertLevel.INFO,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestAlertDiagnostics -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: add alert diagnostics with loss, reward, KL, completion checks"
```

---

### Task 5: GPU snapshot and Markdown report generation

**Files:**
- Modify: `src/student/train_grpo_base.py` (add methods to `TrackingManager`)
- Test: `src/tests/test_grpo_base.py` (new class `TestSystemAndReport`)

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_grpo_base.py`:

```python
class TestSystemAndReport:
    def test_snapshot_gpu_calls_log_gpu(self):
        from src.student.train_grpo_base import TrackingManager

        gpu_called = []
        def fake_log_gpu(run=None, device=None):
            gpu_called.append(True)
            return {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init",
                       lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
            mp.setattr(__import__("trackio", fromlist=["log_gpu"]), "log_gpu", fake_log_gpu)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            mgr.snapshot_gpu()
            assert len(gpu_called) == 1

    def test_generate_report_logs_markdown(self):
        from src.student.train_grpo_base import TrackingManager

        log_calls = []
        def fake_log(metrics, step=None):
            log_calls.append((dict(metrics), step))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init",
                       lambda **kw: type("R", (), {"name": "r", "project": "p", "config": {}})())
            mp.setattr(__import__("trackio", fromlist=["log"]), "log", fake_log)
            mp.setattr(__import__("trackio", fromlist=["finish"]), "finish", lambda: None)

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={"lr": 5e-7}, track="outcome", server_url=None)
            mgr.generate_report({"loss": 0.42, "reward": 0.75})

            md_calls = [c for c in log_calls if "report/summary" in str(c[0].keys())]
            assert len(md_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestSystemAndReport -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement snapshot_gpu and generate_report**

Add to `TrackingManager` class:

```python
    def snapshot_gpu(self) -> None:
        """Call trackio.log_gpu() for per-step GPU system metrics."""
        if not self._active:
            return
        try:
            import trackio
            trackio.log_gpu()
        except Exception as e:
            logger.debug(f"GPU snapshot failed (expected if no GPU): {e}")

    def generate_report(self, final_logs: Dict[str, Any]) -> None:
        """Generate and log a Markdown training summary report."""
        if not self._active:
            return
        try:
            import trackio

            lines = ["# Training Summary", ""]
            lines.append(f"**Track:** {self._track}")
            lines.append(f"**Final loss:** {final_logs.get('loss', 'N/A')}")
            lines.append(f"**Final reward:** {final_logs.get('reward', 'N/A')}")
            lines.append(f"**Final KL:** {final_logs.get('kl', 'N/A')}")
            lines.append(f"**Completion length:** {final_logs.get('completion_length', 'N/A')}")
            lines.append("")

            if self._loss_history:
                lines.append(f"**Steps logged:** {len(self._loss_history)}")
                lines.append(f"**First loss:** {self._loss_history[0][1]:.4f} (step {self._loss_history[0][0]})")
                lines.append(f"**Last loss:** {self._loss_history[-1][1]:.4f} (step {self._loss_history[-1][0]})")
                lines.append("")

            md = trackio.Markdown("\n".join(lines))
            trackio.log({"report/summary": md})
            logger.info("Training summary report logged")
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestSystemAndReport -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: add GPU snapshot and Markdown report generation"
```

---

### Task 6: TrackingCallback and flush_reward_data

**Files:**
- Modify: `src/student/train_grpo_base.py` (replace `TrackioCallback`, add `flush_reward_data` to manager)
- Test: `src/tests/test_grpo_base.py` (new class `TestTrackingCallback`)

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_grpo_base.py`:

```python
class TestTrackingCallback:
    def test_callback_delegates_to_manager(self):
        from src.student.train_grpo_base import TrackingCallback, TrackingManager

        manager = TrackingManager()
        manager._active = True
        diag_calls = []
        gpu_calls = []
        flush_calls = []
        manager.check_diagnostics = lambda s, l: diag_calls.append(s)
        manager.snapshot_gpu = lambda: gpu_calls.append(True)
        manager.flush_reward_data = lambda s: flush_calls.append(s)

        callback = TrackingCallback(manager)

        class FakeState:
            global_step = 100

        callback.on_log(None, FakeState(), None, {"loss": 0.5})
        assert 100 in diag_calls
        assert len(gpu_calls) == 1
        assert 100 in flush_calls

    def test_callback_noop_when_manager_inactive(self):
        from src.student.train_grpo_base import TrackingCallback, TrackingManager

        calls = []
        manager = TrackingManager()
        manager._active = False
        manager.check_diagnostics = lambda s, l: calls.append("diag")
        manager.snapshot_gpu = lambda: calls.append("gpu")
        manager.flush_reward_data = lambda s: calls.append("flush")

        callback = TrackingCallback(manager)

        class FakeState:
            global_step = 50

        callback.on_log(None, FakeState(), None, {"loss": 0.5})
        assert len(calls) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestTrackingCallback -v --tb=short`
Expected: FAIL with `TrackingCallback` not defined

- [ ] **Step 3: Implement TrackingCallback and flush_reward_data**

Add `flush_reward_data` to `TrackingManager` class:

```python
    def flush_reward_data(self, step: int) -> None:
        """Flush accumulated reward samples as Table + Histograms, then clear."""
        if not self._active or not self._reward_samples and not self._reward_table_rows:
            return

        if self._reward_table_rows:
            self.log_reward_table(step, self._reward_table_rows)

        if self._reward_samples:
            self.log_reward_histograms(step, self._reward_samples)

            # Log mean reward scalars
            means = {}
            for name, values in self._reward_samples.items():
                if values:
                    means[name] = sum(values) / len(values)
            if means:
                self.log_rewards(step, means)

        self._reward_samples.clear()
        self._reward_table_rows.clear()
```

Replace `TrackioCallback` class (lines 203-247) with `TrackingCallback`:

```python
class TrackingCallback(TrainerCallback):
    """Trainer callback that delegates all tracking to a TrackingManager.

    Replaces TrackioCallback. The manager handles diagnostics, GPU snapshots,
    and reward data flushing on each logging step.

    Usage:
        tracker = TrackingManager()
        tracker.init(project="...", name="...", config={}, track="outcome")
        callback = TrackingCallback(tracker)
        trainer.add_callback(callback)
        # ... training ...
        tracker.finish()
    """

    def __init__(self, manager: TrackingManager) -> None:
        super().__init__()
        self._manager = manager

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._manager._active or not logs:
            return
        step = getattr(state, "global_step", 0)
        self._manager.check_diagnostics(step, logs)
        self._manager.snapshot_gpu()
        self._manager.flush_reward_data(step)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_grpo_base.py::TestTrackingCallback -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo_base.py src/tests/test_grpo_base.py
git commit -m "feat: replace TrackioCallback with TrackingCallback + flush_reward_data"
```

---

### Task 7: Wire up train_grpo_outcome.py (v3 track)

**Files:**
- Modify: `src/student/train_grpo_outcome.py`
- Modify: `src/student/grpo_config_outcome.py` (change `report_to`)

- [ ] **Step 1: Change report_to in grpo_config_outcome.py**

In `src/student/grpo_config_outcome.py` line 79, change:
```python
        report_to="none",
```
to:
```python
        report_to="trackio",
```

- [ ] **Step 2: Wire TrackingManager in train_grpo_outcome.py**

Replace the current `trackio.init()` and `TrackioCallback` usage with:

```python
from src.student.train_grpo_base import TrackingManager, TrackingCallback, build_outcome_dataset

# ... in main():
tracker = TrackingManager()
tracker.init(
    project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
    name=f"v3_outcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={"track": "outcome", "lr": 5e-7, "max_steps": max_steps},
    track="outcome",
    server_url=os.environ.get("TRACKIO_SERVER_URL"),
)

trainer.add_callback(TrackingCallback(tracker))

# Replace build_reward_fn_with_docs call with tracker.wrap_reward_fn:
reward_fn_wrapped = tracker.wrap_reward_fn(
    compute_outcome_reward_fn,  # the (completions, docs) -> scores function
    reward_name="outcome",
    doc_index=doc_index,
)
```

Remove the old `TrackioCallback` import and `trackio.init()` / `trackio.finish()` calls. Add `tracker.finish()` at the end of training.

- [ ] **Step 3: Verify import works**

Run: `python3 -c "from src.student.train_grpo_base import TrackingManager, TrackingCallback; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/student/train_grpo_outcome.py src/student/grpo_config_outcome.py
git commit -m "feat: wire v3 outcome track to TrackingManager"
```

---

### Task 8: Wire up train_grpo_process.py (v4 track)

**Files:**
- Modify: `src/student/train_grpo_process.py`
- Modify: `src/student/grpo_config_process.py` (change `report_to`)

- [ ] **Step 1: Change report_to in grpo_config_process.py**

In `src/student/grpo_config_process.py` line 112, change:
```python
        report_to="none",
```
to:
```python
        report_to="trackio",
```

- [ ] **Step 2: Wire TrackingManager in train_grpo_process.py**

Replace the current `trackio.init()` and `TrackioCallback` usage with:

```python
from src.student.train_grpo_base import TrackingManager, TrackingCallback, build_outcome_dataset

# ... in main():
tracker = TrackingManager()
tracker.init(
    project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
    name=f"v4_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={"track": "process", "lr": 5e-7, "max_steps": max_steps},
    track="process",
    server_url=os.environ.get("TRACKIO_SERVER_URL"),
)

trainer.add_callback(TrackingCallback(tracker))

# For v4, wrap each reward component separately:
outcome_wrapped = tracker.wrap_reward_fn(
    lambda c, d: [compute_outcome_reward(doc, comp) for comp, doc in zip(c, d)],
    reward_name="outcome",
    doc_index=doc_index,
)
```

Remove old `TrackioCallback` import and `trackio.init()`/`trackio.finish()` calls. Add `tracker.finish()` at end.

- [ ] **Step 3: Verify import works**

Run: `python3 -c "from src.student.train_grpo_base import TrackingManager, TrackingCallback; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/student/train_grpo_process.py src/student/grpo_config_process.py
git commit -m "feat: wire v4 process track to TrackingManager"
```

---

### Task 9: Dry-run E2E test

**Files:**
- Create: `src/tests/test_trackio_dry_run.py`

- [ ] **Step 1: Write the dry-run E2E test**

Create `src/tests/test_trackio_dry_run.py`:

```python
"""Dry-run E2E test for TrackingManager full lifecycle.

Exercises init, reward wrapping, diagnostics, flush, report generation,
and finish without model loading or GPU.
"""
import pytest


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

        mp.setattr(__import__("trackio", fromlist=["init"]), "init",
                   lambda **kw: (state["init_calls"].append(kw), FakeRun())[1])
        mp.setattr(__import__("trackio", fromlist=["log"]), "log",
                   lambda metrics, step=None: state["log_calls"].append((metrics, step)))
        mp.setattr(__import__("trackio", fromlist=["alert"]), "alert",
                   lambda **kw: state["alert_calls"].append(kw))
        mp.setattr(__import__("trackio", fromlist=["log_gpu"]), "log_gpu",
                   lambda **kw: state["gpu_calls"].append(kw))
        mp.setattr(__import__("trackio", fromlist=["finish"]), "finish",
                   lambda: state["finish_calls"].append(True))

        return state

    def test_full_lifecycle_dry_run(self):
        from src.student.train_grpo_base import TrackingManager, TrackingCallback

        with pytest.MonkeyPatch.context() as mp:
            state = self._mock_trackio(mp)

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
                lambda c, d: [1.0 if d.get("answer") == "+" else 0.0 for c_i, d_i in zip(c, d)],
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
            assert len(log_calls) >= 3  # table + histogram + scalar
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

            diag_before = len(state["alert_calls"])
            callback.on_log(None, FakeState(), None, {"loss": 0.3, "reward": 0.9})
            assert len(state["gpu_calls"]) >= 2

            # 10. Finish
            mgr.finish()
            assert len(state["finish_calls"]) == 1
            assert mgr._active is False

    def test_lifecycle_with_failed_init_is_safe(self):
        from src.student.train_grpo_base import TrackingManager

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(__import__("trackio", fromlist=["init"]), "init",
                       lambda **kw: (_ for _ in ()).throw(RuntimeError("server down")))

            mgr = TrackingManager()
            mgr.init(project="p", name="r", config={}, track="outcome", server_url=None)
            assert mgr._active is False

            # All operations should be safe no-ops
            wrapped = mgr.wrap_reward_fn(lambda c: [1.0], reward_name="test")
            wrapped(["c1"], ["p1"])
            mgr.check_diagnostics(10, {"loss": float("nan")})
            mgr.snapshot_gpu()
            mgr.flush_reward_data(10)
            mgr.generate_report({"loss": 0.5})
            mgr.finish()

    def test_v4_multi_reward_tracking(self):
        from src.student.train_grpo_base import TrackingManager

        with pytest.MonkeyPatch.context() as mp:
            state = self._mock_trackio(mp)

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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest src/tests/test_trackio_dry_run.py -v --tb=short`
Expected: PASS (3 tests)

- [ ] **Step 3: Add test to run_e2e_tests.sh**

In `scripts/run_e2e_tests.sh`, add `test_trackio_dry_run.py` to the GRPO test batch line:

```bash
python3 -m pytest $TEST_DIR/test_grpo_training.py ... $TEST_DIR/test_trackio_dry_run.py -${VERBOSE} --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add src/tests/test_trackio_dry_run.py scripts/run_e2e_tests.sh
git commit -m "test: add dry-run E2E test for full TrackingManager lifecycle"
```

---

### Task 10: Cleanup and final verification

**Files:**
- Modify: `src/student/train_grpo_base.py` (remove old `TrackioCallback` if still present)

- [ ] **Step 1: Verify no remaining references to TrackioCallback**

Run: `grep -rn "TrackioCallback" src/`
Expected: No results (or only in legacy/ deprecated files)

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest src/tests/test_grpo_base.py src/tests/test_trackio_dry_run.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Verify imports work**

Run: `python3 -c "from src.student.train_grpo_base import TrackingManager, TrackingCallback, build_outcome_dataset, build_reward_fn_with_docs; print('All imports OK')"`
Expected: All imports OK

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor: complete Track.io full integration with TrackingManager"
```

---

## Self-Review

**1. Spec coverage:**
- Rich run tracking (init, config, group, auto_log_gpu) - Task 1
- Alert diagnostics (NaN loss, divergence, stall, reward collapse, KL, short completions, milestones) - Task 4
- Markdown reports - Task 5
- System metrics enhancement (GPU snapshot) - Task 5
- Reward Tables - Task 3
- Reward Histograms - Task 3
- Completion Trace - Task 3
- report_to="trackio" in GRPOConfig - Tasks 7-8
- Single file for all config (alert thresholds, etc.) - TrackingManager class constants
- Dry-run E2E test - Task 9
- All trackio.* calls wrapped in try/except - all methods

**2. Placeholder scan:** No TBD, TODO, or "implement later" found. All code blocks are complete.

**3. Type consistency:**
- `TrackingManager.init()` signature consistent across all tasks
- `wrap_reward_fn` returns TRL-compatible callable - used consistently in Tasks 2, 7, 8, 9
- `flush_reward_data` clears `_reward_samples` and `_reward_table_rows` - matches Task 6 callback usage
- `check_diagnostics` takes `(step, logs)` - matches callback and test usage
- `report_to="trackio"` added to both config files

No issues found.
