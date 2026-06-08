import pytest


class TestFormatVRAM:
    def test_formats_zero_bytes(self):
        from src.utils.memory_profiler import format_vram
        assert format_vram(0) == "0.00 GB"

    def test_formats_one_gigabyte(self):
        from src.utils.memory_profiler import format_vram
        one_gb = 1024 ** 3
        assert format_vram(one_gb) == "1.00 GB"

    def test_formats_fifteen_gigabytes(self):
        from src.utils.memory_profiler import format_vram
        fifteen_gb = 15 * (1024 ** 3)
        assert format_vram(fifteen_gb) == "15.00 GB"

    def test_formats_fractional_gigabytes(self):
        from src.utils.memory_profiler import format_vram
        fifteen_point_75_gb = 15.75 * (1024 ** 3)
        assert format_vram(fifteen_point_75_gb) == "15.75 GB"


class TestVRAMHelpersNoCUDA:
    def test_get_vram_allocated_gb_returns_zero_without_cuda(self):
        from src.utils.memory_profiler import get_vram_allocated_gb
        result = get_vram_allocated_gb()
        assert result == 0.0

    def test_get_vram_reserved_gb_returns_zero_without_cuda(self):
        from src.utils.memory_profiler import get_vram_reserved_gb
        result = get_vram_reserved_gb()
        assert result == 0.0

    def test_get_vram_peak_gb_returns_zero_without_cuda(self):
        from src.utils.memory_profiler import get_vram_peak_gb
        result = get_vram_peak_gb()
        assert result == 0.0


class TestMemorySnapshot:
    def test_capture_without_cuda(self):
        from src.utils.memory_profiler import MemorySnapshot
        snap = MemorySnapshot.capture("test_label")
        assert snap.label == "test_label"
        assert snap.allocated_bytes == 0
        assert snap.reserved_bytes == 0
        assert snap.peak_bytes == 0

    def test_snapshot_fields_present(self):
        from src.utils.memory_profiler import MemorySnapshot
        snap = MemorySnapshot.capture("init")
        assert hasattr(snap, "label")
        assert hasattr(snap, "allocated_bytes")
        assert hasattr(snap, "reserved_bytes")
        assert hasattr(snap, "peak_bytes")


class TestTrainingMemoryTracker:
    def test_init(self):
        from src.utils.memory_profiler import TrainingMemoryTracker
        tracker = TrainingMemoryTracker()
        assert tracker.snapshots == []

    def test_record(self):
        from src.utils.memory_profiler import TrainingMemoryTracker
        tracker = TrainingMemoryTracker()
        tracker.record(step=1, label="forward", extra={"loss": 0.5})
        assert len(tracker.snapshots) == 1
        snap = tracker.snapshots[0]
        assert snap.step == 1
        assert snap.label == "forward"
        assert snap.extra == {"loss": 0.5}

    def test_summary(self):
        from src.utils.memory_profiler import TrainingMemoryTracker
        tracker = TrainingMemoryTracker()
        tracker.record(step=0, label="start")
        tracker.record(step=100, label="mid")
        summary = tracker.summary()
        assert isinstance(summary, dict)
        assert "total_snapshots" in summary
        assert summary["total_snapshots"] == 2

    def test_phase_context_manager(self):
        from src.utils.memory_profiler import TrainingMemoryTracker
        tracker = TrainingMemoryTracker()
        with tracker.phase(10, "training_loop"):
            pass
        assert len(tracker.snapshots) >= 1
