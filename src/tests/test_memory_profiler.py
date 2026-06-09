import pytest

from src.utils.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    TrainingMemoryTracker,
    force_memory_cleanup,
    format_vram,
    get_vram_allocated_gb,
    get_vram_peak_gb,
    get_vram_reserved_gb,
)


class TestFormatVRAM:
    def test_formats_zero_bytes(self):
        assert format_vram(0) == "0.00 GB"

    def test_formats_one_gigabyte(self):
        one_gb = 1024 ** 3
        assert format_vram(one_gb) == "1.00 GB"

    def test_formats_fifteen_gigabytes(self):
        fifteen_gb = 15 * (1024 ** 3)
        assert format_vram(fifteen_gb) == "15.00 GB"

    def test_formats_fractional_gigabytes(self):
        fifteen_point_75_gb = 15.75 * (1024 ** 3)
        assert format_vram(fifteen_point_75_gb) == "15.75 GB"


class TestVRAMHelpersNoCUDA:
    def test_get_vram_allocated_gb_returns_zero_without_cuda(self):
        assert get_vram_allocated_gb() == 0.0

    def test_get_vram_reserved_gb_returns_zero_without_cuda(self):
        assert get_vram_reserved_gb() == 0.0

    def test_get_vram_peak_gb_returns_zero_without_cuda(self):
        assert get_vram_peak_gb() == 0.0


class TestMemorySnapshot:
    def test_capture_without_cuda(self):
        snap = MemorySnapshot.capture("test_label")
        assert snap.label == "test_label"
        assert snap.allocated_bytes == 0
        assert snap.reserved_bytes == 0
        assert snap.peak_bytes == 0


class TestTrainingMemoryTracker:
    def test_init(self):
        tracker = TrainingMemoryTracker()
        assert tracker.snapshots == []

    def test_record(self):
        tracker = TrainingMemoryTracker()
        tracker.record(step=1, label="forward", extra={"loss": 0.5})
        assert len(tracker.snapshots) == 1
        snap = tracker.snapshots[0]
        assert snap.step == 1
        assert snap.label == "forward"
        assert snap.extra == {"loss": 0.5}

    def test_summary(self):
        tracker = TrainingMemoryTracker()
        tracker.record(step=0, label="start")
        tracker.record(step=100, label="mid")
        summary = tracker.summary()
        assert isinstance(summary, dict)
        assert "total_snapshots" in summary
        assert summary["total_snapshots"] == 2

    def test_summary_empty_tracker(self):
        tracker = TrainingMemoryTracker()
        summary = tracker.summary()
        assert summary == {"total_snapshots": 0}

    def test_phase_context_manager(self):
        tracker = TrainingMemoryTracker()
        with tracker.phase(10, "training_loop"):
            pass
        assert len(tracker.snapshots) == 2
        assert tracker.snapshots[0].label == "training_loop_start"
        assert tracker.snapshots[1].label == "training_loop_end"


class TestForceMemoryCleanup:
    def test_force_memory_cleanup_no_cuda(self):
        result = force_memory_cleanup()
        assert isinstance(result, dict)
        assert result["before_allocated_gb"] == 0.0
        assert result["after_allocated_gb"] == 0.0
        assert result["freed_allocated_gb"] == 0.0
