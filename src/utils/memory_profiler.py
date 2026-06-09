import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


def format_vram(bytes_val: float) -> str:
    """Format bytes as human-readable GB string."""
    gb = bytes_val / (1024 ** 3)
    return f"{gb:.2f} GB"


def get_vram_allocated_gb() -> float:
    """Return currently allocated VRAM in GB. Returns 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_vram_reserved_gb() -> float:
    """Return currently reserved VRAM in GB. Returns 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024 ** 3)


def get_vram_peak_gb() -> float:
    """Return peak VRAM usage in GB. Returns 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


@dataclass
class MemorySnapshot:
    """Point-in-time VRAM snapshot."""
    label: str
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_bytes: int = 0

    @classmethod
    def capture(cls, label: str) -> "MemorySnapshot":
        """Capture current VRAM stats. Returns zeros if no CUDA."""
        if torch.cuda.is_available():
            return cls(
                label=label,
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        return cls(label=label)


@dataclass
class TrackerSnapshot:
    """Snapshot with step and optional metadata."""
    step: int
    label: str
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingMemoryTracker:
    """Track VRAM usage across training steps."""

    def __init__(self):
        self.snapshots: List[TrackerSnapshot] = []

    def record(self, step: int, label: str, extra: Optional[Dict[str, Any]] = None):
        """Record a VRAM snapshot at a given step."""
        base = MemorySnapshot.capture(label)
        self.snapshots.append(TrackerSnapshot(
            step=step,
            label=label,
            allocated_bytes=base.allocated_bytes,
            reserved_bytes=base.reserved_bytes,
            peak_bytes=base.peak_bytes,
            extra=extra or {},
        ))

    @contextmanager
    def phase(self, step: int, label: str):
        """Context manager that records VRAM on entry and exit of a phase."""
        self.record(step, f"{label}_start")
        try:
            yield
        finally:
            self.record(step, f"{label}_end")

    def summary(self) -> Dict[str, Any]:
        """Return summary dict of tracked memory."""
        if not self.snapshots:
            return {"total_snapshots": 0}

        peak_allocated = max(s.allocated_bytes for s in self.snapshots)
        peak_reserved = max(s.reserved_bytes for s in self.snapshots)
        peak_peak = max(s.peak_bytes for s in self.snapshots)

        return {
            "total_snapshots": len(self.snapshots),
            "peak_allocated_gb": format_vram(peak_allocated),
            "peak_reserved_gb": format_vram(peak_reserved),
            "peak_peak_gb": format_vram(peak_peak),
            "labels": [s.label for s in self.snapshots],
        }


class MemoryProfiler:
    """Wrapper around torch.profiler with memory tracking enabled."""

    def __init__(self, *args, **kwargs):
        self.profiler = torch.profiler.profile(
            *args,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            **kwargs,
        )

    def start(self) -> None:
        self.profiler.start()

    def stop(self) -> None:
        self.profiler.stop()

    def step(self) -> None:
        self.profiler.step()

    def key_averages(self, group_by_input_n: bool = False):
        return self.profiler.key_averages(group_by_input_n=group_by_input_n)

    def export_chrome_trace(self, path: str) -> None:
        self.profiler.export_chrome_trace(path)

    def __enter__(self) -> "MemoryProfiler":
        self.profiler.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.profiler.__exit__(*args)


def force_memory_cleanup() -> Dict[str, float]:
    """Force garbage collection and CUDA cache clear. Returns before/after VRAM."""
    before_allocated = get_vram_allocated_gb()
    before_reserved = get_vram_reserved_gb()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    after_allocated = get_vram_allocated_gb()
    after_reserved = get_vram_reserved_gb()

    return {
        "before_allocated_gb": before_allocated,
        "before_reserved_gb": before_reserved,
        "after_allocated_gb": after_allocated,
        "after_reserved_gb": after_reserved,
        "freed_allocated_gb": before_allocated - after_allocated,
        "freed_reserved_gb": before_reserved - after_reserved,
    }
