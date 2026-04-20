"""
VRAM Monitoring Utilities

Monitor GPU VRAM usage during training to ensure we stay within
RTX 5090 (32GB) limits.
"""

from contextlib import contextmanager


def _get_torch():
    """Lazy import of torch."""
    import torch

    return torch


VRAM_LIMIT_GB = 30  # Keep under 32GB GPU capacity


class VRAMMonitor:
    """Context manager for monitoring GPU VRAM usage."""

    def __init__(self):
        self.peak_vram_gb = 0.0
        self.initial_vram_gb = 0.0

    def __enter__(self):
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.initial_vram_gb = self._get_current_vram_gb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    def _get_current_vram_gb(self) -> float:
        """Get current VRAM usage in GB."""
        torch = _get_torch()
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        return allocated

    def check_peak(self):
        """Update and return peak VRAM usage."""
        torch = _get_torch()
        if torch.cuda.is_available():
            current = self._get_current_vram_gb()
            self.peak_vram_gb = max(self.peak_vram_gb, current)
        return self.peak_vram_gb

    def is_under_limit(self) -> bool:
        """Check if peak VRAM is under the limit."""
        return self.peak_vram_gb < VRAM_LIMIT_GB

    def get_available_vram_gb(self) -> float:
        """Get available VRAM in GB."""
        torch = _get_torch()
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total - self.peak_vram_gb


@contextmanager
def vram_monitor():
    """Context manager for monitoring VRAM during operations."""
    monitor = VRAMMonitor()
    try:
        yield monitor
    finally:
        monitor.check_peak()
