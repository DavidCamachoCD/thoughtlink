"""Device selection helper used by the in-process trainers.

Picks CUDA > MPS > CPU. MPS support is required for Apple Silicon Macs --
without it, the fallback CUDA-or-CPU logic stays on CPU and slows EEGNet
training ~5-10x. Lives in its own module so tests and other call sites can
import it without dragging the heavier `eval.training` import chain.
"""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Return the best available torch device on this host."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
