from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Pick the best available torch device: CUDA, then Metal (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
