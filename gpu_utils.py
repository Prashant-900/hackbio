"""gpu_utils.py - GPU detection and device management for RL acceleration.

Detects CUDA (NVIDIA), MPS (Apple Silicon), or falls back to CPU.
Provides hardware info for the dashboard GPU indicator.

References:
  PyTorch device management: https://pytorch.org/docs/stable/tensor_attributes.html
"""

from __future__ import annotations

import torch


def get_device(force_cpu: bool = False) -> torch.device:
    """Return the best available compute device."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_info() -> dict:
    """Return GPU hardware details for the dashboard UI."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "device": str(get_device()),
        "gpu_name": None,
        "gpu_memory_mb": None,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = round(props.total_mem / 1e6)
    return info
