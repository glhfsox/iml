"""Pick the best available device: AMD GPU via HIP (ROCm) or NVIDIA via CUDA -> Apple MPS -> CPU.

PyTorch exposes the HIP backend under the ``"cuda"`` device name (``torch.device("hip")``
does not exist), so on the ROCm server ``get_device()`` returns ``cuda`` and the AMD card is
used. ``torch.version.hip`` tells the two apart; env.json records it.
"""

from __future__ import annotations

import platform
from typing import Any

import torch

GPU = "cuda"  # device string PyTorch uses for both HIP (ROCm) and CUDA
_ALIASES = {"hip": GPU, "rocm": GPU, "gpu": GPU}


def is_hip() -> bool:
    """True when this torch build runs on ROCm (AMD GPU through HIP)."""
    return getattr(torch.version, "hip", None) is not None


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``torch.device``; ``prefer`` forces a backend ("hip"/"cuda", "mps", "cpu")."""
    if prefer is not None:
        return torch.device(_ALIASES.get(prefer.lower(), prefer))
    if torch.cuda.is_available():  # HIP on ROCm builds, CUDA on NVIDIA builds
        return torch.device(GPU)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict[str, Any]:
    """Human-readable facts about the device, saved into every run's env.json."""
    info: dict[str, Any] = {
        "device": str(device),
        "backend": ("hip" if is_hip() else "cuda") if device.type == GPU else device.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if device.type == GPU:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["hip"] = getattr(torch.version, "hip", None)  # ROCm version on AMD builds
        info["cuda"] = torch.version.cuda  # None on ROCm builds
    return info
