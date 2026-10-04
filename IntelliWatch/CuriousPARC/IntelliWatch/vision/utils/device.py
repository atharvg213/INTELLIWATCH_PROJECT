"""
vision/utils/device.py
======================
Hardware acceleration and device management utilities for IntelliWatch.

Provides robust runtime device detection, compatibility verification, and
safe CPU fallback for NVIDIA GPUs (including new architectures like Blackwell /
RTX 50-series) where PyTorch binaries may lack pre-compiled kernel images.
"""
import functools
import logging
from typing import Any, Dict, Optional, Tuple
import torch

logger = logging.getLogger("intelliwatch.device")

# Cache probe results to avoid repeated GPU kernel dispatch overhead
_CUDA_EXECUTABLE_CACHED: Optional[bool] = None
_CUDA_ERROR_REASON: Optional[str] = None


def probe_cuda_kernel() -> Tuple[bool, Optional[str]]:
    """
    Performs a lightweight runtime kernel execution probe on CUDA:0.
    Directly checks whether the installed PyTorch build has compatible
    compiled kernels (sm_xx) for the installed GPU hardware.

    Returns:
        (is_compatible: bool, error_description: Optional[str])
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available according to torch.cuda.is_available()."

    try:
        # Create a tiny tensor on CUDA and perform a basic operation to invoke a CUDA kernel
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1.0
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:
        err_msg = str(exc)
        logger.warning(
            f"CUDA device detected but kernel execution probe failed: {err_msg}. "
            "Falling back to CPU inference."
        )
        return False, err_msg


def is_cuda_executable(force_reprobe: bool = False) -> bool:
    """
    Checks if CUDA is available AND capable of executing kernels.
    Caches the result after the first probe.
    """
    global _CUDA_EXECUTABLE_CACHED, _CUDA_ERROR_REASON
    if _CUDA_EXECUTABLE_CACHED is None or force_reprobe:
        compatible, reason = probe_cuda_kernel()
        _CUDA_EXECUTABLE_CACHED = compatible
        _CUDA_ERROR_REASON = reason
    return _CUDA_EXECUTABLE_CACHED


def get_device_diagnostics() -> Dict[str, Any]:
    """
    Returns a comprehensive diagnostic dictionary describing the PyTorch environment,
    GPU hardware, CUDA capability, and active execution mode.
    """
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    device_count = torch.cuda.device_count() if cuda_avail else 0

    cc = (0, 0)
    if cuda_avail and device_count > 0:
        try:
            cc = torch.cuda.get_device_capability(0)
        except Exception:
            cc = (0, 0)

    executable = is_cuda_executable()
    optimal_dev = "cuda" if executable else "cpu"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "N/A",
        "cuda_available": cuda_avail,
        "device_count": device_count,
        "device_name": device_name,
        "compute_capability": f"sm_{cc[0]}{cc[1]} (CC {cc[0]}.{cc[1]})" if cuda_avail else "N/A",
        "cuda_kernel_executable": executable,
        "error_reason": _CUDA_ERROR_REASON,
        "optimal_device": optimal_dev,
    }


def resolve_device(requested_device: Optional[str] = None) -> str:
    """
    Resolves the execution device safely.
    - If requested_device is None or 'auto', selects 'cuda' only if it can actually
      execute kernels, otherwise selects 'cpu'.
    - If 'cuda' is explicitly requested but incompatible, logs a clear warning
      and falls back to 'cpu' to prevent crashes.
    - If 'cpu' is requested, returns 'cpu'.

    Returns:
        Device string ('cpu' or 'cuda').
    """
    req = (requested_device or "auto").strip().lower()

    if req in ("cpu", "none"):
        return "cpu"

    if req in ("cuda", "auto", "gpu"):
        if is_cuda_executable():
            return "cuda"
        else:
            diag = get_device_diagnostics()
            logger.warning(
                f"Requested device '{req}' is not executable on GPU [{diag['device_name']}] "
                f"with PyTorch {diag['torch_version']} (Reason: {diag['error_reason']}). "
                "Safely falling back to CPU."
            )
            return "cpu"

    return req
