"""Runtime dtype policy helpers for Mamba-3 Triton/Tilelang paths."""

from __future__ import annotations

import torch


def supports_cuda_dtype(dtype: torch.dtype, capability: tuple[int, int]) -> bool:
    """Return whether a CUDA device capability supports `dtype` for Mamba-3 kernels."""
    major, minor = capability

    if dtype in (torch.float16, torch.float32):
        return True

    if dtype == torch.bfloat16:
        return (major, minor) >= (8, 0)

    # Float8 families are Hopper+ in practice for our kernel stack.
    float8_dtypes = {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }
    float8_dtypes.discard(None)
    if dtype in float8_dtypes:
        return (major, minor) >= (9, 0)

    return False


def enforce_cuda_dtype_policy(dtype: torch.dtype, *, op_name: str) -> None:
    """Enforce runtime dtype policy for CUDA execution.

    Policy:
    - On T4 (SM75): only fp16/fp32 are accepted.
    - Other dtypes are accepted only on hardware that supports them.
    """
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability == (7, 5) and dtype not in (torch.float16, torch.float32):
        raise TypeError(
            f"{op_name}: T4 (SM75) only supports fp16/fp32, got dtype={dtype}."
        )

    if not supports_cuda_dtype(dtype, capability):
        raise TypeError(
            f"{op_name}: dtype={dtype} is not supported on CUDA capability {capability}."
        )
