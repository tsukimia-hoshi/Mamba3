"""Backend dispatcher for Mamba-3 MIMO combined kernel.

Phase 1 goal: keep public API stable while enabling runtime backend switching.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import Tensor

from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as _tilelang_mamba3_mimo
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_triton import mamba3_mimo_triton


_VALID_BACKENDS = {"auto", "tilelang", "triton"}


def _resolve_backend(backend: str | None) -> str:
    backend = backend or os.getenv("MAMBA3_MIMO_BACKEND", "auto")
    backend = backend.lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported Mamba3 MIMO backend '{backend}'. Valid backends: {sorted(_VALID_BACKENDS)}")
    # Phase 1: auto currently maps to the stable implementation.
    if backend == "auto":
        return "tilelang"
    return backend


def mamba3_mimo_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    MIMO_V: Tensor,
    MIMO_Z: Tensor,
    MIMO_Out: Tensor,
    Angles: Tensor,
    D: Tensor,
    Z: Tensor,
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
    backend: str | None = None,
) -> Tensor | Tuple[Tensor, Tuple]:
    """Dispatch Mamba3 MIMO combined op to selected backend.

    The call signature intentionally matches the existing Tilelang entrypoint.
    """
    selected = _resolve_backend(backend)

    if selected == "tilelang":
        return _tilelang_mamba3_mimo(
            Q=Q,
            K=K,
            V=V,
            ADT=ADT,
            DT=DT,
            Trap=Trap,
            Q_bias=Q_bias,
            K_bias=K_bias,
            MIMO_V=MIMO_V,
            MIMO_Z=MIMO_Z,
            MIMO_Out=MIMO_Out,
            Angles=Angles,
            D=D,
            Z=Z,
            chunk_size=chunk_size,
            rotary_dim_divisor=rotary_dim_divisor,
            dtype=dtype,
            return_state=return_state,
        )

    if selected == "triton":
        return mamba3_mimo_triton(
            Q=Q,
            K=K,
            V=V,
            ADT=ADT,
            DT=DT,
            Trap=Trap,
            Q_bias=Q_bias,
            K_bias=K_bias,
            MIMO_V=MIMO_V,
            MIMO_Z=MIMO_Z,
            MIMO_Out=MIMO_Out,
            Angles=Angles,
            D=D,
            Z=Z,
            chunk_size=chunk_size,
            rotary_dim_divisor=rotary_dim_divisor,
            dtype=dtype,
            return_state=return_state,
        )

    raise RuntimeError(f"Unexpected backend resolution: {selected}")
