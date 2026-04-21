"""Pure Triton entrypoint for Mamba-3 MIMO combined kernel."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_triton import mamba3_mimo_triton


_VALID_BACKENDS = {"auto", "triton"}


def _resolve_backend(backend: str | None) -> str:
    backend = (backend or "triton").lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unsupported Mamba3 MIMO backend '{backend}'. "
            f"Valid backends for pure Triton prefill are: {sorted(_VALID_BACKENDS)}"
        )
    return "triton"


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
    """Run pure Triton MIMO prefill forward/backward kernel."""
    _resolve_backend(backend)
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
