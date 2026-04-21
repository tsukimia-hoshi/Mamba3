"""Triton backend for Mamba-3 MIMO combined op.

Phase 2/3 scope in this change:
- Wire a real Triton execution path for MIMO rank=1 by reusing the existing
  Triton SISO combined kernels.
- Add explicit hardware/dtype policy checks (notably for T4 / SM75).
- Keep behavior strict: no automatic chunk-size fallback or silent degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


@dataclass(frozen=True)
class TritonLaunchPolicy:
    # Phase-3 policy knobs. We keep this explicit for visibility and reproducibility.
    t4_supported_dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.float32)
    recommended_chunk_sizes_t4: tuple[int, ...] = (8, 16, 32)


_POLICY = TritonLaunchPolicy()


def _check_triton_runtime(dtype: torch.dtype, chunk_size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton MIMO backend requires CUDA")

    if chunk_size < 8:
        raise ValueError(f"chunk_size must be >= 8, got {chunk_size}")



def _head_scale(x: Tensor, scale: Tensor) -> Tensor:
    # x: (B, L, H, P), scale: (H, P)
    return x * scale.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)


def mamba3_mimo_triton(
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
    MIMO_Out: Optional[Tensor],
    Angles: Tensor,
    D: Optional[Tensor],
    Z: Optional[Tensor],
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run Mamba3 MIMO via Triton kernels.

    Current support in this phase is strict and explicit:
    - MIMO rank must be 1.
    - Computation routes through Triton SISO combined kernels.
    - No automatic fallback for unsupported configs.
    """
    _check_triton_runtime(dtype, chunk_size)

    if Q.shape[2] != 1:
        raise NotImplementedError(
            "Phase-2 Triton MIMO backend currently supports mimo_rank == 1 only."
        )

    if rotary_dim_divisor not in (2, 4):
        raise ValueError(f"rotary_dim_divisor must be 2 or 4, got {rotary_dim_divisor}")

    # Rank-1 MIMO reduces to elementwise per-head scaling around SISO core.
    v_scale = MIMO_V[:, 0, :]
    v_eff = _head_scale(V, v_scale)

    z_eff = None
    if Z is not None:
        z_scale = MIMO_Z[:, 0, :]
        z_eff = _head_scale(Z, z_scale)

    out = mamba3_siso_combined(
        Q=Q.squeeze(2),
        K=K.squeeze(2),
        V=v_eff,
        ADT=ADT,
        DT=DT,
        Trap=Trap,
        Q_bias=Q_bias.squeeze(1),
        K_bias=K_bias.squeeze(1),
        Angles=Angles,
        D=D,
        Z=z_eff,
        chunk_size=chunk_size,
        Input_States=None,
        return_final_states=return_state,
        angles_cumsum=True,
    )

    if not return_state:
        if MIMO_Out is None:
            return out.unsqueeze(2)  # (B, L, 1, H, P)
        out_scale = MIMO_Out[:, 0, :]
        return _head_scale(out, out_scale)

    y, last_angle, last_state, last_k, last_v, *rest = out
    if MIMO_Out is None:
        y = y.unsqueeze(2)  # (B, L, 1, H, P)
    else:
        out_scale = MIMO_Out[:, 0, :]
        y = _head_scale(y, out_scale)
    last_angle = torch.remainder(last_angle + torch.pi, 2 * torch.pi) - torch.pi
    last_k = last_k.unsqueeze(1)  # (B, 1, H, N) to match MIMO cache layout
    return y, last_angle, last_state, last_k, last_v
