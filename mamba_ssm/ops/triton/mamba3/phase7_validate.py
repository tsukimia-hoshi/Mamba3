"""Phase-7 validator: forward/backward consistency between tilelang and triton backends.

This script focuses on the currently-supported Triton MIMO subset (mimo_rank == 1).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import torch

from mamba_ssm.modules.mamba3 import Mamba3


@dataclass
class ValidationResult:
    output_max_abs: float
    output_mean_abs: float
    grad_max_abs: float
    grad_mean_abs: float


def _collect_param_grads(model: Mamba3) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grads[name] = p.grad.detach().float().clone()
    return grads


def _compare_grad_dict(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    shared = sorted(set(a.keys()).intersection(b.keys()))
    if not shared:
        raise RuntimeError("No shared parameter gradients to compare")

    for k in shared:
        diff = (a[k] - b[k]).abs()
        max_abs = max(max_abs, diff.max().item())
        sum_abs += diff.sum().item()
        count += diff.numel()
    return max_abs, (sum_abs / max(1, count))


def run_validation(args: argparse.Namespace) -> ValidationResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for phase7 validation")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda")

    model_kwargs = dict(
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.headdim,
        expand=args.expand,
        is_mimo=True,
        mimo_rank=1,
        chunk_size=args.chunk_size,
        device=device,
        dtype=dtype,
    )

    model_tile = Mamba3(**model_kwargs, mimo_backend="tilelang").train()
    model_triton = Mamba3(**model_kwargs, mimo_backend="triton").train()
    model_triton.load_state_dict(model_tile.state_dict())

    x_tile = torch.randn(args.batch, args.seqlen, args.d_model, device=device, dtype=dtype, requires_grad=True)
    x_tri = x_tile.detach().clone().requires_grad_(True)

    out_tile = model_tile(x_tile)
    out_tri = model_triton(x_tri)

    out_diff = (out_tile - out_tri).abs().float()

    # Same scalar objective to compare gradients.
    loss_tile = out_tile.float().square().mean()
    loss_tri = out_tri.float().square().mean()
    loss_tile.backward()
    loss_tri.backward()

    grads_tile = _collect_param_grads(model_tile)
    grads_tri = _collect_param_grads(model_triton)
    grad_max_abs, grad_mean_abs = _compare_grad_dict(grads_tile, grads_tri)

    return ValidationResult(
        output_max_abs=out_diff.max().item(),
        output_mean_abs=out_diff.mean().item(),
        grad_max_abs=grad_max_abs,
        grad_mean_abs=grad_mean_abs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-7 forward/backward consistency validator")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--tol-out-max", type=float, default=1e-2)
    parser.add_argument("--tol-grad-max", type=float, default=1e-2)
    args = parser.parse_args()

    result = run_validation(args)
    print(json.dumps(asdict(result), ensure_ascii=False))

    if result.output_max_abs > args.tol_out_max:
        raise AssertionError(
            f"output_max_abs={result.output_max_abs} exceeds tol_out_max={args.tol_out_max}"
        )
    if result.grad_max_abs > args.tol_grad_max:
        raise AssertionError(
            f"grad_max_abs={result.grad_max_abs} exceeds tol_grad_max={args.tol_grad_max}"
        )


if __name__ == "__main__":
    main()
