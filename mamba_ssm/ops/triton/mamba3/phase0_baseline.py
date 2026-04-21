"""Phase-0 baseline collector for Mamba3 MIMO.

This script records forward/backward runtime and peak memory for fixed shape sets.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mamba_ssm.modules.mamba3 import Mamba3


@dataclass
class CaseResult:
    batch: int
    seqlen: int
    d_model: int
    d_state: int
    headdim: int
    mimo_rank: int
    chunk_size: int
    dtype: str
    backend: str
    fwd_ms: float
    bwd_ms: float
    max_mem_mb: float


def _parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _bench_one_case(
    batch: int,
    seqlen: int,
    d_model: int,
    d_state: int,
    headdim: int,
    mimo_rank: int,
    chunk_size: int,
    dtype: torch.dtype,
    backend: str,
    warmup: int,
    iters: int,
) -> CaseResult:
    device = torch.device("cuda")
    model = Mamba3(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        is_mimo=True,
        mimo_rank=mimo_rank,
        chunk_size=chunk_size,
        mimo_backend=backend,
        device=device,
        dtype=dtype,
    ).train()

    x = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup):
        y = model(x)
        loss = y.float().square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times = []
    bwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = y.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        model.zero_grad(set_to_none=True)
        x.grad = None

        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)

    return CaseResult(
        batch=batch,
        seqlen=seqlen,
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        mimo_rank=mimo_rank,
        chunk_size=chunk_size,
        dtype=str(dtype).replace("torch.", ""),
        backend=backend,
        fwd_ms=sum(fwd_times) / len(fwd_times),
        bwd_ms=sum(bwd_times) / len(bwd_times),
        max_mem_mb=torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase-0 baseline metrics for Mamba3 MIMO")
    parser.add_argument("--out", type=Path, default=Path("phase0_baseline.jsonl"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlens", type=str, default="1024,2048")
    parser.add_argument("--chunk-sizes", type=str, default="8,16")
    parser.add_argument("--mimo-ranks", type=str, default="1,2,4")
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--backend", type=str, default="tilelang", choices=["auto", "tilelang", "triton"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for phase0 baseline collection")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    seqlens = _parse_int_list(args.seqlens)
    chunk_sizes = _parse_int_list(args.chunk_sizes)
    mimo_ranks = _parse_int_list(args.mimo_ranks)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for seqlen in seqlens:
            for chunk_size in chunk_sizes:
                for mimo_rank in mimo_ranks:
                    if args.backend == "triton" and mimo_rank != 1:
                        raise ValueError("Phase-2 triton backend currently supports only mimo_rank=1")
                    result = _bench_one_case(
                        batch=args.batch,
                        seqlen=seqlen,
                        d_model=args.d_model,
                        d_state=args.d_state,
                        headdim=args.headdim,
                        mimo_rank=mimo_rank,
                        chunk_size=chunk_size,
                        dtype=dtype,
                        backend=args.backend,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    record = asdict(result)
                    print(json.dumps(record, ensure_ascii=False))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
