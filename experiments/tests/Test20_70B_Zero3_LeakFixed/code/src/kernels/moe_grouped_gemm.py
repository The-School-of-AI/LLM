"""
Optional grouped-GEMM MoE kernel wrapper.

This module provides a thin compatibility layer around external grouped GEMM
packages (for example, Megatron-style grouped GEMM backends). It does not
ship a kernel itself; it only exposes a stable local API.
"""

from __future__ import annotations

from typing import Iterable, List

import torch

# Profiler: kernel_region for step-level kernel timing (no-op when profiler inactive)
try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager
    def kernel_region(name: str):
        @contextmanager
        def _noop():
            yield
        return _noop()

try:
    import grouped_gemm as _grouped_gemm
except Exception:
    _grouped_gemm = None


HAS_MOE_GROUPED_GEMM = _grouped_gemm is not None


def _normalize_m_sizes_list(m_sizes: torch.Tensor | Iterable[int]) -> List[int]:
    if isinstance(m_sizes, torch.Tensor):
        # Keep on CPU if already there, otherwise transfer (unavoidable for list API)
        # This is only called if the backend requires list[int] instead of tensor
        if m_sizes.device.type == 'cpu':
            values = m_sizes.detach().tolist()
        else:
            # NOTE: This causes a GPU-CPU sync. Most backends now support tensor m_sizes,
            # so this path is rarely taken in modern setups.
            values = m_sizes.detach().cpu().tolist()
    else:
        values = list(m_sizes)
    return [int(v) for v in values]


def _normalize_m_sizes_tensor(m_sizes: torch.Tensor | Iterable[int], device: torch.device) -> torch.Tensor:
    if isinstance(m_sizes, torch.Tensor):
        out = m_sizes.detach().to(device=device, dtype=torch.int64)
    else:
        out = torch.tensor(list(m_sizes), device=device, dtype=torch.int64)
    return out.contiguous().view(-1)


def moe_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    m_sizes: torch.Tensor | Iterable[int],
) -> torch.Tensor:
    """
    Run grouped GEMM with expert-group sizes.

    Expected shapes:
    - a: [sum(m_sizes), K]
    - b: [E, K, N]
    - output: [sum(m_sizes), N]
    """
    if _grouped_gemm is None:
        raise RuntimeError(
            "grouped_gemm backend is unavailable. Install grouped_gemm / "
            "Megatron-compatible grouped GEMM backend."
        )

    if a.dim() != 2 or b.dim() != 3:
        raise ValueError(f"Invalid grouped GEMM shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")

    # Different grouped_gemm variants accept either a 1D tensor or list[int].
    # Prefer tensor API to avoid GPU-CPU sync.
    ops = getattr(_grouped_gemm, "ops", _grouped_gemm)

    with kernel_region("moe_grouped_gemm"):
        # grouped_gemm backends require batch_sizes on CPU (RuntimeError: Expected batch_sizes.is_cpu() to be true).
        # Passing a CPU tensor; if m_sizes was on GPU this does one GPU->CPU sync per call.
        sizes_cpu = _normalize_m_sizes_tensor(m_sizes, torch.device("cpu"))

        if hasattr(ops, "gmm"):
            return ops.gmm(a, b, sizes_cpu, trans_b=False)

        if hasattr(ops, "grouped_gemm"):
            return ops.grouped_gemm(a, b, sizes_cpu)

    raise RuntimeError("Unsupported grouped_gemm API: expected ops.gmm or ops.grouped_gemm.")
