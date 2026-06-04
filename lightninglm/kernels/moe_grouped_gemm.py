"""
Optional grouped-GEMM MoE kernel wrapper.

This module provides a thin compatibility layer around external grouped GEMM
packages (for example, Megatron-style grouped GEMM backends). It does not
ship a kernel itself; it only exposes a stable local API.
"""

from __future__ import annotations

from typing import Iterable, List

import torch

try:
    import grouped_gemm as _grouped_gemm
except Exception:
    _grouped_gemm = None


HAS_MOE_GROUPED_GEMM = _grouped_gemm is not None


def _normalize_m_sizes_list(m_sizes: torch.Tensor | Iterable[int]) -> List[int]:
    if isinstance(m_sizes, torch.Tensor):
        values = m_sizes.detach().cpu().tolist()
    else:
        values = list(m_sizes)
    return [int(v) for v in values]


def _normalize_m_sizes_tensor(
    m_sizes: torch.Tensor | Iterable[int], device: torch.device
) -> torch.Tensor:
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
        raise ValueError(
            f"Invalid grouped GEMM shapes: a={tuple(a.shape)}, b={tuple(b.shape)}"
        )

    # Different grouped_gemm variants accept either a 1D tensor or list[int].
    sizes_tensor = _normalize_m_sizes_tensor(m_sizes, torch.device("cpu"))
    sizes_list = _normalize_m_sizes_list(m_sizes)
    ops = getattr(_grouped_gemm, "ops", _grouped_gemm)

    if hasattr(ops, "gmm"):
        try:
            return ops.gmm(a, b, sizes_tensor, trans_b=False)
        except TypeError:
            return ops.gmm(a, b, sizes_list)

    if hasattr(ops, "grouped_gemm"):
        try:
            return ops.grouped_gemm(a, b, sizes_tensor)
        except TypeError:
            return ops.grouped_gemm(a, b, sizes_list)

    raise RuntimeError(
        "Unsupported grouped_gemm API: expected ops.gmm or ops.grouped_gemm."
    )
