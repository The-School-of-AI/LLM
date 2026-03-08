"""
Vendored Liger-style ops used by recurrence_model_1b.py and recurrence_model_70b.py.

Attribution:
- Project: LinkedIn Liger-Kernel
- Repository: https://github.com/linkedin/Liger-Kernel
- License: Apache-2.0

This file is a self-contained adaptation of the same operator family for this
repo: SwiGLU MLP, SiLU-mul, and rotary application helpers. No fused CE (Test 14: standard CE in train.py).

Triton-fused variants are used on CUDA when available; falls back to PyTorch otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..kernels.triton_silu_mul import fused_silu_mul as _triton_silu_mul
    _HAS_FUSED_SILU_MUL = True
except ImportError:
    _HAS_FUSED_SILU_MUL = False

try:
    from ..kernels.triton_rope import fused_rope as _triton_rope
    _HAS_FUSED_ROPE = True
except ImportError:
    _HAS_FUSED_ROPE = False


def liger_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up — Triton on CUDA, PyTorch fallback."""
    if _HAS_FUSED_SILU_MUL:
        return _triton_silu_mul(gate, up)
    return F.silu(gate) * up


def liger_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation — Triton on CUDA, PyTorch fallback."""
    if _HAS_FUSED_ROPE:
        return _triton_rope(x, cos, sin)

    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {x.size(-1)}")

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]

    rot_even = x_even * cos_half - x_odd * sin_half
    rot_odd = x_even * sin_half + x_odd * cos_half
    return torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x)


class LigerSwiGLUMLP(nn.Module):
    """SwiGLU MLP block (Liger-style API)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(liger_silu_mul(self.gate_proj(x), self.up_proj(x)))
