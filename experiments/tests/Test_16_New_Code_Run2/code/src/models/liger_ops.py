"""
Vendored Liger-style ops used by recurrence_model_1b.py and recurrence_model_70b.py.

Attribution:
- Project: LinkedIn Liger-Kernel
- Repository: https://github.com/linkedin/Liger-Kernel
- License: Apache-2.0

This file is a self-contained adaptation of the same operator family for this
repo: SwiGLU MLP, SiLU-mul, and rotary application helpers. No fused CE (Test 14: standard CE in train.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def liger_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused math equivalent: SiLU(gate) * up."""
    return F.silu(gate) * up


def liger_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to the last dimension using precomputed cos/sin.
    
    Supports both:
    1. Full-dim cos/sin: shape (..., D), uses interleaved slicing [..., 0::2]
    2. Half-dim cos/sin: shape (..., D/2), uses directly
    """
    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {x.size(-1)}")
    
    D = x.size(-1)
    
    # 1) Robustness: Ensure device and dtype consistency to avoid silent promotion/mismatch
    if cos.device != x.device or sin.device != x.device:
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

    # 2) Shape handling: Support both (..., D) and (..., D/2) conventions
    # This prevents the most common "silent fail" where D/2 is sliced into D/4.
    if cos.shape[-1] == D:
        cos_half = cos[..., 0::2]
        sin_half = sin[..., 0::2]
    elif cos.shape[-1] == D // 2:
        cos_half = cos
        sin_half = sin
    else:
        raise ValueError(
            f"RoPE cos/sin shape mismatch. Expected last dim {D} or {D//2}, "
            f"got cos={cos.shape}, sin={sin.shape} for x={x.shape}"
        )

    # 3) Interleaved RoPE: (x0, x1) -> (x0*c - x1*s, x0*s + x1*c)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rot_even = x_even * cos_half - x_odd * sin_half
    rot_odd = x_even * sin_half + x_odd * cos_half
    
    # Correctness: stack(dim=-1) + reshape recreates the interleaved [..., x0r, x1r, x2r, x3r] layout.
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
