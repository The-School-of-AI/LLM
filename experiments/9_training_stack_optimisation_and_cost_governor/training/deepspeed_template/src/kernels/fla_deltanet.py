"""
fla (Flash Linear Attention) Wrapper for DeltaNet
===================================================

Wraps the `chunk_gated_delta_rule` kernel from the fla library
(flash-linear-attention) for use with our GatedDeltaNet architecture.

Handles:
- Layout: accepts (B, T, H, d) directly — no transpose needed
- Float32 upcast for q, k, v, beta (fla's tl.dot requires matching dtypes)
- Parameter mapping: sigmoid alpha -> log-space forget gate g (float32)
- D residual term: D * (q . k) * v (not part of fla's formula)

Memory optimization vs original:
- No transpose + contiguous copy (saves 3 × B*T*H*d*4 bytes)
- Inputs already in (B, T, H, d) — fla's native layout

Requirements:
    pip install fla
"""

import torch
from typing import Optional, Tuple

# Check for fla availability
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_gated_delta_rule = None


def fla_gated_delta_rule(
    q: torch.Tensor,       # [B, T, H, d]
    k: torch.Tensor,       # [B, T, H, d]
    v: torch.Tensor,       # [B, T, H, d]
    alpha: torch.Tensor,   # [B, T, H, 1]  (sigmoid output in [0,1])
    beta: torch.Tensor,    # [B, T, H, 1]  (sigmoid output in [0,1])
    D: torch.Tensor,       # [H]  (per-head residual weight)
    num_heads: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Fused DeltaNet recurrence via fla's chunk_gated_delta_rule kernel.

    Memory-efficient: accepts (B, T, H, d) directly — no transpose needed.
    fla expects (B, T, H, d) layout with head_first=False (default).

    Float32 upcast is required because fla's Triton kernels use tl.dot
    internally, which requires all operands to match dtype. The internal
    A matrix is computed in fp32, so q/k/v/beta must also be fp32.

    Parameter mapping:
        Our alpha (sigmoid output in [0,1]) -> fla g = log(alpha) (log-space, float32)
        Our beta  (sigmoid output in [0,1]) -> fla beta (float32)
        scale=1.0 because our q, k are already L2-normalized

    Args:
        q: Query tensor [B, T, H, d]
        k: Key tensor [B, T, H, d]
        v: Value tensor [B, T, H, d]
        alpha: Decay parameter [B, T, H, 1] (sigmoid output)
        beta: Writing strength [B, T, H, 1] (sigmoid output)
        D: Per-head residual weight [H]
        num_heads: Number of attention heads

    Returns:
        Output tensor [B, T, H, d]
    """
    if not HAS_FLA:
        raise ImportError(
            "fla (flash-linear-attention) is required for fla_gated_delta_rule. "
            "Install with: pip install fla"
        )

    # fla's Triton kernels use tl.dot which requires matching dtypes.
    # Internal A matrix is fp32, so all inputs must be fp32.
    # .float() on a contiguous (B,T,H,d) tensor is a dtype cast — no
    # transpose or reshape needed (saves 3 contiguous copies vs old code).
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    # alpha: (B, T, H, 1) -> (B, T, H) in float32 log-space
    g = torch.log(alpha[:, :, :, 0].float().clamp(min=1e-6))

    # beta: (B, T, H, 1) -> (B, T, H) in float32
    beta_fla = beta[:, :, :, 0].float()

    # Call fla fused kernel — scale=1.0 since q, k are L2-normalized
    o_fla, _ = chunk_gated_delta_rule(
        q_f32, k_f32, v_f32, g, beta_fla,
        scale=1.0,
        output_final_state=False,
        chunk_size=chunk_size,
    )

    # Add D residual: D * (q . k) * v  (in original dtype to save memory)
    D_weight = D.view(1, 1, num_heads, 1)  # (1, 1, H, 1) for (B, T, H, d)
    qk_dot = (q * k).sum(dim=-1, keepdim=True)  # (B, T, H, 1)
    d_residual = D_weight * qk_dot * v  # (B, T, H, d)

    return o_fla.to(q.dtype) + d_residual
