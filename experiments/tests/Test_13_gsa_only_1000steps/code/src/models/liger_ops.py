"""
Vendored Liger-style ops used by recurrence_model_1b.py and recurrence_model_70b.py.

Attribution:
- Project: LinkedIn Liger-Kernel
- Repository: https://github.com/linkedin/Liger-Kernel
- License: Apache-2.0

This file is a self-contained adaptation of the same operator family for this
repo: fused linear+CE, SwiGLU MLP, SiLU-mul, and rotary application helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def liger_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused math equivalent: SiLU(gate) * up."""
    return F.silu(gate) * up


def liger_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to the last dimension using precomputed cos/sin."""
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


class LigerFusedLinearCrossEntropyLoss(nn.Module):
    """
    Memory-efficient fused-linear + CE interface.

    This keeps the same external behavior as fused linear+CE paths while staying
    self-contained in this repo by processing token chunks to avoid [B,T,V]
    materialization.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        chunk_size: int = 256,
    ):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.dim() == 3:
            flat_hidden = hidden.reshape(-1, hidden.size(-1))
            flat_targets = targets.reshape(-1)
        elif hidden.dim() == 2:
            flat_hidden = hidden
            flat_targets = targets.reshape(-1)
        else:
            raise ValueError(f"hidden must be rank 2 or 3, got rank {hidden.dim()}")

        if flat_hidden.size(0) != flat_targets.numel():
            raise ValueError(
                f"Token count mismatch: hidden={flat_hidden.size(0)} targets={flat_targets.numel()}"
            )

        total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        total_count = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)

        n_tokens = flat_hidden.size(0)
        for start in range(0, n_tokens, self.chunk_size):
            end = min(start + self.chunk_size, n_tokens)
            chunk_hidden = flat_hidden[start:end]
            chunk_targets = flat_targets[start:end]

            chunk_logits = F.linear(chunk_hidden, weight, bias)
            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                chunk_targets,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss

            if self.ignore_index >= 0:
                chunk_count = (chunk_targets != self.ignore_index).sum(dtype=torch.float32)
            else:
                chunk_count = torch.tensor(
                    chunk_targets.numel(),
                    device=flat_hidden.device,
                    dtype=torch.float32,
                )
            total_count = total_count + chunk_count

        if self.reduction == "sum":
            return total_loss

        return total_loss / total_count.clamp_min(1.0)
