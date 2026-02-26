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
    """SwiGLU MLP block (Liger-style API).

    Uses fused gate_up_proj (single GEMM) instead of separate gate_proj + up_proj.
    This avoids a torch.compile inductor bug where multiple nn.Linear on the same
    input get incorrectly fused with doubled dimensions → CUBLAS / misaligned errors.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self._hidden = hidden_features
        self.gate_up_proj = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle old checkpoints with separate gate_proj/up_proj weights."""
        gate_key = prefix + "gate_proj.weight"
        fused_key = prefix + "gate_up_proj.weight"
        if gate_key in state_dict and fused_key not in state_dict:
            w_gate = state_dict.pop(gate_key)
            w_up = state_dict.pop(prefix + "up_proj.weight")
            state_dict[fused_key] = torch.cat([w_gate, w_up], dim=0)
            # Handle bias if present
            gate_bias_key = prefix + "gate_proj.bias"
            if gate_bias_key in state_dict:
                b_gate = state_dict.pop(gate_bias_key)
                b_up = state_dict.pop(prefix + "up_proj.bias")
                state_dict[prefix + "gate_up_proj.bias"] = torch.cat([b_gate, b_up], dim=0)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split([self._hidden, self._hidden], dim=-1)
        # split() creates non-contiguous views — inductor generates wrong reinterpret_tensor
        gate, up = gate.contiguous(), up.contiguous()
        return self.down_proj(liger_silu_mul(gate, up))
