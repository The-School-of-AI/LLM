"""
SwiGLU Feed-Forward Network
============================

Gated linear unit with Swish activation.
Standard FFN for modern LLMs (LLaMA, Qwen, Mistral).

SwiGLU formula:
    FFN(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down
    
where SiLU(x) = x * sigmoid(x)

Benefits:
- Better gradient flow than GELU
- Improved training stability
- Standard in all major LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Uses gated linear units with Swish (SiLU) activation.

    Architecture:
        input -> [gate_proj, up_proj] -> SiLU(gate) * up -> down_proj -> output

    Typical dimensions:
    - hidden_size: 2048
    - intermediate_size: 5504 (~2.7x hidden for optimal SwiGLU)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate projection: x -> gate
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Up projection: x -> up
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Down projection: intermediate -> hidden
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU: SiLU(gate) * up, then project down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        hidden = gate * up
        hidden = self.dropout(hidden)

        output = self.down_proj(hidden)
        return output


class GeGLUFFN(nn.Module):
    """
    GEGLU Feed-Forward Network.

    Variant using GELU instead of SiLU.
    Used in some models like PaLM.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network (non-gated).

    Traditional FFN: up -> activation -> down
    Included for comparison/ablation studies.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }.get(activation, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class FusedSwiGLUFFN(nn.Module):
    """
    Fused SwiGLU for better performance.

    Combines gate and up projections into single matrix multiply.
    More efficient on GPU but uses 2x memory for projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Fused gate+up projection
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul for gate and up
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # SiLU activation on gate, multiply with up
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)

        return self.down_proj(hidden)
