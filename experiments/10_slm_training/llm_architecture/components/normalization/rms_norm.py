"""
RMS Normalization
=================

Root Mean Square Layer Normalization as used in modern LLMs.
More efficient than LayerNorm - no mean subtraction, no bias.

Used by: LLaMA, Qwen, Mistral, DeepSeek, etc.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

    More efficient than LayerNorm:
    - No mean computation
    - No bias term
    - Comparable performance
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        # x^2
        variance = x.pow(2).mean(-1, keepdim=True)
        # x * rsqrt(variance + eps)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized tensor of same shape
        """
        output = self._norm(x.float()).type_as(x)

        if self.weight is not None:
            output = output * self.weight

        return output

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


class FusedRMSNorm(nn.Module):
    """
    Fused RMS Normalization for better performance.

    Uses torch.compile or custom CUDA kernels when available.
    Falls back to standard implementation otherwise.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Try to use optimized implementation
        self._use_fused = self._check_fused_available()

    def _check_fused_available(self) -> bool:
        """Check if fused implementation is available."""
        try:
            # Check for flash-attn's RMSNorm
            import flash_attn.ops.rms_norm as flash_attn_rms_norm

            return hasattr(flash_attn_rms_norm, "rms_norm")
        except ImportError:
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fused and x.is_cuda:
            from flash_attn.ops.rms_norm import rms_norm

            return rms_norm(x, self.weight, self.eps)
        else:
            # Fallback to standard implementation
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            return x_normed * self.weight
