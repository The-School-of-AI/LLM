"""
Rotary Position Embeddings (RoPE)
=================================

Efficient relative position encoding using rotation matrices.
Standard implementation used by LLaMA, Qwen, Mistral, etc.

Reference: RoFormer (Su et al., 2021)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information by rotating query and key vectors.

    Features:
    - Relative position encoding
    - Linear extrapolation beyond training length
    - Efficient computation via complex multiplication
    - torch.compile compatible (pre-computed cache)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute full cos/sin cache for torch.compile compatibility
        self._precompute_cos_sin_cache(max_position_embeddings, device, dtype)

    def _compute_inv_freq(
        self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Compute inverse frequencies for rotation."""
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim)
        )
        return inv_freq

    def _precompute_cos_sin_cache(
        self, max_seq_len: int, device: Optional[torch.device], dtype: torch.dtype
    ):
        """
        Pre-compute full cos/sin cache at initialization.

        This makes the module torch.compile compatible by avoiding
        data-dependent control flow during forward pass.
        """
        # Create position indices
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        # Compute frequencies: [max_seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Compute cos and sin: [max_seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cached = emb.cos().to(dtype)
        sin_cached = emb.sin().to(dtype)

        # Register as buffers for proper device/dtype handling
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for given positions.

        Args:
            x: Input tensor of shape [batch, seq_len, ...] (used for seq_len and device)
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            Tuple of (cos, sin) tensors for rotation
        """
        seq_len = x.shape[1]

        # Ensure cache is on correct device (handles model.to(device) after init)
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        if position_ids is not None:
            # Use provided positions
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            # Use sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dims.

    Split x into two halves and swap with negation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine tensor from RotaryEmbedding
        sin: Sine tensor from RotaryEmbedding
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Add head dimension for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation using the formula:
    # R(x, θ) = x * cos(θ) + rotate_half(x) * sin(θ)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_rotary_pos_emb_q(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply rotary position embedding to query only."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)


def apply_rotary_pos_emb_k(
    k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply rotary position embedding to key only."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)


class RotaryEmbeddingFast(nn.Module):
    """
    Optimized RoPE using complex number operations.

    More efficient for modern hardware by using complex multiplication.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute all frequencies as complex numbers
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings)
        freqs = torch.outer(t, inv_freq)

        # Store as complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply rotary embeddings using complex multiplication.

        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            position_ids: Optional position indices

        Returns:
            Rotated tensor
        """
        seq_len = x.shape[2]

        # Get frequencies for positions
        if position_ids is not None:
            freqs = self.freqs_complex[position_ids]  # [batch, seq, dim/2]
        else:
            freqs = self.freqs_complex[:seq_len]  # [seq, dim/2]

        # Reshape to complex
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Apply rotation via complex multiplication
        freqs = freqs.unsqueeze(1)  # Add head dim
        x_rotated = x_complex * freqs

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x)
