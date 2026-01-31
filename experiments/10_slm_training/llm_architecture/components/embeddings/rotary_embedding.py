"""
Rotary Position Embeddings (RoPE)
=================================

Efficient relative position encoding using rotation matrices.
Standard implementation used by LLaMA, Qwen, Mistral, etc.

Reference: RoFormer (Su et al., 2021)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information by rotating query and key vectors.
    
    Features:
    - Relative position encoding
    - Linear extrapolation beyond training length
    - Efficient computation via complex multiplication
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos and sin
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        
    def _compute_inv_freq(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Compute inverse frequencies for rotation."""
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim
            )
        )
        return inv_freq
    
    def _update_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        """Update cached cos and sin values."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor
            
            # Compute frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq)
            
            # Compute cos and sin: [seq_len, dim]
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None
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
        
        # Update cache if needed
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        if position_ids is not None:
            # Use provided positions
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]
        else:
            # Use sequential positions
            cos = self._cos_cached[:seq_len].unsqueeze(0)
            sin = self._sin_cached[:seq_len].unsqueeze(0)
            
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
    unsqueeze_dim: int = 1
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
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply rotary position embedding to query only."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)


def apply_rotary_pos_emb_k(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
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
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None
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
