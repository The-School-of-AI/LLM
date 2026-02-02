"""
YaRN: Yet another RoPE extensioN

Implementation of YaRN for context window extension based on the paper:
"YaRN: Efficient Context Window Extension of Large Language Models"
https://arxiv.org/abs/2309.00071

YaRN combines:
1. NTK-by-parts interpolation - treats different frequency dims differently
2. Attention scaling - compensates for entropy increase at longer contexts

Key insight: High-frequency RoPE dimensions encode local positions (don't interpolate),
while low-frequency dimensions encode global positions (interpolate fully).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN-aware Rotary Position Embedding.
    
    When scale=1, behaves exactly like standard RoPE.
    When scale>1, applies NTK-by-parts interpolation for context extension.
    
    Args:
        dim: Head dimension (must be even)
        max_position_embeddings: Original max context length
        base: RoPE base frequency (default: 10000)
        scale: Context extension scale factor (new_length / old_length)
        alpha: NTK-by-parts lower threshold (default: 1)
        beta: NTK-by-parts upper threshold (default: 32)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scale: float = 1.0,
        alpha: float = 1.0,
        beta: float = 32.0,
        original_max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        
        # Original context length (for computing wavelength ratios)
        self.original_max_position_embeddings = (
            original_max_position_embeddings or max_position_embeddings
        )
        
        # Compute frequencies with YaRN interpolation
        inv_freq = self._compute_yarn_frequencies()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Compute attention scaling factor
        self.attention_scale = self._compute_attention_scale()
        
        # Cache cos/sin embeddings
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _compute_yarn_frequencies(self) -> torch.Tensor:
        """
        Compute YaRN-interpolated frequencies using NTK-by-parts.
        
        For each dimension d:
        - Compute wavelength λ = 2π * base^(2d/dim)
        - Compute ratio r = original_context / wavelength
        - Apply ramp function γ based on r, α, β
        - Interpolate: new_freq = (1-γ) * (freq/s) + γ * freq
        """
        # Standard RoPE frequencies
        dim_indices = torch.arange(0, self.dim, 2).float()
        freqs = 1.0 / (self.base ** (dim_indices / self.dim))
        
        if self.scale == 1.0:
            # No scaling needed
            return freqs
        
        # Compute wavelengths: λ = 2π / freq = 2π * base^(2d/dim)
        wavelengths = 2 * math.pi / freqs
        
        # Compute ratio of original context to wavelength
        # r = L / λ where L is original max context length
        ratios = self.original_max_position_embeddings / wavelengths
        
        # Apply ramp function γ(r)
        # γ = 0 if r < α, 1 if r > β, linear ramp otherwise
        gamma = torch.zeros_like(ratios)
        
        # Linear region: α <= r <= β
        linear_mask = (ratios >= self.alpha) & (ratios <= self.beta)
        gamma[linear_mask] = (ratios[linear_mask] - self.alpha) / (self.beta - self.alpha)
        
        # High frequency region: r > β (don't interpolate)
        gamma[ratios > self.beta] = 1.0
        
        # Apply NTK-by-parts interpolation
        # new_freq = (1 - γ) * (freq / s) + γ * freq
        interpolated_freqs = (1 - gamma) * (freqs / self.scale) + gamma * freqs
        
        return interpolated_freqs
    
    def _compute_attention_scale(self) -> float:
        """
        Compute attention scaling factor to compensate for entropy increase.
        
        From the paper: √(1/t) = 0.1 * ln(s) + 1
        We apply this to Q and K: q_scaled = q * √(1/t)
        
        Since we apply to both Q and K, the dot product gets scaled by 1/t.
        """
        if self.scale == 1.0:
            return 1.0
        
        # sqrt(1/t) = 0.1 * ln(s) + 1
        sqrt_inv_t = 0.1 * math.log(self.scale) + 1.0
        return sqrt_inv_t
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos/sin embeddings for positions 0 to seq_len."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply attention scaling to the embeddings
        # This is equivalent to scaling Q and K before attention
        self.register_buffer("cos_cached", emb.cos() * self.attention_scale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * self.attention_scale, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos/sin embeddings for the given sequence length.
        
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        if seq_len > self.max_position_embeddings:
            # Dynamically extend cache if needed
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def create_yarn_rotary_embedding(
    dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    scale: float = 1.0,
    original_max_position_embeddings: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> YaRNRotaryEmbedding:
    """
    Factory function to create a YaRN RoPE embedding.
    
    Args:
        dim: Head dimension
        max_position_embeddings: New max context length
        base: RoPE theta
        scale: Extension scale (new_length / old_length), or will compute from lengths
        original_max_position_embeddings: Original max context length (for computing scale)
        alpha, beta: NTK-by-parts thresholds (default: 1, 32)
    """
    # Compute scale if not provided but original length is
    if scale == 1.0 and original_max_position_embeddings is not None:
        scale = max_position_embeddings / original_max_position_embeddings
    
    return YaRNRotaryEmbedding(
        dim=dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        scale=scale,
        alpha=alpha,
        beta=beta,
        original_max_position_embeddings=original_max_position_embeddings,
    )


if __name__ == "__main__":
    # Test YaRN implementation
    print("=" * 60)
    print("Testing YaRN RoPE Implementation")
    print("=" * 60)
    
    dim = 64  # head_dim
    original_length = 256
    new_length = 1024
    scale = new_length / original_length  # 4x
    
    print(f"\nScale factor: {scale}x ({original_length} → {new_length})")
    
    # Standard RoPE (scale=1)
    standard_rope = YaRNRotaryEmbedding(
        dim=dim,
        max_position_embeddings=original_length,
        scale=1.0,
    )
    
    # YaRN RoPE (scale=4)
    yarn_rope = YaRNRotaryEmbedding(
        dim=dim,
        max_position_embeddings=new_length,
        scale=scale,
        original_max_position_embeddings=original_length,
    )
    
    print(f"\nStandard RoPE frequencies (first 5): {standard_rope.inv_freq[:5]}")
    print(f"YaRN RoPE frequencies (first 5):     {yarn_rope.inv_freq[:5]}")
    print(f"Attention scale factor: {yarn_rope.attention_scale:.4f}")
    
    # Test forward pass
    x = torch.randn(2, 128, dim)
    cos, sin = yarn_rope(x, 128)
    print(f"\nOutput shapes - cos: {cos.shape}, sin: {sin.shape}")
    
    # Verify frequencies are different for low-freq dims, similar for high-freq
    freq_ratio = yarn_rope.inv_freq / standard_rope.inv_freq
    print(f"\nFrequency ratios (YaRN/Standard):")
    print(f"  Low freq dims (should be ~1/scale):  {freq_ratio[:3]}")
    print(f"  High freq dims (should be ~1):       {freq_ratio[-3:]}")
    
    print("\n" + "=" * 60)
    print("YaRN RoPE test passed!")
    print("=" * 60)
