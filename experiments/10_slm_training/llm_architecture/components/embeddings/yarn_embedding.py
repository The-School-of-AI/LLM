"""
YaRN: Yet another RoPE extensioN
================================

Extends RoPE to handle longer context lengths than seen during training.
Combines NTK-aware interpolation with attention scaling.

Reference: "YaRN: Efficient Context Window Extension of Large Language Models"
           (Peng et al., 2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN: Yet another RoPE extensioN.

    Enables context length extension beyond training length through:
    1. NTK-aware frequency scaling
    2. Dynamic temperature adjustment
    3. Attention scaling compensation

    Key parameters:
    - scale: Target context extension factor (e.g., 8 for 4k->32k)
    - beta_fast/beta_slow: Frequency interpolation bounds
    - mscale: Attention scaling factor

    Note: This implementation is torch.compile compatible by pre-computing
    the full cos/sin cache at initialization.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        original_max_position: int = 4096,
        scale: float = 8.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position = original_max_position
        self.scale = scale
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Compute YaRN frequencies
        inv_freq = self._compute_yarn_inv_freq(device, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Compute attention scaling
        self.attn_scale = self._compute_attn_scale()

        # Pre-compute full cos/sin cache for torch.compile compatibility
        # This avoids data-dependent control flow during forward pass
        self._precompute_cos_sin_cache(max_position_embeddings, device, dtype)

    def _compute_yarn_inv_freq(
        self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Compute YaRN-adjusted inverse frequencies.

        Uses NTK-aware interpolation with frequency mixing.
        """
        # Original RoPE frequencies
        dim_range = torch.arange(0, self.dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.base ** (dim_range / self.dim))

        # Compute wavelengths
        self.original_max_position / self.beta_fast
        self.original_max_position / self.beta_slow

        # Compute interpolation factors for each dimension
        wavelen = 2 * math.pi / inv_freq

        # Ramp function: smooth transition between no-scaling and full-scaling
        ramp = (self.original_max_position / wavelen - self.beta_fast) / (
            self.beta_slow - self.beta_fast
        )
        ramp = ramp.clamp(0, 1)

        # Apply NTK-aware interpolation
        # Low frequencies (high wavelength): interpolate
        # High frequencies (low wavelength): extrapolate
        inv_freq_interpolated = inv_freq / self.scale
        inv_freq_scaled = inv_freq_interpolated * (1 - ramp) + inv_freq * ramp

        return inv_freq_scaled

    def _compute_attn_scale(self) -> float:
        """
        Compute attention scaling factor.

        Compensates for the attention distribution change due to position scaling.
        """
        if self.mscale_all_dim > 0:
            # Scale based on full dimension
            return math.pow(self.scale, self.mscale_all_dim)
        elif self.mscale > 0:
            # Standard YaRN scaling
            return 0.1 * math.log(self.scale) + 1.0
        else:
            return 1.0

    def _precompute_cos_sin_cache(
        self, max_seq_len: int, device: Optional[torch.device], dtype: torch.dtype
    ):
        """
        Pre-compute full cos/sin cache at initialization.

        This makes the module torch.compile compatible by avoiding
        data-dependent control flow during forward pass.
        """
        # Position indices
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Apply mscale to cos/sin (attention temperature adjustment)
        cos_cached = (emb.cos() * self.attn_scale).to(dtype)
        sin_cached = (emb.sin() * self.attn_scale).to(dtype)

        # Register as buffers for proper device/dtype handling
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get YaRN rotary embeddings.

        Args:
            x: Input tensor for shape/device info
            position_ids: Optional position indices

        Returns:
            Tuple of (cos, sin) with YaRN adjustments
        """
        seq_len = x.shape[1]

        # Ensure cache is on correct device (handles model.to(device) after init)
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0)

        return cos, sin

    def get_attn_scale(self) -> float:
        """Get the attention scaling factor for use in attention."""
        return self.attn_scale


class DynamicYaRNEmbedding(nn.Module):
    """
    Dynamic YaRN that automatically scales based on input length.

    Useful when input lengths vary significantly and you want
    optimal scaling for each sequence length.

    Note: This version is NOT torch.compile compatible due to dynamic
    frequency computation. Use YaRNRotaryEmbedding for torch.compile support.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        # Base frequencies (no scaling)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute base cos/sin for sequences <= original_max_position
        self._precompute_base_cache(original_max_position)

    def _precompute_base_cache(self, max_len: int):
        """Pre-compute cache for base (unscaled) positions."""
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("base_cos", emb.cos(), persistent=False)
        self.register_buffer("base_sin", emb.sin(), persistent=False)

    @torch.compiler.disable  # Disable torch.compile for this method due to dynamic control flow
    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic YaRN embeddings based on sequence length.
        """
        seq_len = x.shape[1]

        # Dynamic scale based on sequence length
        if seq_len > self.original_max_position:
            scale = seq_len / self.original_max_position

            # Recompute frequencies with dynamic scale
            inv_freq = self._compute_dynamic_inv_freq(scale, x.device)
            attn_scale = 0.1 * math.log(scale) + 1.0 if self.mscale > 0 else 1.0

            # Compute positions
            if position_ids is None:
                t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            else:
                t = position_ids.float()

            freqs = torch.outer(t.view(-1), inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)

            cos = (emb.cos() * attn_scale).to(x.dtype)
            sin = (emb.sin() * attn_scale).to(x.dtype)
        else:
            # Use pre-computed base cache
            if position_ids is None:
                cos = self.base_cos[:seq_len].to(x.dtype)
                sin = self.base_sin[:seq_len].to(x.dtype)
            else:
                cos = self.base_cos[position_ids].to(x.dtype)
                sin = self.base_sin[position_ids].to(x.dtype)

            if position_ids is not None:
                return cos, sin
            else:
                return cos.unsqueeze(0), sin.unsqueeze(0)

        if position_ids is not None:
            cos = cos.view(*position_ids.shape, -1)
            sin = sin.view(*position_ids.shape, -1)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return cos, sin

    def _compute_dynamic_inv_freq(
        self, scale: float, device: torch.device
    ) -> torch.Tensor:
        """Compute frequencies with dynamic scale."""
        dim_range = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (dim_range / self.dim))

        wavelen = 2 * math.pi / inv_freq

        ramp = (self.original_max_position / wavelen - self.beta_fast) / (
            self.beta_slow - self.beta_fast
        )
        ramp = ramp.clamp(0, 1)

        inv_freq_interpolated = inv_freq / scale
        inv_freq_scaled = inv_freq_interpolated * (1 - ramp) + inv_freq * ramp

        return inv_freq_scaled
