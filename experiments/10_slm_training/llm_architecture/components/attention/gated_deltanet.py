"""
Gated DeltaNet - O(N) Linear Attention
=======================================

Implementation based on arXiv:2412.06464 (Dec 2024).

O(N) linear attention with gating and alpha decay for long-context efficiency.
Used for 75% of layers in the hybrid DeltaNet + GSA architecture.

Key components (Equation 10):
    St = St-1(alpha_t(I - beta_t * kt @ kt^T)) + beta_t * vt @ kt^T

- Alpha (alpha_t): Per-head decay parameter controlling state forgetting
- Beta (beta_t): Writing strength controlling update magnitude
- L2 normalization: For Q/K stability (NOT softmax)
- Short convolutions: Local context integration (kernel_size=4)
- FusedRMSNormSwishGate: Output normalization with gating

Reference: Test_Code/model_1b.py lines 446-707
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Helper Modules
# ============================================================================

class ShortConvolution(nn.Module):
    """
    Short convolution layer with causal padding.
    Used in Gated DeltaNet for local context integration.

    Uses depthwise convolution (groups=dim) for efficiency.
    """

    def __init__(self, dim, conv_size=4, activation='silu'):
        super().__init__()
        self.conv_size = conv_size
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=conv_size,
            padding=conv_size - 1,  # Causal padding
            groups=dim  # Depthwise convolution
        )
        self.activation = nn.SiLU() if activation == 'silu' else nn.Identity()

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)
        x = x[:, :, :-(self.conv_size - 1)]  # Remove extra padding for causality
        x = x.transpose(1, 2)  # (B, T, D)
        return self.activation(x)


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class FusedRMSNormSwishGate(nn.Module):
    """
    Fused RMSNorm with Swish gating for output projection.
    Matches official implementation: g * swish(RMSNorm(x))
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)

    def forward(self, x, g):
        # x: (B, T, D), g: (B, T, D)
        x_norm = self.norm(x)
        return g * F.silu(x_norm)


# ============================================================================
# YARN RoPE (self-contained, matching Test_Code)
# ============================================================================

class DeltaNetRotaryEmbedding(nn.Module):
    """
    YARN (Yet Another RoPE extensioN) Rotary Positional Embedding.

    Self-contained YARN implementation for DeltaNet/GSA attention.
    Uses NTK-aware interpolation for scaling base frequency and
    frequency band interpolation for context extension.

    This uses the interleaved apply_rotary pattern (x[..., ::2], x[..., 1::2])
    which differs from the rotate_half pattern used in the existing RoPE modules.

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(self, dim, max_position_embeddings=8192, base=10000,
                 original_max_position_embeddings=8192, scaling_factor=32.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

        # YARN: NTK-aware interpolation
        if max_position_embeddings > original_max_position_embeddings:
            ext_ratio = max_position_embeddings / original_max_position_embeddings
            scaled_base = base * (ext_ratio ** (dim / (dim - 2)))
        else:
            scaled_base = base

        # Compute inverse frequencies with scaled base
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # YARN: Frequency band interpolation parameters
        self.beta_fast = 32
        self.beta_slow = 1

        # Compute interpolation weights (mscale) for each frequency
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        wavelen = 2 * math.pi / freq_extra
        ramp = torch.clamp((wavelen - self.beta_fast) / (self.beta_slow - self.beta_fast), 0, 1)
        self.register_buffer("mscale", ramp)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # YARN: Apply frequency-dependent interpolation
        scale_factor_per_freq = 1.0 + (self.scaling_factor - 1.0) * self.mscale
        t_scaled = t.unsqueeze(-1) / scale_factor_per_freq.unsqueeze(0)

        freqs = t_scaled * self.inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """Apply rotary embedding using interleaved pattern."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat(
            (x1 * cos[..., ::2] - x2 * sin[..., ::2],
             x1 * sin[..., ::2] + x2 * cos[..., ::2]),
            dim=-1
        )


# ============================================================================
# Gated DeltaNet (75% of layers) - O(N) Linear Attention
# ============================================================================

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet - arXiv:2412.06464 (Dec 2024)

    O(N) linear attention with gating and alpha decay for long-context efficiency.
    Essential for 256k context where quadratic attention is prohibitive.

    Key components from paper (Equation 10):
    St = St-1(alpha_t(I - beta_t*kt*kt^T)) + beta_t*vt*kt^T

    - Alpha (alpha_t): Per-head decay parameter controlling state forgetting
    - Beta (beta_t): Writing strength controlling update magnitude
    - L2 normalization: For Q/K stability (NOT softmax)
    - Short convolutions: Local context integration (kernel_size=4)

    Forward signature: forward(x, attention_mask=None) -> Tensor
    Returns single tensor (B, T, hidden_size), NOT a tuple.
    """

    def __init__(self, hidden_size, num_heads, head_dim,
                 max_seq_len=262144, rope_base=10000,
                 rope_original_max=8192, rope_scaling_factor=32.0,
                 conv_size=4, use_output_norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_output_norm = use_output_norm

        key_dim = num_heads * head_dim
        value_dim = num_heads * head_dim

        # Core projections (Q, K, V, output)
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, value_dim, bias=False)  # Output gate
        self.o_proj = nn.Linear(value_dim, hidden_size, bias=False)

        # Gate projections for alpha/beta computation
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)  # Beta writing strength
        self.gk_proj = nn.Linear(hidden_size, num_heads, bias=True)  # For alpha computation

        # Short convolutions for local context
        self.q_conv1d = ShortConvolution(key_dim, conv_size=conv_size, activation='silu')
        self.k_conv1d = ShortConvolution(key_dim, conv_size=conv_size, activation='silu')
        self.v_conv1d = ShortConvolution(value_dim, conv_size=conv_size, activation='silu')

        # Alpha decay parameters (per-head)
        # Paper: A initialized uniform(0, 16), then log for exponential parameterization
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))

        # D parameter for residual connection (per-head)
        self.D = nn.Parameter(torch.ones(num_heads))

        # dt_bias for Mamba-style gating (per-head)
        dt_init_std = 0.01
        dt_bias = torch.rand(num_heads) * 2 * dt_init_std - dt_init_std
        self.dt_bias = nn.Parameter(dt_bias)

        # Rotary embeddings for Q/K with YARN scaling
        self.rotary_emb = DeltaNetRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
            original_max_position_embeddings=rope_original_max,
            scaling_factor=rope_scaling_factor
        )

        # Output normalization with gating
        if use_output_norm:
            self.o_norm = FusedRMSNormSwishGate(head_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following official implementation."""
        for m in [self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        for m in [self.b_proj, self.gk_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None):
        """
        Forward pass implementing Gated Delta Rule with decay.

        Args:
            x: Input tensor (B, T, hidden_size)
            attention_mask: Optional (not used for linear attention, kept for interface compatibility)

        Returns:
            Output tensor (B, T, hidden_size)
        """
        B, T, C = x.shape
        device = x.device

        # 1. Project to Q, K, V, G
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)  # Output gate

        # 2. Apply short convolutions for local context
        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        # 3. Reshape to separate heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        g = g.view(B, T, self.num_heads, self.head_dim)

        # 4. Apply RoPE to Q/K
        if T > self.rotary_emb.cos_cached.size(0):
            self.rotary_emb._set_cos_sin_cache(T)
        cos = self.rotary_emb.cos_cached[:T].unsqueeze(0).unsqueeze(2)
        sin = self.rotary_emb.sin_cached[:T].unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        # 5. L2 normalization (NOT softmax) - Paper Section 3.3
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 6. Compute beta (writing strength) - sigmoid activation
        beta = torch.sigmoid(self.b_proj(x))  # (B, T, num_heads)
        beta = beta.unsqueeze(-1)  # (B, T, num_heads, 1)

        # 7. Compute alpha (decay parameter) - Paper Equation 10
        gk = self.gk_proj(x)  # (B, T, num_heads)
        A = -torch.exp(self.A_log)  # Negative for decay
        alpha = A.view(1, 1, self.num_heads) * F.softplus(gk + self.dt_bias).unsqueeze(-1)
        alpha = torch.sigmoid(alpha)  # (B, T, num_heads, 1)

        # 8. Transpose for computation
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)  # (B, num_heads, T, 1)
        alpha = alpha.transpose(1, 2)

        # 9. Gated Delta Rule with decay (Paper Equation 10)
        # Initialize state
        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                        device=device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]  # (B, num_heads, head_dim)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            beta_t = beta[:, :, t, 0]  # (B, num_heads)
            alpha_t = alpha[:, :, t, 0]  # (B, num_heads)

            # Query current state
            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)

            # Add D residual (direct token contribution)
            o_t = o_t + self.D.view(1, self.num_heads, 1) * (q_t * k_t).sum(dim=-1, keepdim=True) * v_t

            outputs.append(o_t)

            # Update state with gated delta rule
            v_outer = torch.einsum('bhd,bhe->bhde', v_t, k_t)

            alpha_t = alpha_t.view(B, self.num_heads, 1, 1)
            beta_t = beta_t.view(B, self.num_heads, 1, 1)

            S = alpha_t * S + beta_t * v_outer

        # Stack outputs
        o = torch.stack(outputs, dim=2)  # (B, num_heads, T, head_dim)

        # 10. Apply output normalization with gating
        o = o.transpose(1, 2)  # (B, T, num_heads, head_dim)

        if self.use_output_norm:
            o_norm = []
            for h in range(self.num_heads):
                o_h = o[:, :, h, :]
                g_h = g[:, :, h, :]
                o_norm.append(self.o_norm(o_h, g_h))
            o = torch.stack(o_norm, dim=2)
        else:
            o = o * torch.sigmoid(g)

        # 11. Reshape and project to output
        o = o.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o)
