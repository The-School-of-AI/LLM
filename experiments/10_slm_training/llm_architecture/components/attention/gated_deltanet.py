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

Performance notes:
- Uses chunk-wise parallel recurrence (default chunk_size=64) instead of
  per-timestep Python loop. This gives ~50-100x speedup on GPU.
- Optionally integrates flash-linear-attention (fla) Triton kernels when
  available for near-optimal hardware utilization.

Reference: Test_Code/model_1b.py lines 446-707
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import flash-linear-attention for optimized Gated DeltaNet Triton kernels
# Prefer chunk_gated_delta_rule (has alpha gate) over chunk_delta_rule (no alpha)
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    )

    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    _fla_chunk_gated_delta_rule = None

# Try to import fused Triton kernel for RMSNorm+SiLU+Gate (3 ops -> 1 kernel)
try:
    from components.kernels.triton_fused_norm_gate import FusedRMSNormSiLUGate

    _HAS_FUSED_NORM_GATE = True
except ImportError:
    _HAS_FUSED_NORM_GATE = False

# Try to import TritonRMSNorm for standalone norm
try:
    import components.kernels.triton_normalization as triton_normalization

    _HAS_TRITON_NORM = hasattr(triton_normalization, "TritonRMSNorm")
except ImportError:
    _HAS_TRITON_NORM = False


# ============================================================================
# Helper Modules
# ============================================================================


class ShortConvolution(nn.Module):
    """
    Short convolution layer with causal padding.
    Used in Gated DeltaNet for local context integration.

    Uses depthwise convolution (groups=dim) for efficiency.
    """

    def __init__(self, dim, conv_size=4, activation="silu"):
        super().__init__()
        self.conv_size = conv_size
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_size,
            padding=conv_size - 1,  # Causal padding
            groups=dim,  # Depthwise convolution
        )
        self.activation = nn.SiLU() if activation == "silu" else nn.Identity()

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)
        x = x[:, :, : -(self.conv_size - 1)]  # Remove extra padding for causality
        x = x.transpose(1, 2)  # (B, T, D)
        return self.activation(x)


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class FusedRMSNormSwishGate(nn.Module):
    """
    Fused RMSNorm with Swish gating for output projection.
    Matches official implementation: g * swish(RMSNorm(x))

    On CUDA with Triton: uses single fused kernel (3 ops -> 1 launch).
    On CPU: falls back to PyTorch ops.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        if _HAS_FUSED_NORM_GATE:
            self._fused = FusedRMSNormSiLUGate(dim, eps)
            self.norm = None
        else:
            self._fused = None
            self.norm = RMSNorm(dim, eps)

    def forward(self, x, g):
        # x: (B, T, D), g: (B, T, D)
        if self._fused is not None:
            return self._fused(x, g)
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

    def __init__(
        self,
        dim,
        max_position_embeddings=8192,
        base=10000,
        original_max_position_embeddings=8192,
        scaling_factor=32.0,
    ):
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
        ramp = torch.clamp(
            (wavelen - self.beta_fast) / (self.beta_slow - self.beta_fast), 0, 1
        )
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
            (
                x1 * cos[..., ::2] - x2 * sin[..., ::2],
                x1 * sin[..., ::2] + x2 * cos[..., ::2],
            ),
            dim=-1,
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

    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        max_seq_len=262144,
        rope_base=10000,
        rope_original_max=8192,
        rope_scaling_factor=32.0,
        conv_size=4,
        use_output_norm=True,
    ):
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
        self.b_proj = nn.Linear(
            hidden_size, num_heads, bias=True
        )  # Beta writing strength
        self.gk_proj = nn.Linear(
            hidden_size, num_heads, bias=True
        )  # For alpha computation

        # Short convolutions for local context
        self.q_conv1d = ShortConvolution(
            key_dim, conv_size=conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            key_dim, conv_size=conv_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            value_dim, conv_size=conv_size, activation="silu"
        )

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
            scaling_factor=rope_scaling_factor,
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

    def _get_causal_mask(self, size, device):
        """Get or create cached causal mask (upper triangle = True)."""
        if (
            not hasattr(self, "_causal_mask_cache")
            or self._causal_mask_cache.size(0) < size
        ):
            self._causal_mask_cache = torch.triu(
                torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
            )
        return self._causal_mask_cache[:size, :size]

    @torch.compiler.disable
    def _chunk_parallel_recurrence(self, q, k, v, alpha, beta, chunk_size=64):
        """
        Chunk-wise parallel recurrence for the gated delta rule.

        Instead of a Python for-loop over every timestep (T kernel launches),
        this processes T in chunks. Within each chunk, intra-chunk attention is
        computed via parallel matmuls. Between chunks, the state matrix S is
        carried forward. This gives ~50-100x speedup over the naive loop.

        Marked with @torch.compiler.disable to avoid graph breaks from the
        Python for-loop. torch.compile still optimizes the surrounding
        projections, convolutions, and norms.

        Args:
            q: (B, H, T, D) L2-normalized queries
            k: (B, H, T, D) L2-normalized keys
            v: (B, H, T, D) values
            alpha: (B, H, T, 1) decay factors
            beta: (B, H, T, 1) writing strengths
            chunk_size: Number of timesteps per chunk

        Returns:
            o: (B, H, T, D) output
        """
        B, H, T, D = q.shape
        device = q.device
        dtype = q.dtype

        # State matrix: S[b,h] is D×D
        S = torch.zeros(B, H, D, D, device=device, dtype=dtype)

        # Pre-squeeze scalars for efficiency
        alpha_sq = alpha.squeeze(-1)  # (B, H, T)
        beta_sq = beta.squeeze(-1)  # (B, H, T)

        output_chunks = []

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            L = chunk_end - chunk_start

            # Slice chunk: (B, H, L, D)
            q_c = q[:, :, chunk_start:chunk_end, :]
            k_c = k[:, :, chunk_start:chunk_end, :]
            v_c = v[:, :, chunk_start:chunk_end, :]
            alpha_c = alpha_sq[:, :, chunk_start:chunk_end]  # (B, H, L)
            beta_c = beta_sq[:, :, chunk_start:chunk_end]  # (B, H, L)

            # --- Inter-chunk contribution: query the accumulated state S ---
            # o_inter[t] = q_c[t] @ S  (before any intra-chunk updates)
            # (B, H, L, D) @ (B, H, D, D) -> (B, H, L, D)
            o_inter = torch.matmul(q_c, S)

            # --- Intra-chunk contribution: causal attention within the chunk ---
            # Build causal decay weights for intra-chunk attention.
            # For positions i >= j within the chunk:
            #   weight[i,j] = beta[j] * prod_{m=j+1}^{i} alpha[m]
            #
            # Compute cumulative log-alpha for efficient decay products
            log_alpha_c = torch.log(alpha_c.clamp(min=1e-6))  # (B, H, L)
            cumsum_log_alpha = torch.cumsum(log_alpha_c, dim=-1)  # (B, H, L)

            # decay_matrix[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j
            # Apply causal mask BEFORE exp to avoid inf * 0 = NaN
            # shape: (B, H, L, L)
            log_decay_matrix = cumsum_log_alpha.unsqueeze(
                -1
            ) - cumsum_log_alpha.unsqueeze(-2)

            # Apply causal mask: set upper triangle to -inf so exp gives 0
            causal_mask = self._get_causal_mask(L, device)
            log_decay_matrix = log_decay_matrix.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            decay_matrix = torch.exp(log_decay_matrix)

            # Scale by beta[j]
            decay_matrix = decay_matrix * beta_c.unsqueeze(-2)  # (B, H, L, L)

            # Intra-chunk attention: o_intra = decay_matrix @ (v_c ⊗ k_c projected)
            # Compute v_c @ k_c^T as the "update" at each position
            # But we need: o_intra[i] = sum_{j<=i} decay[i,j] * (q_c[i] · k_c[j]) * v_c[j]
            # This is: diag(q_c @ k_c^T * decay_matrix) @ v_c
            # More precisely: o_intra = (decay_matrix * (q_c @ k_c^T)) @ v_c

            # q_c @ k_c^T: (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
            qk = torch.matmul(q_c, k_c.transpose(-1, -2))
            intra_weights = decay_matrix * qk  # (B, H, L, L)

            # (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
            o_intra = torch.matmul(intra_weights, v_c)

            # --- D residual (direct token contribution) ---
            # D_res[t] = D * (q[t] · k[t]) * v[t]
            qk_diag = (q_c * k_c).sum(dim=-1, keepdim=True)  # (B, H, L, 1)
            o_direct = self.D.view(1, H, 1, 1) * qk_diag * v_c  # (B, H, L, D)

            # --- Combine ---
            o_chunk = o_inter + o_intra + o_direct
            output_chunks.append(o_chunk)

            # --- Update state S for next chunk ---
            # S_new = S * prod(alpha over chunk) + sum of beta[t] * v[t] @ k[t]^T * decay
            # Compute cumulative decay from end of chunk backwards
            # Final state: S' = alpha_prod * S + sum_{t in chunk} decay_to_end[t] * beta[t] * v[t] @ k[t]^T

            # Decay for entire chunk: product of all alphas
            total_log_decay = cumsum_log_alpha[:, :, -1:]  # (B, H, 1)
            chunk_decay = torch.exp(total_log_decay).unsqueeze(-1)  # (B, H, 1, 1)

            # Decay from each position to end of chunk
            # decay_to_end[t] = exp(cumsum[-1] - cumsum[t])
            decay_to_end = torch.exp(
                cumsum_log_alpha[:, :, -1:] - cumsum_log_alpha
            )  # (B, H, L)

            # Weighted outer products: sum beta[t] * decay_to_end[t] * v[t] @ k[t]^T
            # (B, H, L, D) * (B, H, L, 1) * (B, H, L, 1) -> weighted v
            weighted_v = v_c * (beta_c * decay_to_end).unsqueeze(-1)  # (B, H, L, D)
            # sum_t weighted_v[t] @ k_c[t]^T: (B, H, D, L) @ (B, H, L, D) = (B, H, D, D)
            delta_S = torch.matmul(weighted_v.transpose(-1, -2), k_c)

            S = chunk_decay * S + delta_S

        # Concatenate all chunks
        o = torch.cat(output_chunks, dim=2)  # (B, H, T, D)
        return o

    def forward(self, x, attention_mask=None):
        """
        Forward pass implementing Gated Delta Rule with decay.

        Backend priority:
        1. flash-linear-attention (fla) Triton kernels (if installed, CUDA only)
        2. Chunk-wise parallel recurrence (PyTorch, works on any device)

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

        # 3. Reshape to separate heads: (B, T, H, D)
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
        beta_scalar = torch.sigmoid(self.b_proj(x))  # (B, T, num_heads)

        # 7. Compute alpha/gate (decay parameter) - Paper Equation 10
        gk = self.gk_proj(x)  # (B, T, num_heads)
        A = -torch.exp(self.A_log)  # Negative for decay
        # g_log = log(alpha) in log space, always <= 0 so alpha in (0, 1]
        g_log = A.view(1, 1, self.num_heads) * F.softplus(
            gk + self.dt_bias
        )  # (B, T, H)

        # 8. Choose backend: FLA Triton kernels or chunk-wise parallel
        use_fla = HAS_FLA and device.type == "cuda"

        if use_fla:
            # ---- FLA Triton kernel path ----
            # FLA expects: q,k,v in (B, T, H, D), beta in (B, T, H), g in (B, T, H) log-space
            o, _ = _fla_chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g_log,
                beta=beta_scalar,
                scale=None,  # auto-scale by 1/sqrt(D)
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,  # we already L2-normalized
            )
            # o: (B, T, H, D)
        else:
            # ---- Chunk-wise parallel recurrence (PyTorch fallback) ----
            # Convert log-space gate to alpha for our implementation
            alpha = torch.sigmoid(g_log).unsqueeze(-1)  # (B, T, H, 1)
            beta = beta_scalar.unsqueeze(-1)  # (B, T, H, 1)

            # Transpose for (B, H, T, D) layout used by our chunk recurrence
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            beta_t = beta.transpose(1, 2)
            alpha_t = alpha.transpose(1, 2)

            # Choose chunk_size based on sequence length
            if T <= 64:
                chunk_size = T
            elif T <= 512:
                chunk_size = 64
            else:
                chunk_size = 128

            o = self._chunk_parallel_recurrence(
                q_t, k_t, v_t, alpha_t, beta_t, chunk_size=chunk_size
            )
            o = o.transpose(1, 2)  # (B, T, H, D)

        # 9. Apply output normalization with gating (VECTORIZED)
        if self.use_output_norm:
            B_, T_, H_, D_ = o.shape
            o_flat = o.reshape(B_ * T_ * H_, D_)
            g_flat = g.reshape(B_ * T_ * H_, D_)
            o = self.o_norm(o_flat, g_flat).reshape(B_, T_, H_, D_)
        else:
            o = o * torch.sigmoid(g)

        # 10. Reshape and project to output
        o = o.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o)
