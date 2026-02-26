"""
70B Model Architecture with Hybrid Gated DeltaNet + Gated Sparse Attention (GSA)

Configuration:
- 69.806B total parameters, 3.116B active parameters
- 131,072 vocabulary (2^17)
- 4096 hidden size, 20 layers (15 DeltaNet + 5 GSA)
- 270 real experts + 270 null experts = 540 slots, top-k=10 dynamic
- Multi-Token Prediction (MTP) with 2 predictions
- Multi-Head Composition (mHC) with 4 streams
- Reversible Midpoint Integration for memory efficiency
- Target: 256k context length

Architecture based on:
- Gated DeltaNet: arXiv:2412.06464 (Dec 2024)
- Gated Sparse Attention: arXiv:2601.15305v1 (Jan 2026)
- Multi-Token Prediction: DeepSeek-V3 style
- Null Experts: Data sparsity ρ=0.5
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: Importing for backwards compatibility - we define KroneckerEmbeddings inline
# from kronecker_se_decoder import PFConfig, PFCodec


# ============================================================================
# Kronecker Product Embeddings (formerly PFCodec)
# ============================================================================


@dataclass
class KroneckerConfig:
    """
    Configuration for Byte-Level Kronecker Product Embeddings.

    Encodes tokens as Kronecker products of byte and position embeddings:
    PF(token) = (1/√L) × vec(Σ_{i=1..L} e_byte[b_i] ⊗ e_pos[i])

    Byte-Level Encoding:
    - Input: Unicode string (Python str)
    - Process: str → UTF-8 bytes → each byte (0-255) is a token
    - Universal: 100% coverage of all UTF-8 text (Chinese, Arabic, emoji, etc.)
    - Lossless: Perfect reconstruction via bytes.decode("utf-8")

    Parameters:
    - CHAR_DIM: 256 (bytes 0-255, NOT characters)
    - POS_DIM: 32 (max 32 bytes per token)
    - D: 32 × 256 = 8192 dimensions
    """

    CHAR_DIM: int = 256  # Byte vocabulary (0-255)
    POS_DIM: int = 32  # Max token length in bytes
    D: int = 8192  # CHAR_DIM × POS_DIM = 256 × 32
    length_normalize: bool = True
    truncate_long_words: bool = True

    def __post_init__(self):
        assert self.CHAR_DIM == 256, "CHAR_DIM must be 256 for byte-level encoding"
        assert (
            self.D == self.CHAR_DIM * self.POS_DIM
        ), f"D ({self.D}) must equal CHAR_DIM × POS_DIM ({self.CHAR_DIM} × {self.POS_DIM})"


class KroneckerEmbeddings:
    """
    Byte-Level Kronecker Product Embeddings.

    Encodes tokens using Kronecker product of UTF-8 byte and position embeddings:
    PF(token) = (1/√L) × vec(Σ_{i=1..L} e_byte[b_i] ⊗ e_pos[i])

    Byte-Level Design:
    - Input: Unicode string (Python str)
    - Encoding: str → UTF-8 bytes → Kronecker embeddings
    - Each byte (0-255) is treated as a valid symbol
    - Decoding: bytes → UTF-8 decode → str
    - 100% universal: All UTF-8 text supported (no exclusions)

    Properties:
    - Invertible: Can decode back to original token
    - Length-normalized: 1/√L scaling for length invariance
    - Structured: Separable byte and position information
    - Universal: Perfect coverage of Chinese, Arabic, emoji, etc.

    Configuration:
    - POS_DIM=32: Handles tokens up to 32 UTF-8 bytes
    - CHAR_DIM=256: All bytes 0-255
    - D=8192: Total embedding dimension (32 × 256)

    Note: Cannot tie with lm_head (8192 != hidden_size=4096)
    """

    def __init__(self, cfg: KroneckerConfig):
        self.cfg = cfg
        self.CHAR_DIM = cfg.CHAR_DIM
        self.POS_DIM = cfg.POS_DIM
        self.D = cfg.D
        # Identity bases for exact inversion
        self.E_char = np.eye(self.CHAR_DIM, dtype=np.float32)
        self.P_pos = np.eye(self.POS_DIM, dtype=np.float32)

    def _utf8_safe_truncate(self, byte_seq: bytes, max_bytes: int) -> bytes:
        """
        Truncate byte sequence without splitting UTF-8 multibyte characters.

        Args:
            byte_seq: UTF-8 encoded bytes
            max_bytes: Maximum number of bytes

        Returns:
            Truncated bytes that form valid UTF-8
        """
        if len(byte_seq) <= max_bytes:
            return byte_seq

        # Try decoding at truncation point and move back if invalid
        for end in range(max_bytes, max(max_bytes - 4, 0) - 1, -1):
            try:
                byte_seq[:end].decode("utf-8")
                return byte_seq[:end]
            except UnicodeDecodeError:
                continue

        # Fallback: return empty if can't find valid truncation
        return b""

    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a single token to Kronecker embedding using byte-level encoding.

        Process:
        1. Convert str → UTF-8 bytes
        2. Truncate if needed (UTF-8 safe)
        3. Build byte-position matrix via Kronecker product
        4. Apply length normalization
        5. Flatten to D-dimensional vector

        Args:
            word: Input token (Unicode string)

        Returns:
            Embedding vector of shape (D,) = (256 × 32,) = (8192,)

        Example:
            >>> encoder.encode_word("hello世界")
            # Encodes all 11 UTF-8 bytes: h,e,l,l,o,世(3 bytes),界(3 bytes)
        """
        if word is None or word == "":
            return np.zeros((self.D,), dtype=np.float32)

        # Convert to UTF-8 bytes
        byte_seq = word.encode("utf-8")

        # Truncate if needed (UTF-8 safe)
        if len(byte_seq) > self.POS_DIM:
            if self.cfg.truncate_long_words:
                byte_seq = self._utf8_safe_truncate(byte_seq, self.POS_DIM)
            else:
                raise ValueError(
                    f"Token byte length {len(byte_seq)} exceeds POS_DIM={self.POS_DIM}"
                )

        L = len(byte_seq)
        if L == 0:
            return np.zeros((self.D,), dtype=np.float32)

        # Build byte-position matrix
        M = np.zeros((self.CHAR_DIM, self.POS_DIM), dtype=np.float32)
        for i, byte_val in enumerate(byte_seq):
            # byte_val is already 0-255 (int)
            M[byte_val, i] = 1.0

        # Length normalization
        if self.cfg.length_normalize:
            M *= 1.0 / math.sqrt(L)

        return M.reshape(self.D)

    def decode_word(self, pf_vec: np.ndarray, threshold: float = 1e-6) -> str:
        """
        Decode Kronecker embedding back to token using byte-level decoding.

        Process:
        1. Reshape D-vector to 256×32 matrix
        2. Find active positions (non-zero columns)
        3. Extract byte value at each position (argmax)
        4. Collect bytes → decode UTF-8 → str

        Args:
            pf_vec: Embedding vector of shape (D,)
            threshold: Minimum magnitude to consider a position active

        Returns:
            Decoded token string

        Example:
            >>> embedding = encoder.encode_word("hello世界")
            >>> decoder.decode_word(embedding)
            "hello世界"  # Perfect reconstruction
        """
        if pf_vec.shape != (self.D,):
            raise ValueError(f"pf_vec must have shape ({self.D},), got {pf_vec.shape}")

        # Reshape to byte-position matrix
        M = pf_vec.reshape(self.CHAR_DIM, self.POS_DIM)

        # Find active positions (non-zero columns)
        col_norms = np.linalg.norm(M, axis=0)
        positions = [i for i, cn in enumerate(col_norms) if cn > threshold]

        # Extract byte at each position
        bytes_list = []
        for i in positions:
            byte_val = int(np.argmax(M[:, i]))  # 0-255
            bytes_list.append(byte_val)

        # Convert bytes to string
        byte_seq = bytes(bytes_list)
        try:
            return byte_seq.decode("utf-8")
        except UnicodeDecodeError:
            # Should never happen with properly encoded data
            # But handle gracefully just in case
            return byte_seq.decode("utf-8", errors="replace")

    def encode_batch(self, words: List[str]) -> np.ndarray:
        """Encode a batch of words."""
        return np.stack([self.encode_word(w) for w in words], axis=0)

    def decode_batch(self, pf_mat: np.ndarray, threshold: float = 1e-6) -> List[str]:
        """Decode a batch of embeddings."""
        return [self.decode_word(pf_mat[i], threshold) for i in range(pf_mat.shape[0])]


# Aliases for backwards compatibility
PFCodec = KroneckerEmbeddings
PFConfig = KroneckerConfig


# ============================================================================
# CONFIGURATION
# ============================================================================


class ModelConfig:
    """1B Dense Model Configuration"""

    # Architecture
    vocab_size = 131072  # 2^17
    hidden_size = 4096
    num_layers = 8

    # Attention Mix (75% DeltaNet / 25% GSA)
    num_deltanet_layers = 6
    num_gsa_layers = 2

    # DeltaNet Configuration
    delta_v_heads = 32  # hidden_size / delta_head_dim = 4096 / 128
    delta_qk_heads = 16  # delta_v_heads / 2
    delta_head_dim = 128
    delta_gate_dim = 384  # 9.4% of hidden_size

    # GSA Configuration
    gsa_num_heads = 16  # hidden_size / attn_head_dim = 4096 / 256
    gsa_head_dim = 256
    gsa_k_base = 512  # Adaptive sparsity budget for 256k context
    gsa_k_min = 32
    gsa_k_max = 1024  # Increased for 256k context
    gsa_indexer_heads = 4

    # MoE Configuration (DENSE MODEL - No MoE)
    num_real_experts = 0
    num_null_experts = 0
    total_expert_slots = 0
    top_k = 0  # Not used in dense model
    expert_intermediate_size = 1024  # Not used in dense model
    shared_expert_intermediate_size = 2048  # Acts as dense FFN
    data_sparsity = 0.0  # No data sparsity (dense)

    # MTP Configuration
    enable_mtp = True
    mtp_num_predictions = 2

    # mHC Configuration
    n_streams = 4
    sinkhorn_iters = 20

    # Context and RoPE (YARN Scaling)
    max_seq_len = 262144  # 256k context
    rope_base = 10000
    rope_original_max_position = 8192  # Original training context
    rope_scaling_factor = 32.0  # 256k / 8k = 32x extension

    # Training
    dropout = 0.0  # Required for reversible integration


# ============================================================================
# Embedding Layer (Kronecker Product)
# ============================================================================


class PureHybridEmbeddingTorch(nn.Module):
    """
    Pure Kronecker Product Embedding.

    Uses KroneckerEmbeddings (formerly PFCodec) to encode vocabulary words
    as Kronecker products of character and position embeddings.

    Configuration:
    - POS_DIM=32: Handles tokens up to 32 characters
    - CHAR_DIM=256: Full ASCII + extended character set
    - D=8192: Total embedding dimension (32 × 256)

    Process:
    1. Precomputes PF(word) for entire vocabulary
    2. At runtime: fetches PF vector for each token
    3. Normalizes per-token (zero mean, unit std)
    4. Projects to hidden_size via pf_to_model layer

    Note: Embedding tying NOT possible (D=8192 != hidden_size=4096)
    """

    def __init__(self, vocab_words: List[str], pf_codec: KroneckerEmbeddings):
        super().__init__()
        PF_table = pf_codec.encode_batch(vocab_words)  # (vocab_size, D)
        PF_np = PF_table.astype(np.float32)
        pf_tensor = torch.from_numpy(PF_np).to(torch.bfloat16)
        self.register_buffer("PF_table", pf_tensor, persistent=True)

    def forward(self, token_ids):
        """
        Forward pass: fetch and normalize Kronecker embeddings.

        Args:
            token_ids: Token indices (B, T)

        Returns:
            Normalized embeddings (B, T, D=8192)
        """
        PF = self.PF_table[token_ids].to(dtype=torch.float32)
        # Normalize per token (zero mean, unit std)
        PF_centered = PF - PF.mean(dim=-1, keepdim=True)
        PF_std = PF_centered.std(dim=-1, keepdim=True) + 1e-6
        PFn = PF_centered / PF_std
        return PFn

    def module(self):
        return self


# ============================================================================
# Core Components
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """
    YARN (Yet Another RoPE extensioN) Rotary Positional Embedding.

    Extends RoPE to 256k context using:
    - NTK-aware interpolation for scaling base frequency
    - Temperature-based frequency band interpolation
    - Attention sink preservation for initial tokens

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: int = 10000,
        original_max_position_embeddings: int = 8192,
        scaling_factor: float = 32.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

        # YARN: NTK-aware interpolation
        # Scale the base frequency to accommodate longer context
        if max_position_embeddings > original_max_position_embeddings:
            # NTK-by-parts: scale base exponentially based on extension ratio
            ext_ratio = max_position_embeddings / original_max_position_embeddings
            # Use a gentler scaling exponent for YARN (typically around 1.0)
            scaled_base = base * (ext_ratio ** (dim / (dim - 2)))
            print(
                f"   🧶 YARN RoPE: Scaling base {base} -> {scaled_base:.0f} for {max_position_embeddings:,} context"
            )
        else:
            scaled_base = base

        # Compute inverse frequencies with scaled base
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # YARN: Frequency band interpolation parameters
        # Interpolate low frequencies, extrapolate high frequencies
        # beta_fast: controls high-freq behavior (extrapolation)
        # beta_slow: controls low-freq behavior (interpolation)
        self.beta_fast = 32  # High frequencies (extrapolate)
        self.beta_slow = 1  # Low frequencies (interpolate)

        # Compute interpolation weights (mscale) for each frequency
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Determine which frequencies to interpolate vs extrapolate
        # High frequencies (small wavelengths) get less interpolation
        wavelen = 2 * math.pi / freq_extra
        # Ramp function: 0 at beta_fast, 1 at beta_slow
        ramp = torch.clamp(
            (wavelen - self.beta_fast) / (self.beta_slow - self.beta_fast), 0, 1
        )
        self.register_buffer("mscale", ramp)  # Interpolation weight per frequency

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # YARN: Apply frequency-dependent interpolation
        # t_scaled = t / (1 + (scaling_factor - 1) * ramp)
        # This interpolates low frequencies more, extrapolates high frequencies
        scale_factor_per_freq = 1.0 + (self.scaling_factor - 1.0) * self.mscale
        t_scaled = t.unsqueeze(-1) / scale_factor_per_freq.unsqueeze(0)

        freqs = t_scaled * self.inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat(
            (
                x1 * cos[..., ::2] - x2 * sin[..., ::2],
                x1 * sin[..., ::2] + x2 * cos[..., ::2],
            ),
            dim=-1,
        )


# ============================================================================
# Helper Modules for Gated DeltaNet
# ============================================================================


class ShortConvolution(nn.Module):
    """
    Short convolution layer with causal padding.
    Used in Gated DeltaNet for local context integration.
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
# Gated DeltaNet (75% of layers) - O(N) Linear Attention
# ============================================================================


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet - arXiv:2412.06464 (Dec 2024)

    O(N) linear attention with gating and alpha decay for long-context efficiency.
    Essential for 256k context where quadratic attention is prohibitive.

    Key components from paper (Equation 10):
    St = St-1(αt(I - βtktkt^T)) + βtvtkt^T

    - Alpha (αt): Per-head decay parameter controlling state forgetting
    - Beta (βt): Writing strength controlling update magnitude
    - L2 normalization: For Q/K stability (NOT softmax)
    - Short convolutions: Local context integration (kernel_size=4)
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
        self.A_log = nn.Parameter(torch.log(A_init))  # log(A) for stability

        # D parameter for residual connection (per-head)
        self.D = nn.Parameter(torch.ones(num_heads))

        # dt_bias for Mamba-style gating (per-head)
        # Special initialization: log-uniform for stable gating
        dt_init_std = 0.01
        dt_bias = torch.rand(num_heads) * 2 * dt_init_std - dt_init_std
        self.dt_bias = nn.Parameter(dt_bias)

        # Rotary embeddings for Q/K with YARN scaling
        self.rotary_emb = RotaryEmbedding(
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
        # Linear projections: std=0.02 (DeepScreen initialization)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Gate projections
        for m in [self.b_proj, self.gk_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None):
        """
        Forward pass implementing Gated Delta Rule with decay.

        Args:
            x: Input tensor (B, T, hidden_size)
            attention_mask: Optional attention mask (not used for linear attention)

        Returns:
            Output tensor (B, T, hidden_size)
        """
        B, T, C = x.shape
        device = x.device

        # 1. Project to Q, K, V, G
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
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
        # alpha = -exp(A_log) * softplus(gk + dt_bias)
        # This ensures alpha is in (0, 1) for stability
        gk = self.gk_proj(x)  # (B, T, num_heads)
        A = -torch.exp(self.A_log)  # Negative for decay
        alpha = A.view(1, 1, self.num_heads) * F.softplus(gk + self.dt_bias).unsqueeze(
            -1
        )
        # Clamp alpha to reasonable range for stability
        alpha = torch.sigmoid(alpha)  # (B, T, num_heads, 1)

        # 8. Transpose for computation
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)  # (B, num_heads, T, 1)
        alpha = alpha.transpose(1, 2)

        # 9. Gated Delta Rule with decay (Paper Equation 10)
        # St = St-1 * (alpha * (I - beta * k * k^T)) + beta * v * k^T
        # Simplified for initial implementation: St = alpha * St-1 + beta * v @ k^T
        # Using cumulative computation for O(N) complexity

        # Initialize state
        S = torch.zeros(
            B,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=x.dtype,
        )
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]  # (B, num_heads, head_dim)
            k_t = k[:, :, t, :]  # (B, num_heads, head_dim)
            v_t = v[:, :, t, :]  # (B, num_heads, head_dim)
            beta_t = beta[:, :, t, 0]  # (B, num_heads) - scalar per head
            alpha_t = alpha[:, :, t, 0]  # (B, num_heads) - scalar per head

            # Query current state
            o_t = torch.einsum("bhd,bhde->bhe", q_t, S)  # (B, num_heads, head_dim)

            # Add D residual (direct token contribution)
            o_t = (
                o_t
                + self.D.view(1, self.num_heads, 1)
                * (q_t * k_t).sum(dim=-1, keepdim=True)
                * v_t
            )

            outputs.append(o_t)

            # Update state with gated delta rule
            # Compute outer product: v @ k^T
            v_outer = torch.einsum(
                "bhd,bhe->bhde", v_t, k_t
            )  # (B, num_heads, head_dim, head_dim)

            # Apply decay and update: S = alpha * S + beta * v @ k^T
            # Reshape alpha_t and beta_t for broadcasting: (B, num_heads) -> (B, num_heads, 1, 1)
            alpha_t = alpha_t.view(B, self.num_heads, 1, 1)
            beta_t = beta_t.view(B, self.num_heads, 1, 1)

            S = alpha_t * S + beta_t * v_outer

        # Stack outputs
        o = torch.stack(outputs, dim=2)  # (B, num_heads, T, head_dim)

        # 10. Apply output normalization with gating
        o = o.transpose(1, 2)  # (B, T, num_heads, head_dim)
        g = g  # (B, T, num_heads, head_dim) - already in correct shape

        if self.use_output_norm:
            # Apply RMSNorm and gating per head
            o_norm = []
            for h in range(self.num_heads):
                o_h = o[:, :, h, :]  # (B, T, head_dim)
                g_h = g[:, :, h, :]  # (B, T, head_dim)
                o_norm.append(self.o_norm(o_h, g_h))
            o = torch.stack(o_norm, dim=2)  # (B, T, num_heads, head_dim)
        else:
            o = o * torch.sigmoid(g)

        # 11. Reshape and project to output
        o = o.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o)


# ============================================================================
# Gated Sparse Attention (25% of layers) - From test model
# ============================================================================


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) - arXiv:2601.15305v1

    Implements adaptive sparse attention with gating for quality.
    Used for 25% of layers to complement DeltaNet's efficiency.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_seq_len=262144,
        rope_base=10000,
        k_base=512,
        k_min=32,
        k_max=1024,
        indexer_heads=4,
        rope_original_max=8192,
        rope_scaling_factor=32.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len

        # Adaptive Sparsity Hyperparams
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.indexer_heads = indexer_heads

        # Lightning Indexer
        self.d_idx = 32
        self.W_Iq = nn.Linear(hidden_size, indexer_heads * self.d_idx, bias=False)
        self.W_Ik = nn.Linear(hidden_size, self.d_idx, bias=False)
        self.W_Iw = nn.Linear(hidden_size, indexer_heads, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(indexer_heads))

        self.register_buffer("variance_ema", torch.tensor(1.0))
        self.variance_alpha = 0.01

        # Attention Projections
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dual Gating
        self.W_gv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_go = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary embeddings with YARN scaling
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
            original_max_position_embeddings=rope_original_max,
            scaling_factor=rope_scaling_factor,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [
            self.W_Iq,
            self.W_Ik,
            self.W_Iw,
            self.W_q,
            self.W_k,
            self.W_v,
            self.o_proj,
            self.W_gv,
            self.W_go,
        ]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device

        # Lightning Indexer
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)
        k_I = self.W_Ik(x)
        w = torch.sigmoid(self.W_Iw(x))

        q_I_p = q_I.permute(0, 2, 1, 3)
        k_I_p = k_I.permute(0, 2, 1).unsqueeze(1)

        match_logits = torch.matmul(q_I_p, k_I_p)
        match_logits = match_logits + self.gate_bias.view(1, self.indexer_heads, 1, 1)
        match_gate = torch.sigmoid(match_logits)

        w_exp = w.permute(0, 2, 1).unsqueeze(-1)
        importance_score = (w_exp * match_gate).sum(dim=1)

        # Causal masking (memory-efficient: no explicit T×T matrix)
        if T > 1:
            # Use broadcasting instead of creating T×T mask (saves 64GB for 256k context!)
            # importance_score is [B, T_query, T_key]
            # For causal: only attend to positions <= current position
            positions = torch.arange(T, device=device)
            # Shape: [1, T, 1] compared to [1, 1, T] -> broadcasts to [1, T, T]
            causal_mask_broadcast = positions.view(1, -1, 1) >= positions.view(1, 1, -1)
            importance_score_masked = importance_score.masked_fill(
                ~causal_mask_broadcast, 0.0
            )
            causal_mask = causal_mask_broadcast  # Store for later use
        else:
            importance_score_masked = importance_score
            causal_mask = None

        # Adaptive Sparsity
        var_t = importance_score_masked.var(dim=-1, unbiased=False)

        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        is_reversible_reconstruct = (
            self.training
            and torch.is_grad_enabled()
            and getattr(self, "_saved_selection", None) is not None
        )

        if is_reversible_forward:
            var_t_mean = var_t.mean().detach()
            self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)

        if is_reversible_reconstruct:
            k_t, top_indices = self._saved_selection
            self._saved_selection = None
            avg_V = self.variance_ema.clamp(min=1e-6)
        else:
            avg_V = self.variance_ema.clamp(min=1e-6)
            k_t_float = self.k_base * var_t / avg_V
            k_t = k_t_float.floor().clamp(min=self.k_min, max=self.k_max).long()

            if T > 1:
                importance_for_selection = importance_score.masked_fill(
                    ~causal_mask, -float("inf")
                )
            else:
                importance_for_selection = importance_score

            # Attention sinks
            sink_size = 4
            if T > sink_size:
                sink_mask = torch.zeros_like(importance_for_selection, dtype=torch.bool)
                sink_mask[:, :, :sink_size] = True
                importance_for_selection = importance_for_selection.masked_fill(
                    sink_mask, float("inf")
                )

            k_limit = min(T, max(k_t.max().item(), sink_size))
            _, top_indices = importance_for_selection.topk(k_limit, dim=-1)

            if is_reversible_forward:
                self._saved_selection = (k_t, top_indices)

        # Construct boolean mask
        k_limit = top_indices.size(-1)
        range_k = torch.arange(k_limit, device=device).unsqueeze(0).unsqueeze(0)
        keep_in_topk = range_k < k_t.unsqueeze(-1)

        selection_mask = torch.zeros_like(importance_score, dtype=torch.bool)
        selection_mask.scatter_(dim=-1, index=top_indices, src=keep_in_topk)

        if T > 1:
            selection_mask = selection_mask & causal_mask

        # Dual Gating & Attention
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        g_v = torch.sigmoid(self.W_gv(x))
        v = v * g_v

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Rotary
        if T > self.rotary_emb.cos_cached.size(0):
            self.rotary_emb._set_cos_sin_cache(T)
        cos = self.rotary_emb.cos_cached[:T].unsqueeze(0).unsqueeze(2)
        sin = self.rotary_emb.sin_cached[:T].unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Masked attention
        min_val = torch.finfo(q.dtype).min
        bias_mask = torch.zeros_like(selection_mask, dtype=q.dtype)
        bias_mask = bias_mask.masked_fill(~selection_mask, min_val)

        if attention_mask is not None:
            bias_mask = bias_mask + attention_mask

        o_sparse = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias_mask.unsqueeze(1), dropout_p=0.0, is_causal=False
        )

        o_sparse = o_sparse.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # Output gate
        g_o = torch.sigmoid(self.W_go(x))

        return self.o_proj(o_sparse * g_o)


# ============================================================================
# MoE with Null Experts (from test model)
# ============================================================================


class MoEGate(nn.Module):
    """Router gate for MoE with null experts."""

    def __init__(
        self, d_model: int, num_experts: int, top_k: int, data_sparsity: float = 0.5
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity

        self.num_null_copies = int(num_experts * (1 - data_sparsity) / data_sparsity)
        self.total_slots = num_experts + self.num_null_copies

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        self.gate.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        real_logits = self.gate(x) + self.logit_bias
        null_logits = (
            self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)
        )
        logits = torch.cat([real_logits, null_logits], dim=-1)

        probs = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        is_null = topk_idx >= self.num_experts
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # Auxiliary losses
        P = probs.mean(dim=(0, 1))
        idx_flat = topk_idx.view(-1)
        counts = torch.bincount(idx_flat, minlength=self.total_slots).float()
        f = counts / (B * T)
        L_bal = self.total_slots * torch.sum(f * P)

        lse = torch.logsumexp(logits, dim=-1)
        L_z = (lse**2).mean()

        aux_loss = 2e-2 * L_bal + 1e-3 * L_z

        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """MoE FFN with null experts (batched tensor implementation)."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int = 270,
        top_k: int = 10,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        # Expert weights (batched)
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared Expert
        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None

    def _init_shared_weights(self):
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype

        # Shared expert
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # Routed experts
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()

        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        real_mask = ~flat_is_null
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        real_token_indices = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights = flat_weight[real_mask]

        sort_idx = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]

        expert_counts = torch.bincount(real_expert_indices, minlength=E)
        offsets = expert_counts.cumsum(0)

        num_real_assignments = sorted_token_indices.size(0)
        sorted_out = torch.empty(num_real_assignments, D, device=device, dtype=dtype)

        start = 0
        for e in range(E):
            end = offsets[e].item()
            if end > start:
                chunk_x = sorted_x[start:end]
                h = F.silu(chunk_x @ self.W_gate[e]) * (chunk_x @ self.W_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.W_down[e]
            start = end

        weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out
        )

        y = shared_out + routed_out.view(B, T, D)
        return y, aux_loss


class LightningMLP(nn.Module):
    """MLP wrapper using MoEFFN."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts,
        num_shared_experts,
        top_k,
        data_sparsity=0.5,
    ):
        super().__init__()
        self.moe = MoEFFN(
            d_model=hidden_size,
            d_hidden=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=0.0,
            data_sparsity=data_sparsity,
        )

    def forward(self, x):
        return self.moe(x)


# ============================================================================
# mHC (Multi-Head Composition) - From test model
# ============================================================================


@torch.jit.script
def sinkhorn_knopp(
    logits: torch.Tensor, iters: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """Doubly-stochastic matrix via Sinkhorn-Knopp."""
    M = torch.exp(logits).clamp_min(eps)
    for _ in range(iters):
        M = M / (M.sum(dim=-1, keepdim=True).clamp_min(eps))
        M = M / (M.sum(dim=-2, keepdim=True).clamp_min(eps))
    return M


class MHCCoeffs(nn.Module):
    """Produces routing coefficients for mHC."""

    def __init__(self, d_model: int, n_streams: int = 4, iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.iters = iters

        d_in = self.n * d_model

        self.phi_pre = nn.Linear(d_in, self.n, bias=False)
        self.phi_post = nn.Linear(d_in, self.n, bias=False)
        self.phi_res = nn.Linear(d_in, self.n * self.n, bias=False)

        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        self.alpha_pre = nn.Parameter(torch.tensor(0.1))
        self.alpha_post = nn.Parameter(torch.tensor(0.1))
        self.alpha_res = nn.Parameter(torch.tensor(0.1))

        self.rms = RMSNorm(d_in)

        for m in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x_stream: torch.Tensor):
        B, T, n, D = x_stream.shape
        x_flat = x_stream.reshape(B, T, n * D)
        x_flat = self.rms(x_flat)

        pre_logits = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        post_logits = self.alpha_post * self.phi_post(x_flat) + self.b_post

        res_logits = self.alpha_res * self.phi_res(x_flat)
        res_logits = res_logits.view(B, T, n, n) + self.b_res

        H_pre = torch.sigmoid(pre_logits)
        H_post = 2.0 * torch.sigmoid(post_logits)
        H_res = sinkhorn_knopp(res_logits, iters=self.iters)

        return H_pre, H_post, H_res


class MHCSublayer(nn.Module):
    """Wrap sublayer with mHC residual routing."""

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        sublayer: nn.Module,
        norm: nn.Module,
        iters: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.sublayer = sublayer
        self.norm = norm
        self.coeffs = MHCCoeffs(d_model=d_model, n_streams=n_streams, iters=iters)

    def forward(self, x_stream: torch.Tensor, attention_mask=None):
        H_pre, H_post, H_res = self.coeffs(x_stream)

        x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)
        x_in = self.norm(x_in)

        aux_loss = None
        if attention_mask is None:
            out = self.sublayer(x_in)
        else:
            out = self.sublayer(x_in, attention_mask)

        if isinstance(out, tuple):
            y, aux_loss = out
        else:
            y = out

        y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)
        x_res = torch.einsum("btij,btjd->btid", H_res, x_stream)

        return x_res + y_stream, aux_loss


# ============================================================================
# Decoder Layer (Hybrid DeltaNet + GSA)
# ============================================================================


class LightningDecoderLayer(nn.Module):
    """
    Decoder layer that can be either DeltaNet or GSA.
    Type is determined at initialization.
    """

    def __init__(self, config: ModelConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type  # "deltanet" or "gsa"
        self.n_streams = config.n_streams

        # Select attention mechanism
        if layer_type == "deltanet":
            attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.delta_v_heads,
                head_dim=config.delta_head_dim,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                rope_original_max=config.rope_original_max_position,
                rope_scaling_factor=config.rope_scaling_factor,
                conv_size=4,
                use_output_norm=True,
            )
        elif layer_type == "gsa":
            attn = GatedSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.gsa_num_heads,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                k_base=config.gsa_k_base,
                k_min=config.gsa_k_min,
                k_max=config.gsa_k_max,
                indexer_heads=config.gsa_indexer_heads,
                rope_original_max=config.rope_original_max_position,
                rope_scaling_factor=config.rope_scaling_factor,
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        mlp = LightningMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            num_experts=config.num_real_experts,
            num_shared_experts=1,
            top_k=config.top_k,
            data_sparsity=config.data_sparsity,
        )

        # mHC Wrappers
        self.attn_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=attn,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )

        self.mlp_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=mlp,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )

    def force(self, x):
        """Compute residual delta for reversible integration."""
        h, aux1 = self.attn_block(x, attention_mask=None)
        out, aux2 = self.mlp_block(h, attention_mask=None)

        delta = out - x

        aux = None
        if aux1 is not None:
            aux = aux1
        if aux2 is not None:
            if aux is None:
                aux = aux2
            else:
                aux = aux + aux2

        if aux is None:
            aux = x.new_zeros((), dtype=torch.float32)

        return delta, aux

    def forward(self, x_stream, attention_mask=None):
        x_stream, aux1 = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, aux2 = self.mlp_block(x_stream, attention_mask=None)

        total_aux = None
        if aux1 is not None or aux2 is not None:
            total_aux = (aux1 if aux1 is not None else 0) + (
                aux2 if aux2 is not None else 0
            )

        return x_stream, total_aux


# ============================================================================
# Multi-Token Prediction Block
# ============================================================================


class MTPTransformerBlock(nn.Module):
    """MTP block for predicting t+2 from [h_t; emb_{t+1}]."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_streams = config.n_streams
        self.hidden_size = config.hidden_size

        # Fusion layer
        self.fusion_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # Core sublayers (using DeltaNet for efficiency)
        self.attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.delta_v_heads,
            head_dim=config.delta_head_dim,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            rope_original_max=config.rope_original_max_position,
            rope_scaling_factor=config.rope_scaling_factor,
            conv_size=4,
            use_output_norm=True,
        )

        self.mlp = LightningMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            num_experts=config.num_real_experts,
            num_shared_experts=1,
            top_k=config.top_k,
            data_sparsity=config.data_sparsity,
        )

        # mHC Wrappers
        self.attn_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=self.attn,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )

        self.mlp_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=self.mlp,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffs)):
            return

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_t, next_emb, attention_mask=None):
        batch_size, seq_len, _ = h_t.shape

        # Fuse
        x = torch.cat([h_t, next_emb], dim=-1)
        x = self.fusion_proj(x)

        # Expand to streams
        x_stream = torch.zeros(
            batch_size,
            seq_len,
            self.n_streams,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        x_stream[:, :, 0, :] = x

        # mHC blocks (ignore aux_loss)
        x_stream, _ = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, _ = self.mlp_block(x_stream, attention_mask=None)

        # Collapse
        x_out = x_stream.mean(dim=2)

        return x_out


# ============================================================================
# Complete 70B Model
# ============================================================================


class Model70B(nn.Module):
    """
    70B Model with Hybrid Gated DeltaNet + Gated Sparse Attention.

    Configuration:
    - 69.806B total params, 3.116B active params
    - 75% DeltaNet (15 layers) + 25% GSA (5 layers)
    - 270 real + 270 null experts, top-k=10 dynamic
    - 256k context length target
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type="kronecker",
        bpe_vocab=None,
        pf_codec=None,
    ):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_type = embedding_type.lower()
        self.n_streams = config.n_streams

        # Embeddings
        if self.embedding_type == "kronecker":
            if bpe_vocab is None or pf_codec is None:
                raise ValueError(
                    "bpe_vocab and pf_codec required for Kronecker embeddings"
                )

            self.kronecker_embeddings = PureHybridEmbeddingTorch(
                bpe_vocab, pf_codec
            ).module()
            D_pf = pf_codec.D
            self.pf_to_model = nn.Linear(D_pf, config.hidden_size, bias=False)
            self.embed_norm = RMSNorm(config.hidden_size)
            self.token_embed = None
            self.use_kronecker = True
            self._D_pf = D_pf
        else:
            self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.kronecker_embeddings = None
            self.pf_to_model = None
            self.embed_norm = None
            self.use_kronecker = False

        # Build hybrid layer stack: 75% DeltaNet + 25% GSA
        # Strategy: Alternate for balanced distribution
        layers = []
        layer_types = []

        for i in range(config.num_layers):
            # First 75% are DeltaNet, last 25% are GSA
            if i < config.num_deltanet_layers:
                layer_type = "deltanet"
            else:
                layer_type = "gsa"

            layers.append(LightningDecoderLayer(config, layer_type))
            layer_types.append(layer_type)

        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types

        # Reversible Midpoint Integration
        from reversible_ops_midpoint import ReversibleMidpointStack

        self.stack = ReversibleMidpointStack(
            self.layers,
            step_size=0.25,
            a=0.5,
            noise_eps=0.0,
            bootstrap="euler",
        )

        self.norm = RMSNorm(config.hidden_size)

        # MTP Block
        if config.enable_mtp:
            self.mtp_block = MTPTransformerBlock(config)
        else:
            self.mtp_block = None

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)

        # Re-initialize Kronecker projection for scale matching
        if self.use_kronecker and self.pf_to_model is not None:
            pf_to_model_std = 0.02 / math.sqrt(self._D_pf)
            self.pf_to_model.weight.data.normal_(mean=0.0, std=pf_to_model_std)
            print(
                f"   🔧 pf_to_model (8192→{config.hidden_size}) initialized with std={pf_to_model_std:.6f}"
            )

        # Print configuration
        total_params = sum(p.numel() for p in self.parameters())

        # Calculate embedding parameters
        if self.use_kronecker:
            # Kronecker embeddings: vocab_size × D (buffer, not parameters)
            # pf_to_model: D × hidden_size (trainable)
            embedding_buffer = self.vocab_size * self._D_pf / 1e6  # In millions
            embedding_params = self._D_pf * config.hidden_size / 1e6  # In millions
        else:
            embedding_params = self.vocab_size * config.hidden_size / 1e6
            embedding_buffer = 0

        print("\n🤖 MODEL-1B (DENSE) INITIALIZED:")
        print(f"   Vocabulary: {self.vocab_size:,}")
        print(f"   Hidden Size: {config.hidden_size}")
        if self.use_kronecker:
            print("\n   📐 Kronecker Embeddings:")
            print("      POS_DIM=32 x CHAR_DIM=256 = D=8192")
            print(
                f"      Buffer size: {embedding_buffer:.1f}M (vocab × 8192, non-trainable)"
            )
            print(
                f"      pf_to_model: {embedding_params:.1f}M params (8192 × {config.hidden_size})"
            )
            print(
                f"      ⚠️  Embedding tying NOT possible (8192 ≠ {config.hidden_size})"
            )
        print(f"\n   Total Layers: {config.num_layers}")
        print(
            f"   - DeltaNet: {config.num_deltanet_layers} layers ({config.num_deltanet_layers/config.num_layers*100:.0f}%) - O(N) linear attention"
        )
        print(
            f"   - GSA: {config.num_gsa_layers} layers ({config.num_gsa_layers/config.num_layers*100:.0f}%) - Adaptive sparse"
        )
        print(f"\n   Context Target: {config.max_seq_len:,} tokens (YARN RoPE scaling)")
        print(
            f"   Experts: {config.num_real_experts} real + {config.num_null_experts} null = {config.total_expert_slots} slots"
        )
        print(
            f"   Top-k: {config.top_k} (dynamic, avg 5 with ρ={config.data_sparsity})"
        )
        print(
            f"   MTP: {config.mtp_num_predictions} predictions"
            if config.enable_mtp
            else "   MTP: Disabled"
        )
        print(f"\n   Total Parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
        print("   Target Active: ~3.1B parameters")

    def _init_weights(self, module):
        if self.use_kronecker and self.kronecker_embeddings is not None:
            for name, param in self.kronecker_embeddings.named_modules():
                if module is param:
                    return

        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffs)):
            return

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self, input_ids, next_token_ids=None, attention_mask=None, return_loss=False
    ):
        """
        Forward pass with Multi-Token Prediction.

        Args:
            input_ids: [B, T] - Input token IDs
            next_token_ids: [B, T] - Optional for MTP (t+1 tokens)
            attention_mask: Optional attention mask
            return_loss: Whether to return auxiliary loss

        Returns:
            - logits_ntp: [B, T, vocab_size] - Next Token Prediction
            - logits_mtp: [B, T, vocab_size] or None - Multi-Token Prediction
            - aux_loss: Scalar tensor (if return_loss=True)
        """
        batch_size, seq_len = input_ids.size()

        # Embeddings
        if self.use_kronecker:
            EMB = self.kronecker_embeddings(input_ids)
            dtype_target = self.pf_to_model.weight.dtype
            x = self.pf_to_model(EMB.to(dtype=dtype_target))
            x = self.embed_norm(x)
        else:
            x = self.token_embed(input_ids)

        # Expand to streams
        B, T, D = x.shape
        x_stream = torch.zeros(B, T, self.n_streams, D, device=x.device, dtype=x.dtype)
        x_stream[:, :, 0, :] = x

        # Pass through reversible stack
        x_stream, total_aux_loss = self.stack(x_stream)

        # Collapse streams
        h_main = x_stream.mean(dim=2)
        h_main = self.norm(h_main)

        # NTP Prediction
        logits_ntp = self.lm_head(h_main)

        # MTP Prediction
        logits_mtp = None
        if self.mtp_block is not None and next_token_ids is not None:
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_ids_use = next_token_ids[:, :min_len]

            if self.use_kronecker:
                next_emb = self.kronecker_embeddings(next_ids_use)
                next_emb = self.pf_to_model(
                    next_emb.to(dtype=self.pf_to_model.weight.dtype)
                )
                next_emb = self.embed_norm(next_emb)
            else:
                next_emb = self.token_embed(next_ids_use)

            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=None)
            logits_mtp = self.lm_head(self.norm(h_mtp))

        if return_loss:
            return logits_ntp, logits_mtp, total_aux_loss
        return logits_ntp, logits_mtp


# ============================================================================
# Factory Function
# ============================================================================


def create_model_70b(embedding_type="kronecker", bpe_vocab=None, pf_codec=None):
    """
    Create 70B model with default configuration.

    Args:
        embedding_type: "kronecker" or "standard"
        bpe_vocab: Required for Kronecker embeddings
        pf_codec: Required for Kronecker embeddings

    Returns:
        Model70B instance
    """
    config = ModelConfig()
    return Model70B(
        config, embedding_type=embedding_type, bpe_vocab=bpe_vocab, pf_codec=pf_codec
    )


if __name__ == "__main__":
    # Calculate actual metrics from weight_calculator.py
    from weight_calculator import LightningCalculator, LightningConfig

    config_calc = LightningConfig(
        vocab_size=131072,
        hidden_size=4096,
        target_params=1e9,
        attention_type="gsa",
        deltanet_layer_ratio=0.75,
        num_routed_experts_active=0,  # Dense model, no MoE
        num_shared_experts=0,  # No MoE, pure dense model
        expert_intermediate_size=1024,  # Not used in dense model
        shared_expert_intermediate_size=2048,  # Acts as dense FFN when MoE is disabled
        enable_mtp=True,
        mtp_num_predictions=2,
        num_experts_override=0,  # Dense model
        num_layers_override=8,
    )

    calc = LightningCalculator(config_calc)

    # Use expert override if provided, otherwise solve for optimal expert count
    if config_calc.num_experts_override is not None:
        num_experts = config_calc.num_experts_override
        print(f"⚙️  Using manual expert override: {num_experts} total experts\n")
    else:
        num_experts = calc.solve_for_experts()
        print(f"✓ Solved for {num_experts} optimal experts\n")

    report_df, _ = calc.generate_report(num_experts)

    # Extract actual values
    active_row = report_df[report_df["Component"] == "TOTAL ACTIVE PARAMETERS"]
    total_row = report_df[report_df["Component"] == "TOTAL MODEL PARAMETERS"]
    active_params = float(
        str(active_row["Total Contribution"].iloc[0]).replace(" B", "")
    )
    total_params = float(str(total_row["Total Contribution"].iloc[0]).replace(" B", ""))
    sparsity = total_params / active_params

    config = ModelConfig()

    print("=" * 80)
    print("1B DENSE MODEL ARCHITECTURE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Total Params: {total_params:.3f}B")
    print(f"  Active Params: {active_params:.3f}B")
    print(f"  Sparsity: {sparsity:.1f}x")
    print("\nAttention Mix:")
    print(
        f"  DeltaNet: {config.num_deltanet_layers} layers ({config.num_deltanet_layers/config.num_layers*100:.0f}%) - O(N) for 256k context"
    )
    print(
        f"  GSA: {config.num_gsa_layers} layers ({config.num_gsa_layers/config.num_layers*100:.0f}%) - Adaptive sparse quality"
    )
    print("\nModel Type:")
    if num_experts == 0:
        print("  DENSE MODEL (No MoE)")
        print(f"  Dense FFN intermediate: {config.shared_expert_intermediate_size}")
    else:
        print("  MoE MODEL")
        print(f"  Real Experts: {num_experts}")
        print(f"  Null Experts: {num_experts} (ρ={config.data_sparsity})")
        print(f"  Total slots: {config.total_expert_slots}")
        print(
            f"  Top-k: {config.top_k} (dynamic 0-{config.top_k}, avg {config_calc.num_routed_experts_active})"
        )
        print(
            f"  Shared Expert FFN: {config.shared_expert_intermediate_size} (always active)"
        )
        print(f"  Routed Expert FFN: {config.expert_intermediate_size} (sparse)")
    print(f"\nContext: {config.max_seq_len:,} tokens")
    print("=" * 80)
