"""
1B Dense Baseline Model — Test 14: DeltaNet + GSA (DDDGDDDG), no fused CE

Configuration:
- 1.513B total parameters, 1.513B active parameters (100% dense - no MoE)
- 131,072 vocabulary (2^17)
- 4096 hidden size, 8 layers: 6 DeltaNet + 2 GSA (DDDGDDDG pattern)
- No experts - Dense FFN with 2048 intermediate size (Liger SwiGLU MLP)
- Multi-Token Prediction (MTP) with 2 predictions
- Multi-Head Composition (mHC) with 4 streams
- Reversible Midpoint Integration for memory efficiency
- Target: 256k context length
- Enhanced with Memory Stream Recurrence for infinite-length documents

Architecture:
- DDDGDDDG: layers 1–3 DeltaNet, layer 4 GSA, layers 5–7 DeltaNet, layer 8 GSA.
- DeltaNet: fla chunk_gated_delta_rule (O(N) linear attention).
- GSA: Test 14 Triton sparse attn + indexer (same kernels as before).
- Liger: RoPE (liger_rotary_pos_emb), MLP (LigerSwiGLUMLP). No fused CE (logits only; CE in train.py).

Purpose: Test 14 — Test 6/7-style hybrid with our new kernels, standard CE (no fused CE).
"""

import importlib
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.profiler import time_region


# ── Triton Kernel Imports ────────────────────────────────────────────────────
# Mirror 70B import resolution so 1B behaves identically across launch contexts.
def _import_kernels_module():
    try:
        from llm import kernels as kernels_module

        return kernels_module
    except Exception:
        return None


_kernels_module = _import_kernels_module()
if _kernels_module is not None:
    HAS_TRITON = bool(getattr(_kernels_module, "HAS_TRITON", False))
    HAS_FLA = bool(getattr(_kernels_module, "HAS_FLA", False))
    triton_sparse_attention = getattr(_kernels_module, "triton_sparse_attention", None)
    pytorch_sparse_attention = getattr(
        _kernels_module, "pytorch_sparse_attention", None
    )
    triton_sinkhorn_knopp = getattr(_kernels_module, "triton_sinkhorn_knopp", None)
    pytorch_sinkhorn_knopp = getattr(_kernels_module, "pytorch_sinkhorn_knopp", None)
    triton_rmsnorm = getattr(_kernels_module, "triton_rmsnorm", None)
    pytorch_rmsnorm = getattr(_kernels_module, "pytorch_rmsnorm", None)
    TritonRMSNorm = getattr(_kernels_module, "TritonRMSNorm", None)
    fused_indexer_topk = getattr(_kernels_module, "fused_indexer_topk", None)
    fla_gated_delta_rule = getattr(_kernels_module, "fla_gated_delta_rule", None)
else:
    HAS_TRITON = False
    HAS_FLA = False
    triton_sparse_attention = None
    pytorch_sparse_attention = None
    triton_sinkhorn_knopp = None
    pytorch_sinkhorn_knopp = None
    triton_rmsnorm = None
    pytorch_rmsnorm = None
    TritonRMSNorm = None
    fused_indexer_topk = None
    fla_gated_delta_rule = None

HAS_FUSED_INDEXER = fused_indexer_topk is not None

# ── torch.compile mode ──────────────────────────────────────────────────────
# When True, custom Triton kernels (RMSNorm, Sinkhorn) are bypassed in favor
# of PyTorch ops that torch.compile can fuse with surrounding operations.
# Set by enable_torch_compile() before training starts.
_TORCH_COMPILE_MODE = False


# ── Liger ops (Test 14: RoPE, MLP only — no fused CE) ────────────────────────
def _import_liger_ops_module():
    try:
        from . import liger_ops as liger_module

        return liger_module
    except Exception:
        pass
    src_root = Path(__file__).resolve().parents[1]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    return importlib.import_module("models.liger_ops")


_liger_module = _import_liger_ops_module()
LigerSwiGLUMLP = _liger_module.LigerSwiGLUMLP
liger_rotary_pos_emb = _liger_module.liger_rotary_pos_emb
liger_silu_mul = _liger_module.liger_silu_mul

# ── Kernel availability diagnostics ──────────────────────────────────────────
_kernel_log = logging.getLogger("recurrence_model_1b.kernels")
if not _kernel_log.handlers:
    _kernel_log.addHandler(logging.StreamHandler())
    _kernel_log.setLevel(logging.INFO)

_cuda_available = torch.cuda.is_available()
_kernel_log.info("=" * 60)
_kernel_log.info("Kernel Availability Report (1B):")
_kernel_log.info(f"  CUDA available:       {_cuda_available}")
_kernel_log.info(f"  HAS_TRITON:           {HAS_TRITON}")
_kernel_log.info(
    f"  Triton RMSNorm:       {'ENABLED' if HAS_TRITON and triton_rmsnorm is not None and _cuda_available else 'FALLBACK (PyTorch)'}"
)
_kernel_log.info(
    f"  Triton Sinkhorn:      {'ENABLED' if HAS_TRITON and triton_sinkhorn_knopp is not None and _cuda_available else 'FALLBACK (PyTorch)'}"
)
_kernel_log.info(
    f"  Triton Sparse Attn:   {'ENABLED' if HAS_TRITON and triton_sparse_attention is not None and _cuda_available else 'FALLBACK (PyTorch)'}"
)
_kernel_log.info(
    f"  fla GatedDeltaRule:   {'ENABLED' if HAS_FLA and fla_gated_delta_rule is not None and _cuda_available else 'UNAVAILABLE (pip install fla)'}"
)
if not _cuda_available:
    _kernel_log.info(
        "  NOTE: Triton kernels require CUDA. Running on MPS/CPU uses PyTorch fallbacks."
    )
_kernel_log.info("=" * 60)

# Note: Importing for backwards compatibility - we define KroneckerEmbeddings inline
# from kronecker_se_decoder import PFConfig, PFCodec


def _token_keep_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize attention masks to a boolean keep-mask of shape [B, T]."""
    if attention_mask is None:
        return None

    mask = attention_mask
    if mask.dim() == 2:
        pass
    elif mask.dim() == 3 and mask.size(1) == 1:
        mask = mask[:, 0, :]
    elif mask.dim() == 4 and mask.size(1) == 1 and mask.size(2) == 1:
        mask = mask[:, 0, 0, :]
    elif mask.dim() == 4 and mask.size(1) == 1 and mask.size(2) == seq_len:
        # Convert [B, 1, T, T] to [B, T] key-validity.
        mask = mask[:, 0, :, :]
        if mask.dtype == torch.bool:
            mask = mask.any(dim=1)
        elif torch.is_floating_point(mask):
            if torch.any(mask < 0):
                mask = mask.max(dim=1).values >= 0
            else:
                mask = mask.max(dim=1).values > 0
        else:
            mask = mask.max(dim=1).values > 0
    else:
        raise ValueError(
            f"Unsupported attention_mask shape {tuple(mask.shape)}. "
            "Expected [B,T], [B,1,T], [B,1,1,T], or [B,1,T,T]."
        )

    if mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"attention_mask shape {tuple(mask.shape)} does not match expected {(batch_size, seq_len)}."
        )

    if mask.dtype == torch.bool:
        keep = mask
    elif torch.is_floating_point(mask):
        if torch.any(mask < 0):
            keep = mask >= 0
        else:
            keep = mask > 0
    else:
        keep = mask > 0

    return keep.to(device=device, dtype=torch.bool)


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

    # Attention Mix (75% DeltaNet / 25% GSA) — DDDGDDDG pattern
    num_deltanet_layers = 6
    num_gsa_layers = 2

    # DeltaNet Configuration
    delta_v_heads = 32  # hidden_size / delta_head_dim = 4096 / 128
    delta_head_dim = 128
    delta_gate_dim = 384  # 9.4% of hidden_size

    # GSA Configuration
    gsa_num_heads = 16  # hidden_size / attn_head_dim = 4096 / 256
    gsa_head_dim = 256
    gsa_k_base = 128  # FIX-PERF-02: Reduced from 512 — at T=4096, 512 keys/query = 25% dense; 128 is sufficient
    gsa_k_min = 32
    gsa_k_max = 256  # FIX-PERF-02: Reduced from 1024 — limits atomic scatter in dK/dV backward kernel
    gsa_indexer_heads = 4

    # MoE Configuration (DENSE MODEL - No MoE) — same as Test 5 for parity
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
    sinkhorn_iters = (
        20  # Keep at 20 — sufficient for mHC routing quality; do not reduce
    )

    # Context and RoPE (standard RoPE)
    max_seq_len = 262144  # 256k context
    rope_base = 10000
    rope_original_max_position = 8192  # Original training context
    rope_scaling_factor = 32.0  # 256k / 8k = 32x extension

    # Training
    dropout = 0.0  # Required for reversible integration
    require_fused_deltanet_kernel = True
    require_fused_gsa_kernel = True


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
        # FIX #27: Make PF_table non-persistent (saves ~2GB in checkpoints)
        # Will be regenerated deterministically from vocab at load time
        self.register_buffer("PF_table", pf_tensor, persistent=False)

    def forward(self, token_ids):
        """
        Forward pass: fetch and normalize Kronecker embeddings.

        Args:
            token_ids: Token indices (B, T)

        Returns:
            Normalized embeddings (B, T, D=8192)
        """
        PF = self.PF_table[token_ids].to(dtype=torch.float32)  # type: ignore
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
    """
    Root Mean Square Layer Normalization with fp32 statistics.

    FIX #43: Computes variance in fp32 for numerical stability at 256k context.
    Critical for preventing rare NaN spikes with bf16/fp16 training.

    Triton acceleration: When available, uses fused Triton kernel that computes
    variance + rsqrt + weight multiply in a single kernel launch.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._use_triton = HAS_TRITON and triton_rmsnorm is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton path: Liger-style fused forward+backward (LigerRMSNormFunction in kernels).
        # Safe with grad enabled — reversible backward uses this during recompute.
        if self._use_triton and x.is_cuda and triton_rmsnorm is not None:
            try:
                return triton_rmsnorm(x, self.weight, self.eps)
            except Exception:
                pass  # Fall through to PyTorch path

        # PyTorch fallback (FIX #43: fp32 variance for stability)
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm.to(x.dtype) + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE rotary positional embedding (YaRN removed).

    MEMORY OPTIMIZATION:
    Computes cos/sin on-the-fly instead of caching to save VRAM.

    Caching approach would use: 262,144 × 128 × 2 = 268MB per layer × 8 layers = 2.1GB VRAM.
    On-the-fly computation: ~0MB cache, only 5-10% slower (negligible with modern GPUs).

    For 256k context training, we need every GB of VRAM for activations and optimizer states.
    Trading 5-10% RoPE compute time for 2.1GB free memory is an excellent trade-off.
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

        # Compatibility note:
        # `original_max_position_embeddings` and `scaling_factor` remain in the
        # signature for checkpoint/config compatibility, but YaRN scaling is
        # intentionally removed and standard RoPE is applied.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_cos_sin(self, seq_len: int, device, dtype=None):
        """
        Compute cos/sin on-the-fly for given sequence length.
        FIX #30: Uses forward-pass cache if available (set by model.forward())
        FIX #39: Include dtype in cache key for mixed-precision safety
        FIX #42: Cast output to requested dtype (prevents float32/bf16 mismatches)
        Saves 2.1GB VRAM compared to persistent caching (268MB × 8 layers).
        """
        # FIX #30: Check if cache exists (set at model forward start)
        # FIX #39: Include dtype in cache key (default to None for backward compatibility)
        cache_key = (seq_len, device, dtype)
        if hasattr(self, "_forward_cache") and cache_key in self._forward_cache:  # type: ignore
            return self._forward_cache[cache_key]  # type: ignore

        t = torch.arange(seq_len, device=device).float()
        freqs = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # type: ignore
        emb = torch.cat((freqs, freqs), dim=-1)

        # FIX #42: Cast to requested dtype to match query/key precision
        # Prevents implicit upcasts and memory/bandwidth issues at 256k context
        cos_out = emb.cos()
        sin_out = emb.sin()
        if dtype is not None:
            cos_out = cos_out.to(dtype)
            sin_out = sin_out.to(dtype)

        return cos_out, sin_out

    @staticmethod
    def _apply_rotary(x, cos, sin):
        return liger_rotary_pos_emb(x, cos, sin)


# ============================================================================
# Helper Modules for Gated DeltaNet
# ============================================================================


class ShortConvolution(nn.Module):
    """
    Short convolution layer with causal padding.
    Used in GatedDeltaNet for local context integration.
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
    Key components: alpha (decay), beta (writing strength), L2 norm on Q/K,
    short convolutions. Uses fla's chunk_gated_delta_rule when available.
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
        require_fused_kernel=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_output_norm = use_output_norm
        self.require_fused_kernel = require_fused_kernel

        key_dim = num_heads * head_dim
        value_dim = num_heads * head_dim
        self._key_dim = key_dim
        self._value_dim = value_dim

        # PERF: Fused QKV+G projection — single GEMM reads x from HBM once
        # instead of 4 separate reads (saves ~3ms/call on 8×H100)
        self.qkvg_proj = nn.Linear(
            hidden_size, key_dim + key_dim + value_dim + value_dim, bias=False
        )
        self.o_proj = nn.Linear(value_dim, hidden_size, bias=False)

        # PERF: Fused beta+gk projection — single GEMM instead of 2 reads of x
        self.bgk_proj = nn.Linear(hidden_size, num_heads * 2, bias=True)

        self.q_conv1d = ShortConvolution(
            key_dim, conv_size=conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            key_dim, conv_size=conv_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            value_dim, conv_size=conv_size, activation="silu"
        )

        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))

        self.D = nn.Parameter(torch.ones(num_heads))
        dt_bias = torch.rand(num_heads) * 0.02 - 0.01
        self.dt_bias = nn.Parameter(dt_bias)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=4096,
            base=10000,
            original_max_position_embeddings=4096,
            scaling_factor=1.0,
        )

        if use_output_norm:
            self.o_norm = FusedRMSNormSwishGate(head_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkvg_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        # STABILITY-FIX: Reduce init scale for alpha/beta gating logic
        nn.init.normal_(self.bgk_proj.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.bgk_proj.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Checkpoint compatibility: load old separate q/k/v/g_proj into fused qkvg_proj."""
        q_key = prefix + "q_proj.weight"
        if q_key in state_dict:
            state_dict[prefix + "qkvg_proj.weight"] = torch.cat(
                [
                    state_dict.pop(prefix + "q_proj.weight"),
                    state_dict.pop(prefix + "k_proj.weight"),
                    state_dict.pop(prefix + "v_proj.weight"),
                    state_dict.pop(prefix + "g_proj.weight"),
                ],
                dim=0,
            )

        b_key = prefix + "b_proj.weight"
        if b_key in state_dict:
            state_dict[prefix + "bgk_proj.weight"] = torch.cat(
                [
                    state_dict.pop(prefix + "b_proj.weight"),
                    state_dict.pop(prefix + "gk_proj.weight"),
                ],
                dim=0,
            )
            state_dict[prefix + "bgk_proj.bias"] = torch.cat(
                [
                    state_dict.pop(prefix + "b_proj.bias"),
                    state_dict.pop(prefix + "gk_proj.bias"),
                ],
                dim=0,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device
        token_keep = _token_keep_mask(attention_mask, B, T, device)
        token_keep_f = None

        # ── Fused projections (single GEMM reads x once from HBM) ────────────
        qkvg = self.qkvg_proj(x)
        q, k, v, g = qkvg.split(
            [self._key_dim, self._key_dim, self._value_dim, self._value_dim], dim=-1
        )
        # split() creates non-contiguous views — inductor generates wrong reinterpret_tensor
        q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()

        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        g = g.view(B, T, self.num_heads, self.head_dim)

        # L2 Normalization MUST happen first, otherwise it destroys the RoPE rotational structure
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        cos, sin = self.rotary_emb._compute_cos_sin(T, device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        # ── Alpha/Beta gating (fused bgk projection + log-space g) ────────────
        bgk = self.bgk_proj(x)
        b_raw, gk = bgk.split([self.num_heads, self.num_heads], dim=-1)
        b_raw, gk = b_raw.contiguous(), gk.contiguous()
        beta = torch.sigmoid(b_raw).unsqueeze(-1)  # (B, T, num_heads, 1)
        A = torch.exp(self.A_log)
        # g_logspace = log(alpha) directly — FLA needs log-space, so skip the
        # pointless exp() here followed by log() in fla_deltanet.py
        g_logspace = -A.view(1, 1, self.num_heads) * F.softplus(
            gk + self.dt_bias
        )  # (B, T, H)
        # Safety clamp: the old exp→log path had clamp(alpha, min=1e-6) before log(),
        # which effectively clamped g to >= log(1e-6) ≈ -13.8. Preserve that safety.
        g_logspace = g_logspace.clamp(min=-13.8)
        alpha = torch.exp(g_logspace).unsqueeze(-1)  # (B, T, H, 1) — only for masking

        if token_keep is not None:
            token_keep_f = token_keep.to(dtype=q.dtype).view(B, T, 1, 1)
            q = q * token_keep_f
            k = k * token_keep_f
            v = v * token_keep_f
            g = g * token_keep_f
            beta = beta * token_keep_f
            alpha = alpha * token_keep_f + (1.0 - token_keep_f)

        fla_available = HAS_FLA and fla_gated_delta_rule is not None and q.is_cuda
        if self.require_fused_kernel and not fla_available:
            raise RuntimeError(
                "DeltaNet fused kernel is required but unavailable. "
                f"HAS_FLA={HAS_FLA}, fla_gated_delta_rule={fla_gated_delta_rule is not None}, "
                f"q.is_cuda={q.is_cuda}. Install fla: pip install fla"
            )

        if fla_available and fla_gated_delta_rule is not None:
            try:
                with time_region("deltanet.fla"):
                    o = fla_gated_delta_rule(
                        q=q,
                        k=k,
                        v=v,
                        alpha=alpha,
                        beta=beta,
                        D=self.D,
                        num_heads=self.num_heads,
                        g_precomputed=g_logspace,
                    )
            except Exception as e:
                if self.require_fused_kernel:
                    raise RuntimeError(
                        "DeltaNet fused kernel execution failed and fallback is disabled."
                    ) from e
                raise
        else:
            raise RuntimeError(
                "DeltaNet requires fla (pip install fla) when require_fused_kernel=True."
            )

        if self.use_output_norm:
            o_flat = o.reshape(B * T * self.num_heads, self.head_dim)
            g_flat = g.reshape(B * T * self.num_heads, self.head_dim)
            o_normed = self.o_norm(o_flat, g_flat)
            o = o_normed.view(B, T, self.num_heads, self.head_dim)
        else:
            o = o * torch.sigmoid(g)

        if token_keep_f is not None:
            o = o * token_keep_f

        o = o.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o)


# ============================================================================
# Gated Sparse Attention (25% of layers in Test 14 — DDDGDDDG)
# ============================================================================


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) - arXiv:2601.15305v1

    Implements adaptive sparse attention with gating. In Test 14, layers 4 and 8
    use GSA (DDDGDDDG); layers 1–3, 5–7 use GatedDeltaNet.

    Memory complexity: O(T·k) via fused_indexer_topk chunked kernel.
    Architecture:
    - Shared indexer keys (W_Ik → [B, T, d_idx]) across indexer heads
    - Per-attention-head diversity via head_importance_bias on attention logits
    - Adaptive sparsity budget k_t from variance-based heuristic
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
        require_fused_kernel=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.require_fused_kernel = require_fused_kernel

        # Adaptive Sparsity Hyperparams
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.indexer_heads = indexer_heads

        # Lightning Indexer (shared keys across indexer heads for kernel compatibility)
        self.d_idx = (
            128  # ARCH-01: paper Table 1 specifies d_idx=128 (was 32, 4× under-spec)
        )
        # PERF: Fused indexer projection — single GEMM instead of 3 reads of x
        self._idx_split = [indexer_heads * self.d_idx, self.d_idx, indexer_heads]
        self.idx_proj = nn.Linear(hidden_size, sum(self._idx_split), bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(indexer_heads))

        self.register_buffer("variance_ema", torch.tensor(1.0))
        # Snapshot of variance_ema captured at the start of each reversible
        # forward pass (torch.no_grad()).  The backward reconstruct (torch.enable_grad())
        # reads this snapshot instead of the live EMA, guaranteeing that
        # fused_indexer_topk produces identical k_t / top_indices in both passes.
        # Without this, gradient-accumulation or async NCCL updates can mutate
        # variance_ema between forward and reconstruct, breaking reversibility.
        self.register_buffer("_variance_ema_snapshot", torch.tensor(1.0))
        self.variance_alpha = 0.01

        # PERF: Fused attention + gating projection — single GEMM instead of 5 reads of x
        self.qkvgg_proj = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary embeddings (standard RoPE)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
            original_max_position_embeddings=rope_original_max,
            scaling_factor=rope_scaling_factor,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.idx_proj, self.qkvgg_proj, self.o_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Checkpoint compatibility: load old separate projections into fused ones."""
        # Indexer: W_Iq + W_Ik + W_Iw → idx_proj
        iq_key = prefix + "W_Iq.weight"
        if iq_key in state_dict:
            state_dict[prefix + "idx_proj.weight"] = torch.cat(
                [
                    state_dict.pop(prefix + "W_Iq.weight"),
                    state_dict.pop(prefix + "W_Ik.weight"),
                    state_dict.pop(prefix + "W_Iw.weight"),
                ],
                dim=0,
            )

        # Attention + gates: W_q + W_k + W_v + W_gv + W_go → qkvgg_proj
        wq_key = prefix + "W_q.weight"
        if wq_key in state_dict:
            state_dict[prefix + "qkvgg_proj.weight"] = torch.cat(
                [
                    state_dict.pop(prefix + "W_q.weight"),
                    state_dict.pop(prefix + "W_k.weight"),
                    state_dict.pop(prefix + "W_v.weight"),
                    state_dict.pop(prefix + "W_gv.weight"),
                    state_dict.pop(prefix + "W_go.weight"),
                ],
                dim=0,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device
        token_keep = _token_keep_mask(attention_mask, B, T, device)

        gsa_fused_available = (
            HAS_FUSED_INDEXER and triton_sparse_attention is not None and x.is_cuda
        )
        is_grad_enabled = torch.is_grad_enabled()
        # Training uses differentiable PyTorch sparse attention; fused Triton
        # path remains required for no-grad / inference execution.
        if (
            self.require_fused_kernel
            and (not is_grad_enabled)
            and not gsa_fused_available
        ):
            raise RuntimeError(
                "GSA fused kernels are required but unavailable for no-grad execution. "
                f"fused_indexer_topk={HAS_FUSED_INDEXER}, "
                f"triton_sparse_attention={triton_sparse_attention is not None}, "
                f"x.is_cuda={x.is_cuda}."
            )

        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        is_reconstruct = self.training and torch.is_grad_enabled()

        if not hasattr(self, "_index_cache"):
            self._index_cache = []

        if is_reconstruct and len(self._index_cache) > 0:
            (
                var_t,
                k_t,
                keep_mask,
                base_idx,
                leak_attempt_mask,
                attempt_den,
                leak_final_mask,
                final_den,
            ) = self._index_cache.pop(0)
            k_limit = base_idx.size(-1)

            self.last_gsa_leak_attempt_fraction = (
                leak_attempt_mask.float().sum() / attempt_den
            ).detach()
            self.last_gsa_leak_fraction = (
                leak_final_mask.float().sum() / final_den
            ).detach()
        else:
            # Lightning Indexer — O(T·k) via fused chunked kernel
            # Uses fused_indexer_topk to avoid materializing [B, heads, T, T] importance scores.
            # PERF: Single GEMM for all indexer projections (reads x once)
            idx_out = self.idx_proj(x)
            q_I_raw, k_I, w_raw = idx_out.split(self._idx_split, dim=-1)
            q_I_raw, k_I, w_raw = (
                q_I_raw.contiguous(),
                k_I.contiguous(),
                w_raw.contiguous(),
            )
            q_I = q_I_raw.view(B, T, self.indexer_heads, self.d_idx)  # [B, T, 4, 128]
            scale_idx = 1.0 / math.sqrt(self.d_idx)

            if is_reversible_forward:
                self._variance_ema_snapshot.copy_(self.variance_ema)  # type: ignore
            ema_for_indexer = (
                self._variance_ema_snapshot
                if is_reversible_forward
                else self.variance_ema
            )

            if not HAS_FUSED_INDEXER:
                raise RuntimeError(
                    "GSA fused indexer kernel is required but unavailable. "
                    "Fallback indexer path is disabled."
                )

            with time_region("gsa.indexer"):
                var_t, k_t, top_indices = fused_indexer_topk(  # type: ignore
                    q=q_I,
                    k=k_I,
                    w=w_raw,
                    b=self.gate_bias,
                    scale=scale_idx,
                    causal=True,
                    k_base=self.k_base,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    variance_ema=ema_for_indexer,  # snapshot or live
                    is_training=False,
                    sink_size=4,
                )

            if is_reversible_forward:
                var_t_mean = var_t.mean().detach()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        var_t_mean, op=torch.distributed.ReduceOp.AVG
                    )
                self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)  # type: ignore

            # Build per-query keep mask from adaptive k_t with strict causal safety.
            k_limit = top_indices.size(-1)
            base_idx = top_indices.long()  # [B, T, k_limit]
            q_pos = torch.arange(T, device=device, dtype=base_idx.dtype).view(1, T, 1)
            causal_cap = (q_pos + 1).to(dtype=k_t.dtype)
            k_t = torch.minimum(k_t, causal_cap.squeeze(-1)).clamp(min=1)

            range_k = torch.arange(k_limit, device=device)
            keep_mask = range_k.view(1, 1, -1) < k_t.unsqueeze(-1)  # [B, T, k_limit]
            causal_selected = base_idx <= q_pos

            if token_keep is not None:
                query_keep = token_keep.unsqueeze(-1)
                invalid_query = ~token_keep
                if invalid_query.any():
                    fallback_idx = (
                        torch.arange(T, device=device).view(1, T).expand(B, T)
                    )
                    base_idx = base_idx.clone()
                    base_idx[..., 0] = torch.where(
                        invalid_query, fallback_idx, base_idx[..., 0]
                    )
                    causal_selected = base_idx <= q_pos

                key_keep = torch.gather(
                    token_keep, dim=1, index=base_idx.reshape(B, -1)
                ).view(B, T, k_limit)
                keep_mask = keep_mask & key_keep & query_keep

                # Keep at least one index for masked queries to avoid empty-kernel rows.
                if invalid_query.any():
                    keep_mask = keep_mask.clone()
                    keep_mask[..., 0] = keep_mask[..., 0] | invalid_query

            attempt_keep_mask = keep_mask
            leak_attempt_mask = attempt_keep_mask & ~causal_selected
            keep_mask = attempt_keep_mask & causal_selected
            leak_final_mask = keep_mask & ~causal_selected
            attempt_den = attempt_keep_mask.sum().clamp(min=1).float()
            final_den = keep_mask.sum().clamp(min=1).float()

            self.last_gsa_leak_attempt_fraction = (
                leak_attempt_mask.float().sum() / attempt_den
            ).detach()
            self.last_gsa_leak_fraction = (
                leak_final_mask.float().sum() / final_den
            ).detach()

            if is_reversible_forward:
                self._index_cache.append(
                    (
                        var_t,
                        k_t,
                        keep_mask,
                        base_idx,
                        leak_attempt_mask,
                        attempt_den,
                        leak_final_mask,
                        final_den,
                    )
                )

        # Dual Gating & Attention Projections
        # PERF: Single GEMM for q, k, v, gate_v, gate_o (reads x once)
        qkvgg = self.qkvgg_proj(x)
        H = self.hidden_size
        q, k_attn, v, g_v_raw, g_o_raw = qkvgg.split([H, H, H, H, H], dim=-1)
        q, k_attn, v, g_v_raw, g_o_raw = (
            q.contiguous(),
            k_attn.contiguous(),
            v.contiguous(),
            g_v_raw.contiguous(),
            g_o_raw.contiguous(),
        )

        g_v = torch.sigmoid(g_v_raw)
        v = v * g_v

        q = q.view(B, T, self.num_heads, self.head_dim)
        k_attn = k_attn.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        if token_keep is not None:
            token_keep_f = token_keep.to(dtype=q.dtype).view(B, T, 1, 1)
            q = q * token_keep_f
            k_attn = k_attn * token_keep_f
            v = v * token_keep_f

        # Rotary (computed on-the-fly to save 2.1GB VRAM)
        cos, sin = self.rotary_emb._compute_cos_sin(T, device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k_attn = self.rotary_emb._apply_rotary(k_attn, cos, sin)

        # ── Sparse attention via triton_sparse_attention kernel ────────
        # O(T*k) complexity: kernel iterates only over k_limit selected
        # keys per query using online softmax. No T×T tensor ever created.
        # Memory: O(B*H*T*k_limit) for indices/mask, NOT O(T²).
        #
        # At T=256k, B=1, k_limit=1024:
        #   indices: [1, 16, 256k, 1024] int64 = 32GB  (vs 128GB for [B,1,T,T])
        #   BUT indices are shared across heads → [B, 1, T, k_limit] expanded
        #   as views, so actual memory = [B, T, k_limit] * 8 bytes = 2GB.

        # Kernel expects indices: [B, H, T, k_sel] int64, mask: [B, H, T, k_sel] float32
        # base_idx is [B, T, k_limit], keep_mask is [B, T, k_limit] bool
        # Expand to [B, H, T, k_limit] as views (stride=0 on H dim).
        # Triton kernel uses stride-based access, so zero-stride broadcast works
        # without copying. Memory: only [B, T, k_limit] actually allocated.
        sparse_idx = base_idx.unsqueeze(1).expand(B, self.num_heads, T, k_limit)
        sparse_mask = (
            keep_mask.float().unsqueeze(1).expand(B, self.num_heads, T, k_limit)
        )

        scale_attn = 1.0 / math.sqrt(self.head_dim)

        # q, k_attn, v are [B, T, H, D].
        # Tests 6-11 require fused Triton sparse attention for both
        # forward and backward in training.
        if not (HAS_TRITON and triton_sparse_attention is not None and q.is_cuda):
            raise RuntimeError(
                "GSA fused sparse attention kernel (forward+backward) is required. "
                "PyTorch fallback is disabled for this test."
            )
        with time_region("gsa.sparse_attn"):
            o_sparse = triton_sparse_attention(
                q,
                k_attn,
                v,
                sparse_idx,
                sparse_mask,
                scale_attn,
                use_triton_backward=True,
            )
        if token_keep is not None:
            o_sparse = o_sparse * token_keep.to(dtype=o_sparse.dtype).view(B, T, 1, 1)

        # Output is [B, T, H, D] from kernel, reshape to [B, T, hidden_size]
        o_sparse = o_sparse.contiguous().view(B, T, self.hidden_size)

        # Output gate (g_o_raw already computed in fused qkvgg_proj above)
        g_o = torch.sigmoid(g_o_raw)

        return self.o_proj(o_sparse * g_o)


# ============================================================================
# Dense FFN only (Test 13: no MoE)
# ============================================================================


class DenseMLP(nn.Module):
    """SwiGLU FFN (Liger MLP) for dense models — Test 14."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.mlp = LigerSwiGLUMLP(
            in_features=d_model,
            hidden_features=d_hidden,
            out_features=d_model,
            bias=False,
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.mlp.gate_up_proj, self.mlp.down_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LightningMLP(nn.Module):
    """Dense MLP only (Test 13: no MoE). Wraps DenseMLP for decoder and MTP blocks."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = DenseMLP(
            d_model=config.hidden_size,
            d_hidden=config.shared_expert_intermediate_size,
        )

    def forward(self, x):
        out = self.mlp(x)
        # Reversible stack expects (out, aux_loss); DenseMLP returns tensor only
        aux = (out * 0.0).sum()  # zero aux with grad_fn for backprop
        return out, aux


# ============================================================================
# mHC (Multi-Head Composition) - From test model
# ============================================================================


def sinkhorn_knopp(
    logits: torch.Tensor, iters: int = 5, eps: float = 1e-6
) -> torch.Tensor:
    """
    Sinkhorn-Knopp doubly-stochastic normalisation.
    Dispatches to fused Triton kernel (single launch) when available,
    falling back to PyTorch for CPU or when Triton is absent.

    FIX-PERF-06: Removed the `not torch.is_grad_enabled()` guard.
    The Triton kernel is pure computation with no autograd hooks, so it
    is safe inside torch.enable_grad() contexts (e.g. reversible backward
    recomputation pass). Previously the backward path fell back to 20
    separate PyTorch kernel launches per sublayer instead of 1 Triton call.
    """
    if (
        not _TORCH_COMPILE_MODE
        and HAS_TRITON
        and triton_sinkhorn_knopp is not None
        and logits.is_cuda
    ):
        # Stable exp: subtract max along last dim before passing to kernel
        logits_stable = logits - logits.amax(dim=-1, keepdim=True)
        try:
            with time_region("sinkhorn.triton"):
                return triton_sinkhorn_knopp(logits_stable, num_iters=iters, eps=eps)
        except Exception:
            pass  # fall through to PyTorch
    # PyTorch fallback (CPU, Triton unavailable, or torch.compile mode)
    with time_region("sinkhorn.pytorch"):
        return pytorch_sinkhorn_knopp(logits, num_iters=iters, eps=eps)  # type: ignore


class MHCCoeffs(nn.Module):
    """Produces routing coefficients for mHC."""

    def __init__(self, d_model: int, n_streams: int = 4, iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.iters = iters

        d_in = self.n * d_model

        # Fused projection: pre (n) + post (n) + res (n*n)
        # Avoids inductor incorrectly fusing separate Linear layers on the same input
        # (same pattern as qkvg_proj / bgk_proj / idx_proj / qkvgg_proj fusions)
        self._pre_dim = self.n
        self._post_dim = self.n
        self._res_dim = self.n * self.n
        self._phi_split = [self._pre_dim, self._post_dim, self._res_dim]
        self.phi_fused = nn.Linear(
            d_in, self._pre_dim + self._post_dim + self._res_dim, bias=False
        )

        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        self.alpha_pre = nn.Parameter(torch.tensor(0.1))
        self.alpha_post = nn.Parameter(torch.tensor(0.1))
        self.alpha_res = nn.Parameter(torch.tensor(0.1))

        self.rms = RMSNorm(d_in)

        nn.init.normal_(self.phi_fused.weight, mean=0.0, std=0.02)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Handle old checkpoints with separate phi_pre/phi_post/phi_res weights."""
        pre_key = prefix + "phi_pre.weight"
        fused_key = prefix + "phi_fused.weight"
        if pre_key in state_dict and fused_key not in state_dict:
            w_pre = state_dict.pop(pre_key)
            w_post = state_dict.pop(prefix + "phi_post.weight")
            w_res = state_dict.pop(prefix + "phi_res.weight")
            state_dict[fused_key] = torch.cat([w_pre, w_post, w_res], dim=0)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x_stream: torch.Tensor):
        B, T, n, D = x_stream.shape
        x_flat = x_stream.reshape(B, T, n * D).contiguous()
        x_flat = self.rms(x_flat)

        # Cast to weight dtype to prevent float32/bfloat16 mismatch during reversible backward
        x_flat = x_flat.to(self.phi_fused.weight.dtype)

        # Single fused projection, then split
        logits_all = self.phi_fused(x_flat)
        pre_raw, post_raw, res_raw = logits_all.split(self._phi_split, dim=-1)
        # split() creates non-contiguous views — inductor generates wrong reinterpret_tensor
        pre_raw, post_raw, res_raw = (
            pre_raw.contiguous(),
            post_raw.contiguous(),
            res_raw.contiguous(),
        )

        pre_logits = self.alpha_pre * pre_raw + self.b_pre
        post_logits = self.alpha_post * post_raw + self.b_post

        res_logits = self.alpha_res * res_raw
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
# Decoder Layer (Hybrid DeltaNet + GSA — Test 14 DDDGDDDG)
# ============================================================================


class LightningDecoderLayer(nn.Module):
    """
    Decoder layer that can be either DeltaNet or GSA.
    Type is determined at initialization (DDDGDDDG: every 4th layer is GSA).
    """

    def __init__(self, config: ModelConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type  # "deltanet" or "gsa"
        self.n_streams = config.n_streams

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
                require_fused_kernel=config.require_fused_deltanet_kernel,
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
                require_fused_kernel=config.require_fused_gsa_kernel,
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        mlp = LightningMLP(config)

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

    def force(self, x, attention_mask=None):
        """Compute residual delta for reversible integration."""
        h, aux1 = self.attn_block(x, attention_mask=attention_mask)
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
            # Must have a grad_fn for reversible midpoint backward
            # (torch.autograd.grad requires all outputs to be differentiable)
            aux = (delta * 0.0).sum()

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

        # Core sublayers (using GSA for better gradient quality)
        # MTP block runs only once per step (not 8x like backbone layers),
        # so full sparse attention cost is negligible but gradient quality is critical
        self.attn = GatedSparseAttention(
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
            require_fused_kernel=config.require_fused_gsa_kernel,
        )

        self.mlp = LightningMLP(config)

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
        if isinstance(module, (DenseMLP, MHCCoeffs)):
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

        # NOTE: Memory stream injection happens in the main Model1B.forward(),
        # not here. The MTP block receives h_t which already contains recurrence
        # information from the backbone processing.

        # mHC blocks (ignore aux_loss for clean aux-loss accounting)
        x_stream, _ = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, _ = self.mlp_block(x_stream, attention_mask=None)

        # Collapse
        x_out = x_stream.mean(dim=2)

        return x_out


# ============================================================================
# Complete 70B Model
# ============================================================================


class Model1B(nn.Module):
    """
    1B Dense Model — Test 14: Hybrid DeltaNet + GSA (DDDGDDDG), no fused CE.

    Configuration:
    - 1.513B total params, 1.513B active params (100% dense - no MoE)
    - 8 layers: 6 DeltaNet + 2 GSA (DDDGDDDG)
    - No experts (dense FFN 2048, Liger SwiGLU MLP); CE in train.py
    - 256k context length target

    ENHANCED WITH MEMORY STREAM RECURRENCE:
    - Enables processing infinite-length documents via chunking
    - Uses dedicated memory stream (stream 3) for cross-chunk continuity
    - Zero blocking: fully parallel forward pass
    - O(1) memory overhead per chunk

    TRAINING LOSS BALANCE (Empirically Tuned):
    ==========================================
    The forward() method returns (logits_ntp, logits_mtp, aux_loss).
    Training loop should compute total loss as:

        loss_ntp = CrossEntropy(logits_ntp, targets_t+1)
        loss_mtp = CrossEntropy(logits_mtp, targets_t+2)
        total_loss = loss_ntp + 0.3 * loss_mtp + aux_loss

    Rationale:
    - NTP (t+1) is primary task: weight = 1.0
    - MTP (t+2) is auxiliary teacher: weight = 0.3 (prevents aux dominance)
    - Aux loss: Minimal for 1B (dense, no MoE routing losses)

    Note: aux_loss will be near-zero for this model (no MoE routers in backbone)
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

        # Build hybrid layer stack: 75% DeltaNet + 25% GSA (DDDGDDDG)
        layers = []
        layer_types = []
        for i in range(config.num_layers):
            if (i + 1) % 4 == 0:
                layer_type = "gsa"
            else:
                layer_type = "deltanet"
            layers.append(LightningDecoderLayer(config, layer_type))
            layer_types.append(layer_type)

        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types

        # Reversible Midpoint Integration
        from .reversible_ops_midpoint import ReversibleMidpointStack

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

        # ============================================================================
        # Memory Stream Recurrence — "different" style (same as different_recurrence_model_1b_wo_rev.py)
        # Injects into embedding space (before stream expansion); reads from collapsed h_main.
        # (lambda_r, memory_ln, content-dependent memory_gate_proj.)
        # ============================================================================
        self.recurrence_stream_idx = 3  # Unused in "different" style; kept for compat
        self.lambda_r_raw = nn.Parameter(torch.tensor(-2.5))  # Initial strength ~0.078
        self.memory_ln = nn.LayerNorm(
            config.hidden_size
        )  # Normalize memory before injection
        # FIX #25: Content-dependent memory gating (prevents uniform broadcast shortcut learning)
        self.memory_gate_proj = nn.Linear(
            config.hidden_size, 1, bias=True
        )  # Per-token gate from content

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
            f"   - DeltaNet: {config.num_deltanet_layers} layers ({100*config.num_deltanet_layers//config.num_layers}%) - O(N) linear attention"
        )
        print(
            f"   - GSA: {config.num_gsa_layers} layers ({100*config.num_gsa_layers//config.num_layers}%) - Adaptive sparse"
        )
        print(f"\n   Context Target: {config.max_seq_len:,} tokens (standard RoPE)")
        print(
            f"   Dense FFN: {config.shared_expert_intermediate_size} intermediate (no MoE)"
        )
        print(
            f"   MTP: {config.mtp_num_predictions} predictions"
            if config.enable_mtp
            else "   MTP: Disabled"
        )
        print(f"\n   Total Parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
        print("   Active Parameters: ~1.513B (100% active, no MoE sparsity)")

    def _init_weights(self, module):
        # FIX #38: Skip initialization for kronecker_embeddings and all its submodules
        # (was using named_modules() which returns (name, module), not (name, param))
        if self.use_kronecker and self.kronecker_embeddings is not None:
            if module is self.kronecker_embeddings:
                return
            for submodule in self.kronecker_embeddings.modules():
                if module is submodule:
                    return

        if isinstance(module, (DenseMLP, MHCCoeffs)):
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
        self,
        input_ids,
        next_token_ids=None,
        attention_mask=None,
        prev_memory_stream=None,
        return_memory=True,
        return_loss=False,
        ntp_targets=None,
        mtp_targets=None,
        return_hidden=False,
    ):
        """
        Forward pass with Multi-Token Prediction.

        Args:
            input_ids: [B, T] - Input token IDs
            next_token_ids: [B, T] - Optional for MTP (t+1 tokens)
            attention_mask: Optional attention mask
            prev_memory_stream: [B, D] - Memory from previous chunk
            return_memory: Whether to return memory stream for next chunk
            return_loss: Whether to return auxiliary loss
            ntp_targets, mtp_targets: Ignored. CE is computed in train.py.
            return_hidden (bool): If True, skip lm_head and return hidden states
                [B, T, H] instead of logits [B, T, V]. Used by FusedLinearCE
                to avoid materialising the [B*T, vocab] tensor. Default: False.

        Returns:
            When return_hidden=False (inference, default):
                logits_ntp [B, T, V], logits_mtp [B, T, V] or None, + optional aux/memory
            When return_hidden=True (training with FusedLinearCE):
                h_ntp [B, T, H], h_mtp [B, T, H] or None, + optional aux_loss
        """
        batch_size, seq_len = input_ids.size()
        token_keep_mask = _token_keep_mask(
            attention_mask, batch_size, seq_len, input_ids.device
        )

        # Embeddings
        if self.use_kronecker:
            EMB = self.kronecker_embeddings(input_ids)  # type: ignore
            dtype_target = self.pf_to_model.weight.dtype  # type: ignore
            x = self.pf_to_model(EMB.to(dtype=dtype_target))  # type: ignore
            x = self.embed_norm(x)  # type: ignore
        else:
            x = self.token_embed(input_ids)  # type: ignore

        B, T, D = x.shape

        # ============================================================================
        # EMBEDDING-SPACE MEMORY INJECTION (before stream expansion) — "different" style
        # ============================================================================
        if prev_memory_stream is not None:
            prev_memory_stream = prev_memory_stream.detach()
            memory = self.memory_ln(prev_memory_stream)
            memory_gates = torch.sigmoid(self.memory_gate_proj(x))  # (B, T, 1)
            memory_broadcast = memory.unsqueeze(1).expand(B, T, D)
            lambda_r = F.softplus(self.lambda_r_raw)
            x = x + lambda_r * memory_gates * memory_broadcast

        # Expand to streams
        x_stream = torch.zeros(B, T, self.n_streams, D, device=x.device, dtype=x.dtype)
        x_stream[:, :, 0, :] = x

        # FIX #30: Precompute RoPE cos/sin once per forward (shared across all 8 layers)
        # FIX #32: Correct path through MHCSublayer wrapper (was layer.attn, now layer.attn_block.sublayer)
        # FIX #36: Include MTP block in RoPE cache optimization
        # FIX #39: Include dtype in cache key for mixed-precision safety
        # Set cache on all RotaryEmbedding instances - they'll check before computing
        cache_key = (T, x.device, x.dtype)
        for layer in self.layers:
            attn_mod = (
                layer.attn_block.sublayer  # type: ignore
            )  # Access through MHCSublayer wrapper # type: ignore
            if not hasattr(attn_mod.rotary_emb, "_forward_cache"):  # type: ignore
                attn_mod.rotary_emb._forward_cache = {}  # type: ignore
            if cache_key not in attn_mod.rotary_emb._forward_cache:  # type: ignore
                cos, sin = attn_mod.rotary_emb._compute_cos_sin(T, x.device, x.dtype)  # type: ignore
                attn_mod.rotary_emb._forward_cache[cache_key] = (cos, sin)  # type: ignore

        # Also cache for MTP block if enabled
        if self.mtp_block is not None:
            mtp_attn = self.mtp_block.attn_block.sublayer
            if not hasattr(mtp_attn.rotary_emb, "_forward_cache"):
                mtp_attn.rotary_emb._forward_cache = {}  # type: ignore
            if cache_key not in mtp_attn.rotary_emb._forward_cache:  # type: ignore
                cos, sin = mtp_attn.rotary_emb._compute_cos_sin(T, x.device, x.dtype)  # type: ignore
                mtp_attn.rotary_emb._forward_cache[cache_key] = (cos, sin)  # type: ignore

        # Pass through reversible stack
        x_stream, total_aux_loss = self.stack(x_stream, attention_mask=token_keep_mask)

        # Surface GSA leak metrics at model level for train.py logging/guards.
        leak_vals = []
        leak_attempt_vals = []
        for layer in self.layers:
            attn_mod = layer.attn_block.sublayer  # type: ignore
            leak_v = getattr(attn_mod, "last_gsa_leak_fraction", None)
            leak_attempt_v = getattr(attn_mod, "last_gsa_leak_attempt_fraction", None)
            if leak_v is not None:
                leak_vals.append(leak_v.detach().float())
            if leak_attempt_v is not None:
                leak_attempt_vals.append(leak_attempt_v.detach().float())
        self.last_gsa_leak_fraction = (
            torch.stack(leak_vals).mean()
            if leak_vals
            else x_stream.new_tensor(0.0, dtype=torch.float32)
        )
        self.last_gsa_leak_attempt_fraction = (
            torch.stack(leak_attempt_vals).mean()
            if leak_attempt_vals
            else x_stream.new_tensor(0.0, dtype=torch.float32)
        )

        # Collapse streams
        h_main = x_stream.mean(dim=2)
        h_main = self.norm(h_main)

        # ============================================================================
        # EXTRACT MEMORY from collapsed h_main (not stream-3) — "different" style
        # ============================================================================
        if return_memory:
            memory_stream_out = h_main[:, -1, :].detach()
        else:
            memory_stream_out = None

        # NTP Prediction
        # return_hidden=True: skip lm_head, return raw hidden states so train.py
        # can call FusedLinearCrossEntropyLoss without ever creating logit tensors.
        if return_hidden:
            logits_ntp = h_main  # [B, T, H] — NOT logits
        else:
            logits_ntp = self.lm_head(h_main)  # [B, T, V]

        # MTP Prediction
        logits_mtp = None
        if self.mtp_block is not None and next_token_ids is not None:
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_ids_use = next_token_ids[:, :min_len]

            if self.use_kronecker:
                next_emb = self.kronecker_embeddings(next_ids_use)  # type: ignore
                next_emb = self.pf_to_model(
                    next_emb.to(dtype=self.pf_to_model.weight.dtype)  # type: ignore
                )  # type: ignore
                next_emb = self.embed_norm(next_emb)  # type: ignore
            else:
                next_emb = self.token_embed(next_ids_use)  # type: ignore

            mtp_attention_mask = (
                token_keep_mask[:, :min_len] if token_keep_mask is not None else None
            )
            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=mtp_attention_mask)
            h_mtp_normed = self.norm(h_mtp)
            if return_hidden:
                logits_mtp = h_mtp_normed  # [B, T, H] — NOT logits
            else:
                logits_mtp = self.lm_head(h_mtp_normed)  # [B, T, V]

        # FIX #41: Clear RoPE forward-pass cache to prevent accumulation (CRITICAL PATH FIX)
        # Architecture: LightningDecoderLayer → MHCSublayer → GatedSparseAttention → RotaryEmbedding (all 8 layers GSA only)
        for layer in self.layers:
            if hasattr(layer.attn_block.sublayer, "rotary_emb"):  # type: ignore
                if hasattr(layer.attn_block.sublayer.rotary_emb, "_forward_cache"):  # type: ignore
                    layer.attn_block.sublayer.rotary_emb._forward_cache.clear()  # type: ignore

        # Also clear MTP block cache if enabled
        if self.mtp_block is not None:
            if hasattr(self.mtp_block.attn_block.sublayer, "rotary_emb"):
                if hasattr(
                    self.mtp_block.attn_block.sublayer.rotary_emb, "_forward_cache"
                ):
                    self.mtp_block.attn_block.sublayer.rotary_emb._forward_cache.clear()  # type: ignore

        if return_loss:
            if return_memory:
                return logits_ntp, logits_mtp, total_aux_loss, memory_stream_out
            else:
                return logits_ntp, logits_mtp, total_aux_loss
        if return_memory:
            return logits_ntp, logits_mtp, memory_stream_out
        else:
            return logits_ntp, logits_mtp


# ============================================================================
# torch.compile Setup
# ============================================================================


def enable_torch_compile(model, compile_mode="default"):
    """
    Prepare Model1B for torch.compile by:
    1. Switching Triton RMSNorm → PyTorch RMSNorm (compile fuses it with adjacent ops)
    2. Switching Triton Sinkhorn → PyTorch Sinkhorn (compile fuses iteration loop)
    3. Replacing profiler time_region with no-op (eliminates CUDA event graph breaks)
    4. Compiling each layer's force() method (the reversible-midpoint hot path)
    5. Compiling the MTP block forward

    Graph breaks after setup:
    - DeltaNet layers: 1 break (FLA chunk_gated_delta_rule — already optimized)
    - GSA layers: 2 breaks (fused_indexer_topk + triton_sparse_attention)
    - Everything else: fused by the compiler into large efficient kernels

    Args:
        model: Model1B instance (must be called BEFORE DeepSpeed wrapping)
        compile_mode: "default" or "max-autotune" (default is safer, max-autotune is
                      slower to compile but produces faster code)

    Returns:
        The same model (modified in-place), for chaining
    """
    global _TORCH_COMPILE_MODE, time_region

    _TORCH_COMPILE_MODE = True

    # 1. Replace time_region with no-op (profiler context managers cause graph breaks)
    from contextlib import contextmanager

    @contextmanager
    def _noop_region(name):
        yield

    time_region = _noop_region

    # 2. Keep Triton RMSNorm enabled — our kernel handles alignment correctly.
    #    The inductor's generated replacement kernel has alignment bugs during
    #    gradient checkpoint recomputation (misaligned address in fused reduction).
    n_triton_rms = sum(
        1 for m in model.modules() if isinstance(m, RMSNorm) and m._use_triton
    )

    print("\n  torch.compile setup:")
    print(
        f"    RMSNorm: keeping {n_triton_rms} Triton instances (our kernel is alignment-safe)"
    )
    print("    Sinkhorn: Triton bypassed in compile mode (compile will fuse)")
    print("    Profiler time_region: replaced with no-op")

    # 3. Switch bootstrap grad_checkpoint to use_reentrant=True
    #    use_reentrant=False tracks intermediate tensor saves, which conflicts with
    #    torch.compile's aot_autograd (90 vs 89 tensor count mismatch).
    #    use_reentrant=True just saves inputs and replays — compatible with compile.
    if hasattr(model, "stack") and hasattr(model.stack, "_compile_mode"):
        model.stack._compile_mode = True
        model.stack._sync_after_compile = True
        # Enable sync on all ForceWrappers (midpoint layers)
        for mid_layer in model.stack.mid_layers:
            if hasattr(mid_layer, "wrapper"):
                mid_layer.wrapper._sync_after_force = True
        print("    Bootstrap checkpoint: switched to use_reentrant=True")
        print("    Compiled function sync: enabled (sync after each force() call)")

    # 4. Set optimal torch settings for compiled training
    torch.set_float32_matmul_precision(
        "high"
    )  # TF32 for fp32 ops (RMSNorm variance, FLA internals)

    # 5. Compile each layer's force() method
    #    force() is the hot path: called ~4× per step per layer via reversible midpoint
    #    (2× forward evaluations + 2× backward recomputation via functional_call)
    n_compiled = 0
    for i, layer in enumerate(model.layers):
        layer.force = torch.compile(layer.force, mode=compile_mode, fullgraph=False)
        print(f"    Compiled layer {i} force() [{layer.layer_type}]")
        n_compiled += 1

    # 6. Compile MTP block if present
    if hasattr(model, "mtp_block") and model.mtp_block is not None:
        model.mtp_block = torch.compile(
            model.mtp_block, mode=compile_mode, fullgraph=False
        )
        print("    Compiled MTP block")

    print(f"    Compile mode: {compile_mode}")

    # 6. Immediate warmup — triggers JIT compilation NOW so inductor bugs
    #    crash here instead of after 10+ minutes of data loading.
    #    Requires model to be on CUDA already (call after model.to(device)).
    device = next(model.parameters()).device
    if device.type == "cuda":
        print("    Running compile warmup (BS=2, seq=64)...")
        _compile_warmup(model)
        print("    Warmup PASSED — all compiled graphs validated.\n")
    else:
        print(
            f"    Warmup SKIPPED — model on {device} (will compile on first training step).\n"
        )

    return model


def _compile_warmup(model):
    """Run a quick fwd+bwd with dummy data to trigger JIT compilation immediately.
    Model must be on CUDA before calling this."""
    device = next(model.parameters()).device
    cfg = model.config
    BS, SEQ = 2, 64
    VOCAB = cfg.vocab_size

    dummy_ids = torch.randint(0, VOCAB, (BS, SEQ), device=device)
    dummy_mask = torch.ones(BS, SEQ, dtype=torch.long, device=device)
    x_in = dummy_ids[:, :-2].contiguous()
    y_ntp = dummy_ids[:, 1:-1].contiguous()
    mask = dummy_mask[:, :-2].contiguous()

    was_training = model.training
    model.train()

    # Forward
    h_ntp, h_mtp, aux = model(
        x_in,
        next_token_ids=y_ntp,
        attention_mask=mask,
        return_loss=True,
        return_memory=False,
        prev_memory_stream=None,
        return_hidden=True,
    )

    # Backward (exercises MidpointFunction.backward + compiled force recomputation)
    loss = h_ntp.sum()
    if h_mtp is not None:
        loss = loss + h_mtp.sum()
    if aux is not None and aux.numel() > 0:
        loss = loss + aux.mean()
    loss.backward()

    torch.cuda.synchronize()

    # Cleanup
    model.zero_grad(set_to_none=True)
    del h_ntp, h_mtp, aux, loss, dummy_ids, dummy_mask, x_in, y_ntp, mask
    torch.cuda.empty_cache()

    if not was_training:
        model.eval()


# ============================================================================
# Factory Function
# ============================================================================


def create_model_1b(embedding_type="kronecker", bpe_vocab=None, pf_codec=None):
    """
    Create 1B model with default configuration.

    Test 14 is intended for Kronecker embeddings (default). Pass bpe_vocab and pf_codec
    when using embedding_type="kronecker".

    Args:
        embedding_type: "kronecker" (default, recommended) or "standard"
        bpe_vocab: Required for Kronecker embeddings (word list for Kronecker codec)
        pf_codec: Required for Kronecker embeddings (KroneckerEmbeddings instance)

    Returns:
        Model1B instance
    """
    config = ModelConfig()
    return Model1B(
        config, embedding_type=embedding_type, bpe_vocab=bpe_vocab, pf_codec=pf_codec
    )


if __name__ == "__main__":
    # Calculate actual metrics from weight_calculator.py
    from weight_calculator import LightningCalculator, LightningConfig  # type: ignore

    config_calc = LightningConfig(
        vocab_size=131072,
        hidden_size=4096,
        target_params=1e9,
        attention_type="gsa",
        deltanet_layer_ratio=0.75,  # Test 14: DDDGDDDG (6 DeltaNet, 2 GSA)
        num_routed_experts_active=0,
        num_shared_experts=0,
        expert_intermediate_size=1024,
        shared_expert_intermediate_size=2048,
        enable_mtp=True,
        mtp_num_predictions=2,
        num_experts_override=0,  # Dense only (no MoE)
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
    print("\nAttention (Test 14: DDDGDDDG — DeltaNet + GSA):")
    print(
        f"  DeltaNet: {config.num_deltanet_layers} layers ({100*config.num_deltanet_layers//config.num_layers}%) - O(N) linear attention"
    )
    print(
        f"  GSA: {config.num_gsa_layers} layers ({100*config.num_gsa_layers//config.num_layers}%) - Adaptive sparse attention"
    )
    print("\nModel Type: DENSE (No MoE) — Test 14")
    print(f"  Dense FFN intermediate: {config.shared_expert_intermediate_size}")
    print("\nEmbedding: Kronecker (intended for Test 14)")
    print(f"Context: {config.max_seq_len:,} tokens")
    print("=" * 80)
