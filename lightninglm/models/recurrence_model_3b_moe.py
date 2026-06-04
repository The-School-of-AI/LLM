"""
3B MoE Model (Corrected) -- 1B Kernel-Optimized Backbone + Null-Routed MoE FFN

Design goals:
- Keep the proven 1B attention/mHC/reversible/memory-stream implementation intact.
- Replace dense FFN with MoE FFN only (shared expert + routed experts).
- Preserve training contract used by train.py:
    total_loss = loss_ntp + 0.3 * loss_mtp + aux_loss

Configuration:
- 131,072 vocabulary (2^17)
- 4096 hidden size, 8 layers (6 DeltaNet + 2 GSA, DDDGDDDG)
- MoE FFN: 20 real experts + 20 null slots (top-k=2 over total 40 slots)
- Shared expert FFN width: 2048 (always active)
- Routed expert FFN width: 1024 (active only when selected)
- Multi-Token Prediction (MTP) with 2 predictions
- Multi-Head Composition (mHC) with 4 streams
- Reversible Midpoint Integration
- Target context: 256k

Notes:
- Null experts are routing slots, not compute modules.
- Routed compute is performed only for real expert assignments.
- All non-FFN kernel paths remain from recurrence_model_1b.py.
"""

import importlib
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Profiler (optional -- zero overhead when inactive) ────────────────────────
try:
    from ..profiler import time_region
except Exception:
    try:
        from lightninglm.utils.profiler import time_region
    except Exception:
        from contextlib import contextmanager

        @contextmanager
        def time_region(name: str):  # type: ignore[misc]
            yield


# ── Triton Kernel Imports ────────────────────────────────────────────────────
# Mirror the established multi-root kernel import resolution across launch contexts.
def _import_kernels_module():
    # Package-relative import when recurrence_model_1b.py is used inside lightninglm.models.
    try:
        from .. import kernels as kernels_module

        return kernels_module
    except Exception:
        pass

    # Standalone fallback: borrow kernels from known deepspeed_template roots.
    experiments_dir = Path(__file__).resolve().parents[5]
    candidate_roots = [
        experiments_dir
        / "9_training_stack_optimisation_and_cost_governor"
        / "training"
        / "deepspeed_template"
        / "src",
        experiments_dir
        / "9_training_stack_optimisation_and_cost_governor"
        / "training"
        / "deepspeed_template"
        / "dense_hardened"
        / "src",
    ]

    for root in candidate_roots:
        if not (root / "kernels").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            return importlib.import_module("kernels")
        except Exception:
            continue

    return None


_kernels_module = _import_kernels_module()
if _kernels_module is not None:
    HAS_TRITON = bool(getattr(_kernels_module, "HAS_TRITON", False))
    HAS_MOE_GROUPED_GEMM = bool(getattr(_kernels_module, "HAS_MOE_GROUPED_GEMM", False))
    triton_sparse_attention = getattr(_kernels_module, "triton_sparse_attention", None)
    pytorch_sparse_attention = getattr(
        _kernels_module, "pytorch_sparse_attention", None
    )
    triton_sinkhorn_knopp = getattr(_kernels_module, "triton_sinkhorn_knopp", None)
    pytorch_sinkhorn_knopp = getattr(_kernels_module, "pytorch_sinkhorn_knopp", None)
    triton_rmsnorm = getattr(_kernels_module, "triton_rmsnorm", None)
    triton_rmsnorm_fwd_only = getattr(_kernels_module, "triton_rmsnorm_fwd_only", None)
    pytorch_rmsnorm = getattr(_kernels_module, "pytorch_rmsnorm", None)
    TritonRMSNorm = getattr(_kernels_module, "TritonRMSNorm", None)
    fused_indexer_topk = getattr(_kernels_module, "fused_indexer_topk", None)
    fused_delta_entrance = getattr(_kernels_module, "fused_delta_entrance", None)
    triton_deltanet_post_fused = getattr(
        _kernels_module, "triton_deltanet_post_fused", None
    )
    moe_grouped_gemm = getattr(_kernels_module, "moe_grouped_gemm", None)
    # Phase 2 fused kernels
    fused_mhc_collapse = getattr(_kernels_module, "fused_mhc_collapse", None)
    fused_mhc_expand_residual = getattr(
        _kernels_module, "fused_mhc_expand_residual", None
    )
    fused_qk_rope = getattr(_kernels_module, "fused_qk_rope", None)
    fused_sigmoid_gate = getattr(_kernels_module, "fused_sigmoid_gate", None)
    fused_scaled_sigmoid = getattr(_kernels_module, "fused_scaled_sigmoid", None)
    fused_beta_gk_proj_triton = getattr(
        _kernels_module, "fused_beta_gk_proj_triton", None
    )
    # Phase 3 MoE expert kernels
    triton_grouped_gemm = getattr(_kernels_module, "triton_grouped_gemm", None)
    fused_moe_gate_up_silu = getattr(_kernels_module, "fused_moe_gate_up_silu", None)
    fused_weighted_scatter_add = getattr(
        _kernels_module, "fused_weighted_scatter_add", None
    )
else:
    HAS_TRITON = False
    HAS_MOE_GROUPED_GEMM = False
    triton_sparse_attention = None
    pytorch_sparse_attention = None
    triton_sinkhorn_knopp = None
    pytorch_sinkhorn_knopp = None
    triton_rmsnorm = None
    triton_rmsnorm_fwd_only = None
    pytorch_rmsnorm = None
    TritonRMSNorm = None
    fused_indexer_topk = None
    fused_delta_entrance = None
    triton_deltanet_post_fused = None
    moe_grouped_gemm = None
    fused_mhc_collapse = None
    fused_mhc_expand_residual = None
    fused_qk_rope = None
    fused_sigmoid_gate = None
    fused_scaled_sigmoid = None
    fused_beta_gk_proj_triton = None
    triton_grouped_gemm = None
    fused_moe_gate_up_silu = None
    fused_weighted_scatter_add = None

HAS_FUSED_INDEXER = fused_indexer_topk is not None

# DeltaNet fused backend (required): flash-linear-attention
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAS_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    HAS_FLA = False


# ── Liger ops (RoPE + MLP helpers, no fused CE here) ────────────────────────
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
_kernel_log = logging.getLogger("recurrance_model_3b_moe_correct.kernels")
if not _kernel_log.handlers:
    _kernel_log.addHandler(logging.StreamHandler())
    _kernel_log.setLevel(logging.INFO)

_cuda_available = torch.cuda.is_available()
_kernel_log.info("=" * 60)
_kernel_log.info("Kernel Availability Report (3B MoE Correct):")
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
    f"  fla GatedDeltaRule:   {'ENABLED' if HAS_FLA and chunk_gated_delta_rule is not None and _cuda_available else 'UNAVAILABLE (pip install fla)'}"
)
_kernel_log.info(
    f"  MoE Grouped GEMM:     {'ENABLED' if HAS_MOE_GROUPED_GEMM and moe_grouped_gemm is not None and _cuda_available else 'FALLBACK (Vectorized)'}"
)
_kernel_log.info(
    f"  Triton Grouped GEMM:  {'ENABLED' if triton_grouped_gemm is not None else 'DISABLED'}"
)
_kernel_log.info(
    f"  Fused Gate+Up+SiLU:   {'ENABLED' if fused_moe_gate_up_silu is not None else 'DISABLED'}"
)
_kernel_log.info(
    f"  Fused Weighted Scatter: {'ENABLED' if fused_weighted_scatter_add is not None else 'DISABLED'}"
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
    # FAST PATH: [B, T] bool already normalized by collator
    if mask.dim() == 2 and mask.dtype == torch.bool:
        return mask

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
    - Process: str -> UTF-8 bytes -> each byte (0-255) is a token
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
    - Encoding: str -> UTF-8 bytes -> Kronecker embeddings
    - Each byte (0-255) is treated as a valid symbol
    - Decoding: bytes -> UTF-8 decode -> str
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
        1. Convert str -> UTF-8 bytes
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
        4. Collect bytes -> decode UTF-8 -> str

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
    """3B MoE model configuration with 1B-tested backbone defaults."""

    # Architecture
    vocab_size = 131072  # 2^17
    hidden_size = 4096
    num_layers = 8

    # Attention Mix (75% DeltaNet / 25% GSA) -- DDDGDDDG pattern
    num_deltanet_layers = 6
    num_gsa_layers = 2

    # DeltaNet Configuration
    delta_v_heads = 32  # hidden_size / delta_head_dim = 4096 / 128
    delta_head_dim = 128
    delta_gate_dim = 384  # 9.4% of hidden_size

    # GSA Configuration
    gsa_num_heads = 16  # hidden_size / attn_head_dim = 4096 / 256
    gsa_head_dim = 256
    gsa_k_base = 128  # FIX-PERF-02: Reduced from 512 -- at T=4096, 512 keys/query = 25% dense; 128 is sufficient
    gsa_k_min = 32
    gsa_k_max = 256  # FIX-PERF-02: Reduced from 1024 -- limits atomic scatter in dK/dV backward kernel
    gsa_indexer_heads = 4

    # MoE Configuration
    num_real_experts = 20
    num_null_experts = 20
    total_expert_slots = 40
    top_k = 2  # Select top-k over total slots (real + null)
    expert_intermediate_size = 1024  # Routed experts
    shared_expert_intermediate_size = 2048  # Shared expert (always active)
    data_sparsity = 0.5  # Target null-selection rate for router aux regularizer

    # MoE backend policy (fixed T2+T3 path for 3B MoE):
    # - grouped_gemm is the default expert compute path when available
    # - T3 router/permute/scatter optimizations are always enabled in MoEFFN
    # - T4 dispatcher is optional (configurable from YAML)
    moe_backend = "grouped_gemm"
    require_fused_moe_kernel = True
    allow_moe_vectorized_fallback = False
    track_moe_last_indices = False
    moe_t4_enabled = False
    moe_t4_dispatcher = "deepep"  # for future multi-GPU integration
    # Expert-parallel control (requires a real dispatcher backend).
    moe_expert_parallel_size = 1

    # MTP Configuration
    enable_mtp = True
    mtp_num_predictions = 2
    mtp_reversible = True
    mtp_step_size = 0.25
    mtp_a = 0.5
    mtp_bootstrap = "euler"

    # mHC Configuration
    n_streams = 4
    sinkhorn_iters = (
        20  # Keep at 20 -- sufficient for mHC routing quality; do not reduce
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
    Pure Kronecker Product Embedding — gpu_dynamic mode.

    Stores compact byte buffers (~4.5 MB) instead of a full PF_table (~2.15 GB)
    and computes Kronecker product vectors on-the-fly via scatter_add.

    Overhead: ~1-4ms per micro-batch (0.01-0.24% of step time).
    Memory saved: 2.14 GB per GPU (17.1 GB across 8 GPUs).

    Configuration:
    - POS_DIM=32: Handles tokens up to 32 characters
    - CHAR_DIM=256: Full ASCII + extended character set
    - D=8192: Total embedding dimension (32 × 256)

    Note: Embedding tying NOT possible (D=8192 != hidden_size=4096)
    """

    def __init__(self, vocab_words: List[str], pf_codec: KroneckerEmbeddings):
        super().__init__()
        self._pos_dim = pf_codec.POS_DIM
        self._D = pf_codec.D
        self._length_normalize = pf_codec.cfg.length_normalize
        self._chunk_tokens = 2048

        # Compact byte buffers: [V, POS_DIM] uint8 + [V] int16  (~4.5 MB)
        # vs full PF_table: [V, D] bf16  (~2.15 GB)
        token_bytes = np.zeros((len(vocab_words), self._pos_dim), dtype=np.uint8)
        token_lens = np.zeros((len(vocab_words),), dtype=np.int16)
        for i, word in enumerate(vocab_words):
            if not word:
                continue
            byte_seq = word.encode("utf-8")
            if len(byte_seq) > self._pos_dim:
                if pf_codec.cfg.truncate_long_words:
                    byte_seq = pf_codec._utf8_safe_truncate(byte_seq, self._pos_dim)
                else:
                    byte_seq = byte_seq[: self._pos_dim]
            L = len(byte_seq)
            if L == 0:
                continue
            token_bytes[i, :L] = np.frombuffer(byte_seq, dtype=np.uint8, count=L)
            token_lens[i] = L

        self.register_buffer(
            "_token_bytes", torch.from_numpy(token_bytes), persistent=False
        )
        self.register_buffer(
            "_token_lens", torch.from_numpy(token_lens), persistent=False
        )
        self.register_buffer(
            "_pos_ids", torch.arange(self._pos_dim, dtype=torch.long), persistent=False
        )

    @property
    def D(self):
        return self._D

    def forward(self, token_ids):
        """
        Forward pass: compute Kronecker embeddings on-the-fly from byte buffers.

        Args:
            token_ids: Token indices (B, T)

        Returns:
            Normalized embeddings (B, T, D=8192)
        """
        with time_region("embed.kronecker.dynamic"):
            # Lazy device move (deepspeed.zero.Init may leave buffers on CPU)
            if self._token_bytes.device != token_ids.device:
                self._token_bytes = self._token_bytes.to(token_ids.device)
                self._token_lens = self._token_lens.to(token_ids.device)
                self._pos_ids = self._pos_ids.to(token_ids.device)

            flat_ids = token_ids.reshape(-1)
            total = flat_ids.numel()
            device = token_ids.device

            bytes_all = self._token_bytes.index_select(0, flat_ids).to(torch.long)
            lens_all = self._token_lens.index_select(0, flat_ids).to(torch.long)

            pf = torch.zeros((total, self._D), device=device, dtype=torch.float32)
            pos = self._pos_ids.unsqueeze(0).expand(total, -1)
            lin_idx = bytes_all * self._pos_dim + pos

            valid = pos < lens_all.unsqueeze(1)

            if self._length_normalize:
                scales = torch.rsqrt(lens_all.clamp_min(1).to(torch.float32))
                src = valid.to(torch.float32) * scales.unsqueeze(1)
            else:
                src = valid.to(torch.float32)

            pf.scatter_add_(dim=1, index=lin_idx, src=src)

            pf_centered = pf - pf.mean(dim=-1, keepdim=True)
            pf_std = pf_centered.std(dim=-1, keepdim=True) + 1e-6
            out = (pf_centered / pf_std).to(torch.bfloat16)

            return out.view(*token_ids.shape, self._D)

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
        # Safe with grad enabled -- reversible backward uses this during recompute.
        if self._use_triton and x.is_cuda:
            try:
                return triton_rmsnorm(x, self.weight, self.eps)
            except Exception:
                pass  # Fall through to PyTorch path

        # PyTorch fallback (FIX #43: fp32 variance for stability)
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm.to(x.dtype) + self.eps)
        # REPRO-FIX: Cast weight to x.dtype to avoid implicit fp32 upcast in reversible recompute
        return self.weight.to(dtype=x.dtype) * x


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
        # FIX #39: Include dtype in cache key
        #   Include self.dim and return pre-broadcasted views if available
        cache_key = (seq_len, device, dtype, self.dim)
        if hasattr(self, "_forward_cache") and cache_key in self._forward_cache:
            return self._forward_cache[cache_key]

        t = torch.arange(seq_len, device=device).float()
        freqs = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        # BUG-FIX: Use repeat_interleave for interleaved RoPE flavor.
        # cat((f, f)) would result in [f0, f1, f2, f0, f1, f2] which, when sliced via [0::2],
        # would drop all odd frequencies and repeat even ones.
        # repeat_interleave results in [f0, f0, f1, f1, f2, f2] so [0::2] -> [f0, f1, f2].
        emb = torch.repeat_interleave(freqs, 2, dim=-1)

        # FIX #42: Cast to requested dtype to match query/key precision
        # Prevents implicit upcasts and memory/bandwidth issues at 256k context
        cos_out = emb.cos()
        sin_out = emb.sin()
        if dtype is not None:
            cos_out = cos_out.to(dtype)
            sin_out = sin_out.to(dtype)

        # Return (cos, sin, cos_broadcast, sin_broadcast)
        # Default broadcast if not already in cache
        return (
            cos_out,
            sin_out,
            cos_out.unsqueeze(0).unsqueeze(2),
            sin_out.unsqueeze(0).unsqueeze(2),
        )

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

    References used for fused-entrance integration:
    - FLA GatedDeltaNet layer:
      https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
    - FLA gated delta-rule kernels:
      https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
    - NVLabs GatedDeltaNet FLA kernels:
      https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py
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
        # Keep only the post-FLA fusion optimization path.
        self.fuse_post_tail = True
        # Default on: pass native dtype tensors into FLA (bf16 path).
        # Set T17_DN_FLA_NATIVE_DTYPE=0 to force legacy fp32-cast behavior.
        self.fla_native_dtype = os.getenv("T17_DN_FLA_NATIVE_DTYPE", "1") == "1"
        # Optional: use Triton fused Delta Entrance (conv+mask+norm+RoPE).
        # Keep off by default until fully validated on train + eval.
        self.use_fused_delta_entrance = (
            os.getenv("T17_DN_USE_DELTA_ENTRANCE", "0") == "1"
        )

        key_dim = num_heads * head_dim
        value_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.o_proj = nn.Linear(value_dim, hidden_size, bias=False)

        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.gk_proj = nn.Linear(hidden_size, num_heads, bias=True)

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
        proj_modules = [self.o_proj, self.q_proj, self.k_proj, self.v_proj, self.g_proj]
        for m in proj_modules:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # STABILITY-FIX: Reduce init scale for alpha/beta gating logic
        gating_modules = [self.b_proj, self.gk_proj]
        for m in gating_modules:
            nn.init.normal_(m.weight, mean=0.0, std=0.002)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device
        token_keep = _token_keep_mask(attention_mask, B, T, device)
        x = x.to(dtype=self.q_proj.weight.dtype)
        token_keep_f = None

        with time_region("deltanet.entry.proj"):
            q_in = self.q_proj(x)
            k_in = self.k_proj(x)
            v_in = self.v_proj(x)
            g = self.g_proj(x)

        use_fused_entry = (
            self.use_fused_delta_entrance
            and fused_delta_entrance is not None
            and q_in.is_cuda
        )
        if use_fused_entry:
            # Reference behavior from FLA/NVLabs gated DeltaNet paths:
            # short-conv + qk normalization + kernel-compatible inputs.
            with time_region("deltanet.entry.fused"):
                cos, sin, _cos_b, _sin_b = self.rotary_emb._compute_cos_sin(
                    T, device, x.dtype
                )
                cos_half = cos[:, 0::2].contiguous()
                sin_half = sin[:, 0::2].contiguous()
                q, k, v = fused_delta_entrance(
                    q_in,
                    k_in,
                    v_in,
                    self.q_conv1d.conv.weight,
                    self.k_conv1d.conv.weight,
                    self.v_conv1d.conv.weight,
                    self.q_conv1d.conv.bias,
                    self.k_conv1d.conv.bias,
                    self.v_conv1d.conv.bias,
                    cos_half,
                    sin_half,
                    None,
                )
                g = g.view(B, T, self.num_heads, self.head_dim)
        else:
            # ── Separate PyTorch ops (conv1d → L2Norm → RoPE → mask) ────────────
            with time_region("deltanet.entry.conv"):
                q = self.q_conv1d(q_in)
                k = self.k_conv1d(k_in)
                v = self.v_conv1d(v_in)

            with time_region("deltanet.entry.reshape"):
                q = q.view(B, T, self.num_heads, self.head_dim)
                k = k.view(B, T, self.num_heads, self.head_dim)
                v = v.view(B, T, self.num_heads, self.head_dim)
                g = g.view(B, T, self.num_heads, self.head_dim)

            # L2 Normalization MUST happen first, otherwise it destroys the RoPE rotational structure
            with time_region("deltanet.entry.norm_rope"):
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)

                cos, sin, _cos_b, _sin_b = self.rotary_emb._compute_cos_sin(
                    T, device, x.dtype
                )
                if fused_qk_rope is not None and q.is_cuda and q.is_contiguous():
                    cos_half = cos[:, ::2].contiguous()
                    sin_half = sin[:, ::2].contiguous()
                    q, k = fused_qk_rope(q, k, cos_half, sin_half)
                else:
                    q = self.rotary_emb._apply_rotary(q, _cos_b, _sin_b)
                    k = self.rotary_emb._apply_rotary(k, _cos_b, _sin_b)

        with time_region("deltanet.entry.alpha_beta"):
            if fused_beta_gk_proj_triton is not None and x.is_cuda:
                w_cat = torch.cat([self.b_proj.weight, self.gk_proj.weight], dim=0)
                b_cat = torch.cat([self.b_proj.bias, self.gk_proj.bias], dim=0)
                beta, gk = fused_beta_gk_proj_triton(x, w_cat, b_cat, self.num_heads)
                beta = beta.unsqueeze(-1)  # sigmoid already applied
            else:
                beta = torch.sigmoid(self.b_proj(x)).unsqueeze(
                    -1
                )  # (B, T, num_heads, 1)
                gk = self.gk_proj(x)
            A = torch.exp(self.A_log)
            alpha = torch.exp(
                -A.view(1, 1, self.num_heads, 1)
                * F.softplus(gk + self.dt_bias).unsqueeze(-1)
            )

        if token_keep is not None:
            token_keep_f = token_keep.to(dtype=q.dtype).view(B, T, 1, 1)
            q = q * token_keep_f
            k = k * token_keep_f
            v = v * token_keep_f
            g = g * token_keep_f
            beta = beta * token_keep_f
            alpha = alpha * token_keep_f + (1.0 - token_keep_f)

        fla_available = HAS_FLA and chunk_gated_delta_rule is not None and q.is_cuda
        if not fla_available:
            raise RuntimeError(
                "DeltaNet fused kernel is required but unavailable. "
                f"HAS_FLA={HAS_FLA}, chunk_gated_delta_rule={chunk_gated_delta_rule is not None}, "
                f"q.is_cuda={q.is_cuda}. Install fla: pip install fla"
            )

        # Direct fla kernel call (no local wrapper).
        # q/k/v: [B, T, H, d], alpha/beta: [B, T, H, 1]
        if self.fla_native_dtype:
            q_fla = q
            k_fla = k
            v_fla = v
            g_fla = torch.log(alpha[:, :, :, 0].clamp(min=1e-6))
            beta_fla = beta[:, :, :, 0]
        else:
            q_fla = q.float()
            k_fla = k.float()
            v_fla = v.float()
            g_fla = torch.log(alpha[:, :, :, 0].float().clamp(min=1e-6))
            beta_fla = beta[:, :, :, 0].float()

        with time_region("deltanet.fla"):
            o_fla, _ = chunk_gated_delta_rule(
                q_fla,
                k_fla,
                v_fla,
                g_fla,
                beta_fla,
                scale=1.0,
                output_final_state=False,
            )

        use_fused_post = (
            self.use_output_norm
            and self.fuse_post_tail
            and triton_deltanet_post_fused is not None
            and q.is_cuda
        )
        if use_fused_post:
            with time_region("deltanet.post_fused"):
                o = triton_deltanet_post_fused(
                    o_fla.to(q.dtype),
                    q,
                    k,
                    v,
                    g,
                    self.D,
                    self.o_norm.norm.weight,
                    eps=self.o_norm.norm.eps,
                )
        else:
            # Preserve Test17 residual behavior.
            qk_dot = (q * k).sum(dim=-1, keepdim=True)
            d_residual = self.D.view(1, 1, self.num_heads, 1) * qk_dot * v
            o = o_fla.to(q.dtype) + d_residual

            if self.use_output_norm:
                o_flat = o.reshape(B * T * self.num_heads, self.head_dim)
                g_flat = g.reshape(B * T * self.num_heads, self.head_dim)
                o_normed = self.o_norm(o_flat, g_flat)
                o = o_normed.view(B, T, self.num_heads, self.head_dim)
            else:
                if fused_sigmoid_gate is not None and o.is_cuda:
                    o = fused_sigmoid_gate(o, g)
                else:
                    o = o * torch.sigmoid(g)

        if token_keep_f is not None:
            o = o * token_keep_f

        o = o.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o)


# ============================================================================
# Gated Sparse Attention (25% of layers -- DDDGDDDG)
# ============================================================================


class GatedSparseAttention(nn.Module):
    """
    Memory complexity: O(T*k) via fused_indexer_topk chunked kernel.
    Architecture:
    - Shared indexer keys (W_Ik -> [B, T, d_idx]) across indexer heads
    - Per-attention-head diversity via head_importance_bias on attention logits
    - Adaptive sparsity budget k_t from variance-based heuristic
    -   GSA Triton sparse attention kernel is REQUIRED for training throughput.
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
        self.W_Iq = nn.Linear(hidden_size, indexer_heads * self.d_idx, bias=False)
        self.W_Ik = nn.Linear(
            hidden_size, self.d_idx, bias=False
        )  # Shared across indexer heads
        self.W_Iw = nn.Linear(hidden_size, indexer_heads, bias=False)
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

        # Reversibility Cache (single slot avoids Python list churn)
        self._cached_base_idx = None
        self._cached_keep_mask = None
        self._q_pos_cache = None

        # Attention Projections
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dual Gating
        self.W_gv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_go = nn.Linear(hidden_size, hidden_size, bias=False)

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
        token_keep = _token_keep_mask(attention_mask, B, T, device)
        x = x.to(dtype=self.W_q.weight.dtype)

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

        if is_reconstruct and self._cached_base_idx is not None:
            base_idx, keep_mask = self._cached_base_idx, self._cached_keep_mask
            k_limit = base_idx.size(-1)
            # Clear slot after use
            self._cached_base_idx = None
            self._cached_keep_mask = None
        else:
            # Lightning Indexer -- O(T*k) via fused chunked kernel
            # Uses fused_indexer_topk to avoid materializing [B, heads, T, T] importance scores.
            q_I = self.W_Iq(x).view(
                B, T, self.indexer_heads, self.d_idx
            )  # [B, T, 4, 128]
            k_I = self.W_Ik(x)  # [B, T, d_idx]
            w_raw = self.W_Iw(x)  # [B, T, indexer_heads]
            scale_idx = 1.0 / math.sqrt(self.d_idx)

            if is_reversible_forward:
                self._variance_ema_snapshot.copy_(self.variance_ema)
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
                var_t, k_t, top_indices = fused_indexer_topk(
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
                # Local EMA only (kill 2 global syncs per step)
                self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)

            # Build per-query keep mask from adaptive k_t with strict causal safety.
            k_limit = top_indices.size(-1)
            #   Use int32 for indices (saves 50% bandwidth)
            base_idx = top_indices.to(torch.int32)

            # Position cache (avoids 16 arange() calls per step across layers)
            if (
                self._q_pos_cache is None
                or self._q_pos_cache.size(0) != T
                or self._q_pos_cache.device != device
            ):
                self._q_pos_cache = torch.arange(T, device=device, dtype=torch.int32)
            q_pos = self._q_pos_cache.view(1, T, 1)
            causal_cap = (q_pos + 1).to(dtype=k_t.dtype)
            k_t = torch.minimum(k_t, causal_cap.squeeze(-1)).clamp(min=1)

            range_k = torch.arange(k_limit, device=device, dtype=torch.int32)
            keep_mask = range_k.view(1, 1, -1) < k_t.unsqueeze(-1)  # [B, T, k_limit]
            causal_selected = base_idx <= q_pos

            if token_keep is not None:
                query_keep = token_keep.unsqueeze(-1)
                invalid_query = ~token_keep
                if invalid_query.any():
                    fallback_idx = (
                        torch.arange(T, device=device, dtype=torch.int32)
                        .view(1, T)
                        .expand(B, T)
                    )
                    base_idx = base_idx.clone()
                    base_idx[..., 0] = torch.where(
                        invalid_query, fallback_idx, base_idx[..., 0]
                    )
                    causal_selected = base_idx <= q_pos

                key_keep = torch.gather(
                    token_keep, dim=1, index=base_idx.reshape(B, -1).long()
                ).view(B, T, k_limit)
                keep_mask = keep_mask & key_keep & query_keep

                # Keep at least one index for masked queries to avoid empty-kernel rows.
                if invalid_query.any():
                    keep_mask = keep_mask.clone()
                    keep_mask[..., 0] = keep_mask[..., 0] | invalid_query

            # Strict causal enforcement
            keep_mask = keep_mask & causal_selected

            if is_reversible_forward:
                self._cached_base_idx = base_idx
                self._cached_keep_mask = keep_mask

        # Dual Gating & Attention Projections
        q = self.W_q(x)
        k_attn = self.W_k(x)
        v = self.W_v(x)

        if fused_sigmoid_gate is not None and v.is_cuda:
            v = fused_sigmoid_gate(v, self.W_gv(x))
        else:
            g_v = torch.sigmoid(self.W_gv(x))
            v = v * g_v

        q = q.view(B, T, self.num_heads, self.head_dim)
        k_attn = k_attn.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        if token_keep is not None:
            token_keep_v = token_keep.view(B, T, 1, 1).to(q.dtype)
            q = q * token_keep_v
            k_attn = k_attn * token_keep_v
            v = v * token_keep_v

        # Rotary (computed on-the-fly to save 2.1GB VRAM)
        # Uses cache key (T, device, dtype, head_dim) - Safe for centralized sharing
        #   Consumes pre-broadcasted RoPE from centralized cache
        cos, sin, _cos_b, _sin_b = self.rotary_emb._compute_cos_sin(T, device, x.dtype)
        if fused_qk_rope is not None and q.is_cuda and q.is_contiguous():
            cos_half = cos[:, ::2].contiguous()
            sin_half = sin[:, ::2].contiguous()
            q, k_attn = fused_qk_rope(q, k_attn, cos_half, sin_half)
        else:
            q = self.rotary_emb._apply_rotary(q, _cos_b, _sin_b)
            k_attn = self.rotary_emb._apply_rotary(k_attn, _cos_b, _sin_b)

        # ── Sparse attention via triton_sparse_attention kernel ────────
        # O(T*k) complexity: kernel iterates only over k_limit selected
        # keys per query using online softmax. No T×T tensor ever created.
        # Memory: O(B*H*T*k_limit) for indices/mask, NOT O(T²).
        #
        # At T=256k, B=1, k_limit=1024:
        #   indices: [1, 16, 256k, 1024] int64 = 32GB  (vs 128GB for [B,1,T,T])
        #   BUT indices are shared across heads -> [B, 1, T, k_limit] expanded
        #   as views, so actual memory = [B, T, k_limit] * 8 bytes = 2GB.

        # Kernel expects indices: [B, H, T, k_sel] int64, mask: [B, H, T, k_sel] float32
        # base_idx is [B, T, k_limit], keep_mask is [B, T, k_limit] bool
        # Expand to [B, H, T, k_limit] as views (stride=0 on H dim).
        # Triton kernel uses stride-based access, so zero-stride broadcast works
        # without copying. Memory: only [B, T, k_limit] actually allocated.
        sparse_idx = base_idx.unsqueeze(1).expand(B, self.num_heads, T, k_limit)
        sparse_mask = (
            keep_mask.to(dtype=q.dtype)
            .unsqueeze(1)
            .expand(B, self.num_heads, T, k_limit)
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

        # Output gate
        if fused_sigmoid_gate is not None and o_sparse.is_cuda:
            gated_out = fused_sigmoid_gate(o_sparse, self.W_go(x))
        else:
            g_o = torch.sigmoid(self.W_go(x))
            gated_out = o_sparse * g_o

        return self.o_proj(gated_out)


# ============================================================================
# MoE FFN (null-routed) -- shared expert + routed experts
# ============================================================================


class MoEGate(nn.Module):
    """
    Router gate with null slots.

    Routing behavior:
    - Compute probabilities over [real experts + null slots].
    - Select top-k over total slots.
    - Zero and renormalize weights for null selections, so routed compute uses
      only real experts while preserving top-k null-routing semantics.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        data_sparsity: float = 0.5,
        num_null_experts: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity
        self.rho = data_sparsity

        # Prefer explicit null-slot count from config for 20-real / 20-null setups.
        if num_null_experts is not None:
            self.num_null_copies = int(num_null_experts)
        else:
            self.num_null_copies = int(
                num_experts * (1 - data_sparsity) / data_sparsity
            )
        self.total_slots = num_experts + self.num_null_copies

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.gate.weight.data.normal_(mean=0.0, std=0.02)
        # T3 fixed path: top-k on logits + local softmax over selected logits.
        self.router_fusion_enabled = True
        # Reversible recompute cache for deterministic top-k reconstruction.
        self._cached_topk_idx = None

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        is_reconstruct = self.training and torch.is_grad_enabled()

        real_logits = self.gate(x) + self.logit_bias
        null_logits = (
            self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)
        )
        logits = torch.cat([real_logits, null_logits], dim=-1)

        if self.router_fusion_enabled:
            cached_ok = (
                self._cached_topk_idx is not None
                and self._cached_topk_idx.shape[0] == B
                and self._cached_topk_idx.shape[1] == T
            )
            if is_reconstruct and cached_ok:
                topk_idx = self._cached_topk_idx
                self._cached_topk_idx = None
                topk_logits = torch.gather(logits, dim=-1, index=topk_idx.long())
            else:
                if is_reconstruct and self._cached_topk_idx is not None:
                    self._cached_topk_idx = None
                topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
                if is_reversible_forward:
                    self._cached_topk_idx = topk_idx
            topk_weight = F.softmax(topk_logits, dim=-1)
        else:
            cached_ok = (
                self._cached_topk_idx is not None
                and self._cached_topk_idx.shape[0] == B
                and self._cached_topk_idx.shape[1] == T
            )
            if is_reconstruct and cached_ok:
                topk_idx = self._cached_topk_idx
                self._cached_topk_idx = None
            else:
                if is_reconstruct and self._cached_topk_idx is not None:
                    self._cached_topk_idx = None
                probs = F.softmax(logits, dim=-1)
                _, topk_idx = torch.topk(probs, self.top_k, dim=-1)
                if is_reversible_forward:
                    self._cached_topk_idx = topk_idx
            selected_logits = torch.gather(logits, dim=-1, index=topk_idx.long())
            topk_weight = F.softmax(selected_logits, dim=-1)

        is_null = topk_idx >= self.num_experts
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # Balance over real experts only + null-rate regularizer.
        logits_real = logits[:, :, : self.num_experts]
        probs_real = F.softmax(logits_real, dim=-1)
        p_real = probs_real.mean(dim=(0, 1))

        idx_flat = topk_idx.view(-1)
        is_null_flat = idx_flat >= self.num_experts
        idx_real = torch.where(is_null_flat, torch.zeros_like(idx_flat), idx_flat)
        counts_real = torch.bincount(idx_real, minlength=self.num_experts).float()
        counts_real[0] -= is_null_flat.sum().float()
        total_real_assignments = counts_real.sum()
        f_real = counts_real / total_real_assignments.clamp(min=1e-6)
        l_bal = self.num_experts * torch.sum(f_real * p_real)

        lse = torch.logsumexp(logits, dim=-1)
        l_z = (lse**2).mean()

        null_rate = is_null.float().mean()
        l_null = (null_rate - self.rho) ** 2

        # Store routing stats for logging (detached, no gradient impact).
        self._last_null_rate = null_rate.detach()
        self._last_expert_counts = counts_real.detach()

        aux_loss = 2e-2 * l_bal + 1e-3 * l_z + 1e-2 * l_null
        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN with one shared expert and null-routed sparse experts.

    Widths:
    - routed experts: d_hidden (1024 for this 3B config)
    - shared expert: d_shared_hidden (2048 for this 3B config)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_shared_hidden: Optional[int] = None,
        num_experts: int = 20,
        num_null_experts: Optional[int] = None,
        top_k: int = 2,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
        moe_backend: str = "auto",
        require_fused_kernel: bool = False,
        allow_vectorized_fallback: bool = False,
        track_last_indices: bool = False,
        expert_parallel_size: int = 1,
        t4_enabled: bool = False,
        t4_dispatcher: str = "deepep",
    ):
        super().__init__()
        if d_shared_hidden is None:
            d_shared_hidden = d_hidden

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_shared_hidden = d_shared_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        self.require_fused_kernel = require_fused_kernel
        self.allow_vectorized_fallback = allow_vectorized_fallback
        self.track_last_indices = track_last_indices
        self.expert_parallel_size = int(max(1, expert_parallel_size))

        self.gate = MoEGate(
            d_model,
            num_experts,
            top_k,
            data_sparsity=data_sparsity,
            num_null_experts=num_null_experts,
        )

        # Routed real experts.
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared expert (always active).
        self.shared_gate = nn.Linear(d_model, d_shared_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, d_shared_hidden, bias=False)
        self.shared_down = nn.Linear(d_shared_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.active_moe_backend = self._resolve_moe_backend(
            moe_backend, require_fused_kernel
        )
        self.last_indices = None
        # Chunking keeps vectorized fallback from materializing [M, D, H] for large M.
        self.vectorized_chunk_size = int(
            max(1, int(os.getenv("T19_MOE_VECTORIZED_CHUNK", "64")))
        )
        # T3 fixed path in 3B MoE: fused dispatch packing + index_add accumulation.
        self.permute_fusion_enabled = True
        self.fast_scatter_enabled = True
        # T4 optional hook; no-op on single GPU until dispatcher backend is integrated.
        self.t4_enabled = bool(t4_enabled)
        self.t4_dispatcher = str(t4_dispatcher)

    def _init_shared_weights(self):
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _resolve_moe_backend(
        self, requested_backend: str, require_fused_kernel: bool
    ) -> str:
        valid = {"auto", "vectorized", "grouped_gemm"}
        if requested_backend not in valid:
            raise ValueError(
                f"Unknown moe_backend={requested_backend!r}. Valid options: {sorted(valid)}."
            )

        grouped_available = bool(HAS_MOE_GROUPED_GEMM and moe_grouped_gemm is not None)
        if requested_backend == "vectorized":
            if require_fused_kernel:
                raise RuntimeError(
                    "MoE fused kernel is required but moe_backend='vectorized' was requested."
                )
            return "vectorized"

        if requested_backend == "grouped_gemm":
            if not grouped_available:
                raise RuntimeError(
                    "MoE grouped_gemm backend was requested but is unavailable.\n"
                    "Verify/install on this environment with:\n"
                    '  python -c "import grouped_gemm" || '
                    "pip install -U megatron-core transformer-engine"
                )
            return "grouped_gemm"

        # auto
        if grouped_available:
            return "grouped_gemm"
        if require_fused_kernel:
            raise RuntimeError(
                "MoE fused kernel is required but grouped_gemm backend is unavailable.\n"
                "Verify/install on this environment with:\n"
                '  python -c "import grouped_gemm" || '
                "pip install -U megatron-core transformer-engine"
            )
        return "vectorized"

    def _moe_vectorized(
        self, sorted_x: torch.Tensor, sorted_expert_indices: torch.Tensor
    ):
        # Per-assignment vectorized matmul in chunks to reduce peak VRAM.
        m = sorted_x.size(0)
        if m == 0:
            return torch.empty_like(sorted_x)

        out = torch.empty(
            (m, self.d_model), device=sorted_x.device, dtype=sorted_x.dtype
        )
        chunk = self.vectorized_chunk_size
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            x_chunk = sorted_x[start:end]
            idx_chunk = sorted_expert_indices[start:end]

            x_expanded = x_chunk.unsqueeze(1)  # [C, 1, D]
            w_gate_sel = self.W_gate[idx_chunk]  # [C, D, H]
            w_up_sel = self.W_up[idx_chunk]  # [C, D, H]
            w_down_sel = self.W_down[idx_chunk]  # [C, H, D]

            gate_out = torch.bmm(x_expanded, w_gate_sel).squeeze(1)  # [C, H]
            up_out = torch.bmm(x_expanded, w_up_sel).squeeze(1)  # [C, H]
            h = liger_silu_mul(gate_out, up_out)
            if self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout)
            out[start:end] = torch.bmm(h.unsqueeze(1), w_down_sel).squeeze(1)  # [C, D]
        return out

    def _moe_grouped(self, sorted_x: torch.Tensor, expert_counts: torch.Tensor):
        x_in = sorted_x.to(dtype=self.W_gate.dtype)
        # Phase 3: Use fused gate+up+SiLU kernel (1 kernel instead of 3)
        if fused_moe_gate_up_silu is not None:
            h = fused_moe_gate_up_silu(x_in, self.W_gate, self.W_up, expert_counts)
        elif triton_grouped_gemm is not None:
            gate_out = triton_grouped_gemm(x_in, self.W_gate, expert_counts)
            up_out = triton_grouped_gemm(x_in, self.W_up, expert_counts)
            h = liger_silu_mul(gate_out, up_out)
        else:
            gate_out = moe_grouped_gemm(x_in, self.W_gate, expert_counts)
            up_out = moe_grouped_gemm(x_in, self.W_up, expert_counts)
            h = liger_silu_mul(gate_out, up_out)
        if self.training and self.dropout > 0:
            h = F.dropout(h, p=self.dropout)
        # Phase 3: Use Triton grouped GEMM for down projection
        if triton_grouped_gemm is not None:
            out = triton_grouped_gemm(h, self.W_down, expert_counts)
        else:
            out = moe_grouped_gemm(h, self.W_down, expert_counts)
        return out.to(dtype=sorted_x.dtype)

    def forward(self, x: torch.Tensor):
        if self.expert_parallel_size > 1 and not self.t4_enabled:
            raise RuntimeError(
                "moe_expert_parallel_size > 1 requires a dispatcher backend (set moe_t4_enabled=true)."
            )

        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype
        x = x.to(dtype=self.shared_gate.weight.dtype)

        # Shared expert branch (always active).
        shared_h = liger_silu_mul(self.shared_gate(x), self.shared_up(x))
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # Routed branch.
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        if self.track_last_indices:
            self.last_indices = topk_idx.detach()
        else:
            self.last_indices = None

        flat_x = x.reshape(N, D)
        if self.permute_fusion_enabled:
            flat_idx_k = topk_idx.reshape(-1)
            flat_weight_k = topk_weight.reshape(-1)
            flat_is_null_k = is_null.reshape(-1)
            token_idx_k = torch.arange(
                N, device=device, dtype=torch.long
            ).repeat_interleave(K)
            real_mask_k = ~flat_is_null_k
            real_token_indices = token_idx_k[real_mask_k]
            real_expert_indices = flat_idx_k[real_mask_k]
            real_weights = flat_weight_k[real_mask_k]
        else:
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
        sorted_expert_indices = real_expert_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=E)
        num_real_assignments = sorted_token_indices.size(0)
        if num_real_assignments > 0:
            if self.active_moe_backend == "grouped_gemm":
                try:
                    sorted_out = self._moe_grouped(sorted_x, expert_counts)
                except Exception as exc:
                    if self.require_fused_kernel or (
                        not self.allow_vectorized_fallback
                    ):
                        raise RuntimeError(
                            "MoE grouped_gemm execution failed and vectorized fallback is disabled."
                        ) from exc
                    sorted_out = self._moe_vectorized(sorted_x, sorted_expert_indices)
            else:
                sorted_out = self._moe_vectorized(sorted_x, sorted_expert_indices)

            # Phase 3: Fused weighted scatter-add (1 kernel instead of mul + index_add)
            sorted_out = sorted_out.to(dtype=dtype)
            if fused_weighted_scatter_add is not None:
                routed_out = fused_weighted_scatter_add(
                    sorted_out, sorted_weights, sorted_token_indices, N
                )
            elif self.fast_scatter_enabled:
                sorted_out.mul_(sorted_weights.unsqueeze(-1).to(dtype=sorted_out.dtype))
                routed_out = torch.zeros(N, D, device=device, dtype=dtype)
                routed_out.index_add_(0, sorted_token_indices, sorted_out)
            else:
                weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
                routed_out = torch.zeros(N, D, device=device, dtype=dtype)
                routed_out.scatter_add_(
                    0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out
                )
        else:
            routed_out = torch.zeros(N, D, device=device, dtype=dtype)

        y = shared_out + routed_out.view(B, T, D)
        if self.t4_enabled:
            # Placeholder only. Real dispatcher backend to be wired in multi-GPU training stack.
            _ = self.t4_dispatcher
        return y, aux_loss


class LightningMLP(nn.Module):
    """MoE wrapper used by decoder and MTP blocks."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.moe = MoEFFN(
            d_model=config.hidden_size,
            d_hidden=config.expert_intermediate_size,
            d_shared_hidden=config.shared_expert_intermediate_size,
            num_experts=config.num_real_experts,
            num_null_experts=config.num_null_experts,
            top_k=config.top_k,
            dropout=config.dropout,
            data_sparsity=config.data_sparsity,
            moe_backend=config.moe_backend,
            require_fused_kernel=config.require_fused_moe_kernel,
            allow_vectorized_fallback=bool(
                getattr(config, "allow_moe_vectorized_fallback", False)
            ),
            track_last_indices=bool(getattr(config, "track_moe_last_indices", False)),
            expert_parallel_size=int(getattr(config, "moe_expert_parallel_size", 1)),
            t4_enabled=bool(getattr(config, "moe_t4_enabled", False)),
            t4_dispatcher=str(getattr(config, "moe_t4_dispatcher", "deepep")),
        )

    def forward(self, x):
        return self.moe(x)


# ============================================================================
# mHC (Multi-Head Composition) - From test model
# ============================================================================


def sinkhorn_knopp(
    logits: torch.Tensor, iters: int = 5, eps: float = 1e-6
) -> torch.Tensor:
    """
    Sinkhorn-Knopp doubly-stochastic normalisation (Triton-only).

    This path is intentionally strict: no PyTorch fallback.
    Forward and backward must go through the Triton sinkhorn kernel implementation.
    """
    if not (HAS_TRITON and triton_sinkhorn_knopp is not None and logits.is_cuda):
        raise RuntimeError(
            "Sinkhorn requires Triton CUDA kernel (no fallback enabled)."
        )

    logits_stable = logits - logits.amax(dim=-1, keepdim=True)
    with time_region("sinkhorn.triton"):
        return triton_sinkhorn_knopp(logits_stable, num_iters=iters, eps=eps)


class MHCCoeffs(nn.Module):
    """Produces routing coefficients for mHC."""

    def __init__(self, d_model: int, n_streams: int = 4, iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.iters = iters
        # Perf-only ablations (no architecture change):
        # - T17_MHC_FUSE_COEFF_PROJ=1: one fused linear over concatenated phi weights
        # - T17_MHC_RMS_FWD_ONLY=1: use forward-only Triton RMSNorm in no-grad execution
        self.fuse_coeff_proj = os.getenv("T17_MHC_FUSE_COEFF_PROJ", "0") == "1"
        self.rms_fwd_only = os.getenv("T17_MHC_RMS_FWD_ONLY", "0") == "1"

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
        use_rms_fast = (
            self.rms_fwd_only
            and triton_rmsnorm_fwd_only is not None
            and x_flat.is_cuda
            and (not torch.is_grad_enabled())
        )
        if use_rms_fast:
            # Forward-only fast path for inference/no-grad benchmarking.
            x_flat = triton_rmsnorm_fwd_only(x_flat, self.rms.weight, self.rms.eps)
        else:
            x_flat = self.rms(x_flat)

        # Cast to weight dtype to prevent float32/bfloat16 mismatch during reversible backward
        x_flat = x_flat.to(self.phi_pre.weight.dtype)

        if self.fuse_coeff_proj:
            # One projection launch (24 outputs) then split to pre/post/res branches.
            phi_all_weight = torch.cat(
                (self.phi_pre.weight, self.phi_post.weight, self.phi_res.weight),
                dim=0,
            )
            logits_all = F.linear(x_flat, phi_all_weight)
            pre_raw, post_raw, res_raw = torch.split(
                logits_all, (self.n, self.n, self.n * self.n), dim=-1
            )
            pre_logits = self.alpha_pre * pre_raw + self.b_pre
            post_logits = self.alpha_post * post_raw + self.b_post
            res_logits = (self.alpha_res * res_raw).view(B, T, n, n) + self.b_res
        else:
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
        # Profiler labels — overridden by Model3B.__init__ after construction.
        # time_region is zero-overhead when profiler is inactive (1 global read + branch).
        self._prof_coeffs_label: str = ""
        self._prof_sublayer_label: str = ""

    def forward(self, x_stream: torch.Tensor, attention_mask=None):
        _pcl = self._prof_coeffs_label
        _psl = self._prof_sublayer_label

        if _pcl:
            with time_region(_pcl):
                H_pre, H_post, H_res = self.coeffs(x_stream)
        else:
            H_pre, H_post, H_res = self.coeffs(x_stream)

        if (
            fused_mhc_collapse is not None
            and x_stream.is_cuda
            and x_stream.is_contiguous()
        ):
            x_in = fused_mhc_collapse(x_stream, H_pre)
        else:
            x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)
        x_in = self.norm(x_in)

        aux_loss = None
        if _psl:
            with time_region(_psl):
                if attention_mask is None:
                    out = self.sublayer(x_in)
                else:
                    out = self.sublayer(x_in, attention_mask)
        else:
            if attention_mask is None:
                out = self.sublayer(x_in)
            else:
                out = self.sublayer(x_in, attention_mask)

        if isinstance(out, tuple):
            y, aux_loss = out
        else:
            y = out

        if fused_mhc_expand_residual is not None and y.is_cuda and y.is_contiguous():
            out_stream = fused_mhc_expand_residual(y, x_stream, H_post, H_res)
        else:
            y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)
            #   Use matmul instead of einsum (hits optimized GEMM)
            x_res = torch.matmul(H_res.to(dtype=x_stream.dtype), x_stream)
            out_stream = x_res + y_stream

        return out_stream, aux_loss


# ============================================================================
# Decoder Layer (Hybrid DeltaNet + GSA -- DDDGDDDG)
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


class _ReversibleStreamSublayer(nn.Module):
    """Adapter that exposes MHCSublayer as a reversible-force layer."""

    def __init__(self, block: MHCSublayer, use_attention_mask: bool):
        super().__init__()
        self.block = block
        self.use_attention_mask = use_attention_mask

    def force(self, x, attention_mask=None):
        if self.use_attention_mask:
            out, aux = self.block(x, attention_mask=attention_mask)
        else:
            out, aux = self.block(x, attention_mask=None)
        delta = out - x
        if aux is None:
            aux = delta.reshape(-1)[0] * 0.0
        return delta, aux

    def forward(self, x, attention_mask=None):
        if self.use_attention_mask:
            return self.block(x, attention_mask=attention_mask)
        return self.block(x, attention_mask=None)


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
        attn_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=self.attn,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )

        mlp_block = MHCSublayer(
            d_model=config.hidden_size,
            n_streams=config.n_streams,
            sublayer=self.mlp,
            norm=RMSNorm(config.hidden_size),
            iters=config.sinkhorn_iters,
        )
        self.mtp_reversible = bool(getattr(config, "mtp_reversible", True))
        if self.mtp_reversible:
            from .reversible_ops_midpoint import ReversibleMidpointStack

            mtp_blocks = nn.ModuleList(
                [
                    _ReversibleStreamSublayer(attn_block, use_attention_mask=True),
                    _ReversibleStreamSublayer(mlp_block, use_attention_mask=False),
                ]
            )
            self.reversible_stack = ReversibleMidpointStack(
                mtp_blocks,
                step_size=float(getattr(config, "mtp_step_size", 0.25)),
                a=float(getattr(config, "mtp_a", 0.5)),
                noise_eps=0.0,
                bootstrap=str(getattr(config, "mtp_bootstrap", "euler")),
            )
            # Keep backwards-compatible access for profiler/cache code without
            # registering duplicate module paths in state_dict.
            object.__setattr__(self, "attn_block", attn_block)
            object.__setattr__(self, "mlp_block", mlp_block)
        else:
            self.attn_block = attn_block
            self.mlp_block = mlp_block
            self.reversible_stack = None

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
        x_stream = torch.empty(
            batch_size,
            seq_len,
            self.n_streams,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        x_stream[:, :, 0, :] = x
        x_stream[:, :, 1:, :] = 0

        # NOTE: Memory stream injection happens in the main Model3B.forward(),
        # not here. The MTP block receives h_t which already contains recurrence
        # information from the backbone processing.

        if self.reversible_stack is not None:
            x_stream, _ = self.reversible_stack(x_stream, attention_mask=attention_mask)
        else:
            # mHC blocks (ignore aux_loss for clean aux-loss accounting)
            x_stream, _ = self.attn_block(x_stream, attention_mask=attention_mask)
            x_stream, _ = self.mlp_block(x_stream, attention_mask=None)

        # Collapse
        x_out = x_stream.mean(dim=2)

        return x_out


# ============================================================================
# Complete 3B MoE Model
# ============================================================================


class Model3B(nn.Module):
    """
    3B MoE Model -- 1B backbone with null-routed MoE FFN.

    Configuration:
    - 8 layers: 6 DeltaNet + 2 GSA (DDDGDDDG)
    - MoE FFN: 20 real + 20 null slots, top-k=2
    - Shared expert: 2048 (always active), routed experts: 1024
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
    - Aux loss: l_bal (expert balance) + l_z (z-loss) + l_null (null-rate regularizer)
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

        # Profiler labels for per-layer sub-component timing (zero-cost when inactive)
        for i, layer in enumerate(layers):
            kt = layer_types[i]  # "deltanet" or "gsa"
            layer.attn_block._prof_coeffs_label = f"layer{i}.sinkhorn_attn.fwd"
            layer.attn_block._prof_sublayer_label = f"layer{i}.{kt}.fwd"
            layer.mlp_block._prof_coeffs_label = f"layer{i}.sinkhorn_mlp.fwd"
            layer.mlp_block._prof_sublayer_label = f"layer{i}.mlp.fwd"

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
            self.mtp_block.attn_block._prof_coeffs_label = "mtp_block.sinkhorn_attn.fwd"
            self.mtp_block.attn_block._prof_sublayer_label = "mtp_block.gsa.fwd"
            self.mtp_block.mlp_block._prof_coeffs_label = "mtp_block.sinkhorn_mlp.fwd"
            self.mtp_block.mlp_block._prof_sublayer_label = "mtp_block.mlp.fwd"
        else:
            self.mtp_block = None

        # ============================================================================
        # Memory Stream Recurrence -- "different" style (same as different_recurrence_model_1b_wo_rev.py)
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
                f"   🔧 pf_to_model (8192->{config.hidden_size}) initialized with std={pf_to_model_std:.6f}"
            )

        # Print configuration
        total_params = sum(p.numel() for p in self.parameters())

        # Calculate embedding parameters
        if self.use_kronecker:
            # Kronecker embeddings: vocab_size × D (buffer, not parameters)
            # pf_to_model: D × hidden_size (trainable)
            embedding_buffer = (
                self.vocab_size * (32 + 2) / 1e6
            )  # gpu_dynamic: byte+len buffers
            embedding_params = self._D_pf * config.hidden_size / 1e6  # In millions
        else:
            embedding_params = self.vocab_size * config.hidden_size / 1e6
            embedding_buffer = 0

        print("\n🤖 MODEL-3B-MOE-CORRECT INITIALIZED:")
        print(f"   Vocabulary: {self.vocab_size:,}")
        print(f"   Hidden Size: {config.hidden_size}")
        if self.use_kronecker:
            print("\n   📐 Kronecker Embeddings:")
            print("      POS_DIM=32 x CHAR_DIM=256 = D=8192")
            print(
                f"      Buffer size: {embedding_buffer:.1f}M elements ({embedding_buffer * 1:.1f} MB, gpu_dynamic mode)"
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
            f"   Experts: {config.num_real_experts} real + {config.num_null_experts} null = {config.total_expert_slots} slots"
        )
        print(f"   Top-k: {config.top_k} over total slots")
        print(
            f"   Shared Expert FFN: {config.shared_expert_intermediate_size} (always active)"
        )
        print(f"   Routed Expert FFN: {config.expert_intermediate_size} (sparse)")
        print(
            f"   MTP: {config.mtp_num_predictions} predictions"
            if config.enable_mtp
            else "   MTP: Disabled"
        )
        print(f"\n   Total Parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
        print("   Active Parameters: dynamic (depends on real routes + shared expert)")

    def _init_weights(self, module):
        # FIX #38: Skip initialization for kronecker_embeddings and all its submodules
        # (was using named_modules() which returns (name, module), not (name, param))
        if self.use_kronecker and self.kronecker_embeddings is not None:
            if module is self.kronecker_embeddings:
                return
            for submodule in self.kronecker_embeddings.modules():
                if module is submodule:
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

        # Embeddings — shared computation for NTP + MTP
        if next_token_ids is not None:
            all_ids = torch.cat([input_ids, next_token_ids[:, -1:]], dim=1)
        else:
            all_ids = input_ids

        if self.use_kronecker:
            EMB = self.kronecker_embeddings(all_ids)
            dtype_target = self.pf_to_model.weight.dtype
            full_emb = self.pf_to_model(EMB.to(dtype=dtype_target))
            full_emb = self.embed_norm(full_emb)
        else:
            full_emb = self.token_embed(all_ids)

        if next_token_ids is not None:
            x = full_emb[:, :-1, :]
        else:
            x = full_emb

        B, T, D = x.shape

        # ============================================================================
        # EMBEDDING-SPACE MEMORY INJECTION (before stream expansion) -- "different" style
        # ============================================================================
        if prev_memory_stream is not None:
            prev_memory_stream = prev_memory_stream.detach()
            memory = self.memory_ln(prev_memory_stream)
            memory_gates = torch.sigmoid(self.memory_gate_proj(x))  # (B, T, 1)
            memory_broadcast = memory.unsqueeze(1).expand(B, T, D)
            lambda_r = F.softplus(self.lambda_r_raw)
            x = x + lambda_r * memory_gates * memory_broadcast

        # Expand to streams
        x_stream = torch.empty(B, T, self.n_streams, D, device=x.device, dtype=x.dtype)
        x_stream[:, :, 0, :] = x
        x_stream[:, :, 1:, :] = 0

        #   Centralized RoPE caching (Bit-identical sharing across layers)
        # Different layers (DeltaNet vs GSA) use different head_dims (128 vs 256).
        # We pre-compute RoPE for all needed dims exactly once here.
        distinct_dims = set()
        for layer in self.layers:
            distinct_dims.add(layer.attn_block.sublayer.rotary_emb.dim)
        if self.mtp_block is not None:
            distinct_dims.add(self.mtp_block.attn_block.sublayer.rotary_emb.dim)

        # Push the shared reference to all matching layers
        for d in distinct_dims:
            # Pick the specific instance to use as the "master" for this dimension
            _rep_rotary = None
            for layer in self.layers:
                rm = layer.attn_block.sublayer.rotary_emb
                if rm.dim == d:
                    _rep_rotary = rm
                    break

            # Fallback to MTP block if no stack layer matches this dimension
            if _rep_rotary is None and self.mtp_block is not None:
                rm = self.mtp_block.attn_block.sublayer.rotary_emb
                if rm.dim == d:
                    _rep_rotary = rm

            if _rep_rotary is not None:
                # Compute ONCE for this dimension
                _c, _s, _cb, _sb = _rep_rotary._compute_cos_sin(T, x.device, x.dtype)
                _ck = (T, x.device, x.dtype, d)

                # Push the shared 4-tuple references (includes broadcasted views)
                for layer in self.layers:
                    rm = layer.attn_block.sublayer.rotary_emb
                    if rm.dim == d:
                        if not hasattr(rm, "_forward_cache"):
                            rm._forward_cache = {}
                        rm._forward_cache[_ck] = (_c, _s, _cb, _sb)

                if self.mtp_block is not None:
                    rm = self.mtp_block.attn_block.sublayer.rotary_emb
                    if rm.dim == d:
                        if not hasattr(rm, "_forward_cache"):
                            rm._forward_cache = {}
                        rm._forward_cache[_ck] = (_c, _s, _cb, _sb)

        # Pass through reversible stack
        x_stream, total_aux_loss = self.stack(x_stream, attention_mask=token_keep_mask)

        # Collapse streams
        h_main = x_stream.mean(dim=2)
        h_main = self.norm(h_main)

        # ============================================================================
        # EXTRACT MEMORY from collapsed h_main (not stream-3) -- "different" style
        # ============================================================================
        if return_memory:
            memory_stream_out = h_main[:, -1, :].detach()
        else:
            memory_stream_out = None

        # NTP Prediction
        # return_hidden=True: skip lm_head, return raw hidden states so train.py
        # can call FusedLinearCrossEntropyLoss without ever creating logit tensors.
        if return_hidden:
            logits_ntp = h_main  # [B, T, H] -- NOT logits
        else:
            logits_ntp = self.lm_head(h_main)  # [B, T, V]

        # MTP Prediction — reuse shared embedding via slice (no second Kronecker call)
        logits_mtp = None
        if self.mtp_block is not None and next_token_ids is not None:
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_emb = full_emb[:, 1 : 1 + min_len, :]

            mtp_attention_mask = (
                token_keep_mask[:, :min_len] if token_keep_mask is not None else None
            )
            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=mtp_attention_mask)
            h_mtp_normed = self.norm(h_mtp)
            if return_hidden:
                logits_mtp = h_mtp_normed  # [B, T, H] -- NOT logits
            else:
                logits_mtp = self.lm_head(h_mtp_normed)  # [B, T, V]

        # FIX #41: Clear RoPE forward-pass cache to prevent accumulation.
        # Applies to all layer rotary modules (DeltaNet/GSA mix) and optional MTP block.
        for layer in self.layers:
            if hasattr(layer.attn_block.sublayer, "rotary_emb"):
                if hasattr(layer.attn_block.sublayer.rotary_emb, "_forward_cache"):
                    layer.attn_block.sublayer.rotary_emb._forward_cache.clear()

        # Also clear MTP block cache if enabled
        if self.mtp_block is not None:
            if hasattr(self.mtp_block.attn_block.sublayer, "rotary_emb"):
                if hasattr(
                    self.mtp_block.attn_block.sublayer.rotary_emb, "_forward_cache"
                ):
                    self.mtp_block.attn_block.sublayer.rotary_emb._forward_cache.clear()

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
# Factory Function
# ============================================================================


def create_model_3b(embedding_type="kronecker", bpe_vocab=None, pf_codec=None):
    """
    Create 3B MoE model with default configuration.

    Kronecker embeddings are the default path. Pass bpe_vocab and pf_codec
    when using embedding_type="kronecker".

    Args:
        embedding_type: "kronecker" (default, recommended) or "standard"
        bpe_vocab: Required for Kronecker embeddings (word list for Kronecker codec)
        pf_codec: Required for Kronecker embeddings (KroneckerEmbeddings instance)

    Returns:
        Model3B instance
    """
    config = ModelConfig()
    return Model3B(
        config, embedding_type=embedding_type, bpe_vocab=bpe_vocab, pf_codec=pf_codec
    )


# Backward-compatible aliases while this file is being integrated.
Model1B = Model3B
create_model_1b = create_model_3b


if __name__ == "__main__":
    # Calculate actual metrics from weight_calculator.py
    from weight_calculator import LightningCalculator, LightningConfig

    config_calc = LightningConfig(
        vocab_size=131072,
        hidden_size=4096,
        target_params=3e9,
        attention_type="gsa",
        deltanet_layer_ratio=0.75,
        num_routed_experts_active=1,
        num_shared_experts=1,
        expert_intermediate_size=1024,
        shared_expert_intermediate_size=2048,
        enable_mtp=True,
        mtp_num_predictions=2,
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
    print("3B MOE MODEL ARCHITECTURE (CORRECT)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Total Params: {total_params:.3f}B")
    print(f"  Active Params: {active_params:.3f}B")
    print(f"  Sparsity: {sparsity:.1f}x")
    print("\nAttention (DDDGDDDG -- DeltaNet + GSA):")
    print(
        f"  DeltaNet: {config.num_deltanet_layers} layers ({100*config.num_deltanet_layers//config.num_layers}%) - O(N) linear attention"
    )
    print(
        f"  GSA: {config.num_gsa_layers} layers ({100*config.num_gsa_layers//config.num_layers}%) - Adaptive sparse attention"
    )
    print("\nModel Type: MoE")
    print(f"  Real experts: {config.num_real_experts}")
    print(f"  Null experts: {config.num_null_experts}")
    print(f"  Top-k over total slots: {config.top_k}")
    print(f"  Shared expert FFN: {config.shared_expert_intermediate_size}")
    print(f"  Routed expert FFN: {config.expert_intermediate_size}")
    print("\nEmbedding: Kronecker (default)")
    print(f"Context: {config.max_seq_len:,} tokens")
    print("=" * 80)
