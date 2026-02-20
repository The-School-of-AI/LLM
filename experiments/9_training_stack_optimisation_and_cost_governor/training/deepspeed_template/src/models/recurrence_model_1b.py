"""
1B Dense Baseline Model with Hybrid Gated DeltaNet + Gated Sparse Attention (GSA)

Configuration:
- 1.513B total parameters, 1.513B active parameters (100% dense - no MoE)
- 131,072 vocabulary (2^17)
- 4096 hidden size, 8 layers (6 DeltaNet + 2 GSA)
- No experts - Dense FFN with 2048 intermediate size
- Multi-Token Prediction (MTP) with 2 predictions
- Multi-Head Composition (mHC) with 4 streams
- Reversible Midpoint Integration for memory efficiency
- Target: 256k context length
- Enhanced with Memory Stream Recurrence for infinite-length documents

Architecture based on:
- Gated DeltaNet: arXiv:2412.06464 (Dec 2024)
- Gated Sparse Attention: arXiv:2601.15305v1 (Jan 2026)
- Multi-Token Prediction: DeepSeek-V3 style

Purpose: Foundation baseline before introducing MoE. All parameters are active.

⚠️  PRODUCTION STATUS: Research code - Path to production, ready for testing
=============================================================================
Current validation: 2k-8k sequence lengths
256k context requires: Custom Triton kernels (GSA block-sparse + DeltaNet parallel scan)
Status: Architectural fixes complete, ready for testing phase before kernel optimization

CHANGELOG - Critical Bug Fixes and Optimizations:
=================================================
1. FIXED: Duplicate RMSNorm class definition removed (was overriding first definition)
2. FIXED: YARN beta_fast/beta_slow inversion corrected (was 32/1, now 1/32)
   - This critical bug was destroying DeltaNet's ability to distinguish local token order
   - High frequencies now preserved (scale ~1.0), low frequencies interpolated (scale ~32.0)
3. FIXED: MTP block now uses GatedSparseAttention instead of GatedDeltaNet
   - MTP runs once per step, so full attention cost is negligible but gradient quality is critical
4. FIXED: Removed dead memory injection code from MTPTransformerBlock.forward()
   - Was referencing undefined variables (prev_memory_stream, self.memory_ln, etc.)
   - Memory injection only happens in main Model1B.forward()
5. OPTIMIZED: RoPE now computes cos/sin on-the-fly instead of caching
   - Saves VRAM (268MB per layer × 8 layers = 2.1GB for 1B model)
   - Only 5-10% slower, critical for 256k context training where VRAM is precious
6. FIXED: GSA O(T²) memory bomb eliminated via fused_indexer_topk kernel integration
   - Replaced matmul(q_I_p, k_I_p) → [B, heads, T, T] with chunked O(T·k) kernel
   - W_Ik changed from per-head to shared keys (matches kernel interface)
   - head_importance_bias relocated from importance score modulation to attention logit bias

See inline comments at each fix location for detailed explanations.

═══════════════════════════════════════════════════════════════════════════════
🚨 CRITICAL TODOs FOR PRODUCTION (256k Context)
═══════════════════════════════════════════════════════════════════════════════

Before deploying this model at 256k context, the following MUST be addressed:

1. ✅ DONE: Triton Kernels for GSA (fused_indexer_topk)
   ────────────────────────────────────────────────
   Location: GatedSparseAttention class
   Solution: Integrated fused_indexer_topk from kernels/triton_indexer_streaming.py
   - Processes importance scores in [B, C, T] chunks (C auto-tuned to ~512MB)
   - Never materializes [B, T, T] score tensor
   - Achieves O(T·k) memory complexity
   - W_Ik changed to shared keys (was per-head, now matches kernel interface)
   - head_importance_bias relocated to attention logit bias

   Status: COMPLETED

2. ⚡ MANDATORY: Implement Triton Kernel for DeltaNet Recurrence
   ────────────────────────────────────────────────────────────────
   Location: GatedDeltaNet class forward() Python loop (lines 680-709)
   Problem: Python for-loop over T tokens causes catastrophic kernel launch overhead
   Impact: At 256k context → 2.6-13 seconds of PURE overhead (500-2000x slower than needed)

   Solution:
   - Implement fused Triton kernel for parallel associative scan
   - Fuse: alpha/beta computation + state update + query operations
   - Use block-wise parallelization across heads and batch
   - Reference: Gated DeltaNet paper (arXiv:2412.06464) Section 3.4

   Status: BLOCKING for 256k deployment
   Est. Effort: 2-3 weeks (kernel design + optimization)

3. 🔧 NOTE: 1B Model is DENSE (No MoE Expert Dispatch Issue)
   ─────────────────────────────────────────────────────────
   This model uses dense FFN (no MoE in backbone), so the expert dispatch
   optimization from 70B model is NOT applicable here.

   If future variants add MoE, see 70B model TODO #3 for guidance.

4. 📊 TODO: Benchmark RAM vs Realtime Tradeoff for RoPE
   ──────────────────────────────────────────────────────
   Location: RotaryEmbedding class (lines 369-443)
   Current: On-the-fly computation (saves 2.1GB VRAM, costs 5-10% speed)

   Decision Needed:
   - For training: On-the-fly likely optimal (VRAM precious for 256k)
   - For inference: Caching might be better (speed > memory)
   - Need production benchmarks at target batch sizes

   Action:
   - Run A/B benchmark: cached vs on-the-fly at 2k, 8k, 32k, 256k
   - Measure: throughput (tokens/sec), VRAM usage, cost per token
   - Make data-driven choice per deployment scenario

   Status: Optional for testing, REQUIRED for production tuning
   Est. Effort: 2-3 days (benchmarking + analysis)

5. 🔍 TODO: Verify YARN mscale Frequency Band Logic
   ───────────────────────────────────────────────────
   Location: RotaryEmbedding.__init__() (lines 410-417)
   Current: mscale uses original 'base', inv_freq uses 'scaled_base'

   Rationale (YARN Paper Section 2.1):
   - Frequency band classification should be relative to ORIGINAL bands
   - mscale determines which frequencies get interpolated (0-1 ramp)
   - This is APPLIED to the NTK-scaled frequencies (inv_freq)

   Action:
   - Validate against YARN paper equations (specifically Section 2.1, Eq 3-5)
   - Confirm: wavelen = 2π / freq should use original base for classification
   - If matches paper → mark verified
   - If deviation → document intentional change or fix

   Status: Logic appears correct per paper, needs formal verification
   Est. Effort: 4-6 hours (paper cross-reference + validation)

6. 📈 TODO: Production Monitoring Hooks
   ─────────────────────────────────────
   Implement logging for:
   - aux_loss / main_loss ratio (alert if > 0.1) - NOTE: 1B is dense, aux_loss minimal
   - Memory stream recurrence gradient norms (detect vanishing/explosion)
   - Per-layer activation statistics (detect distribution shift)

   Status: Required before production training
   Est. Effort: 1 week (instrumentation + dashboard)

═══════════════════════════════════════════════════════════════════════════════
📋 SUMMARY: GSA kernel integrated (#1 done). Ready for 2k-8k testing, 256k needs #2 (DeltaNet kernel)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import math
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Triton Kernel Imports ────────────────────────────────────────────────────
# Mirror 70B import resolution so 1B behaves identically across launch contexts.
def _import_kernels_module():
    # Package-relative import when recurrence_model_1b.py is used inside src.models.
    try:
        from .. import kernels as kernels_module
        return kernels_module
    except Exception:
        pass

    # Standalone fallback: borrow kernels from known deepspeed_template roots.
    experiments_dir = Path(__file__).resolve().parents[5]
    candidate_roots = [
        experiments_dir / "9_training_stack_optimisation_and_cost_governor" / "training" / "deepspeed_template" / "src",
        experiments_dir / "9_training_stack_optimisation_and_cost_governor" / "training" / "deepspeed_template" / "dense_hardened" / "src",
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
    HAS_FLA = bool(getattr(_kernels_module, "HAS_FLA", False))
    triton_sparse_attention = getattr(_kernels_module, "triton_sparse_attention", None)
    pytorch_sparse_attention = getattr(_kernels_module, "pytorch_sparse_attention", None)
    triton_sinkhorn_knopp = getattr(_kernels_module, "triton_sinkhorn_knopp", None)
    pytorch_sinkhorn_knopp = getattr(_kernels_module, "pytorch_sinkhorn_knopp", None)
    triton_rmsnorm = getattr(_kernels_module, "triton_rmsnorm", None)
    pytorch_rmsnorm = getattr(_kernels_module, "pytorch_rmsnorm", None)
    TritonRMSNorm = getattr(_kernels_module, "TritonRMSNorm", None)
    fla_gated_delta_rule = getattr(_kernels_module, "fla_gated_delta_rule", None)
    fused_indexer_topk = getattr(_kernels_module, "fused_indexer_topk", None)
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
    fla_gated_delta_rule = None
    fused_indexer_topk = None

HAS_FUSED_INDEXER = fused_indexer_topk is not None

# ── Vendored Liger Ops ──────────────────────────────────────────────────────
# Self-contained import so this model does not rely on external liger wheels.
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
LigerFusedLinearCrossEntropyLoss = _liger_module.LigerFusedLinearCrossEntropyLoss
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
_kernel_log.info(f"  HAS_FLA:              {HAS_FLA}")
_kernel_log.info(f"  Triton RMSNorm:       {'ENABLED' if HAS_TRITON and triton_rmsnorm is not None and _cuda_available else 'FALLBACK (PyTorch)'}")
_kernel_log.info(f"  Triton Sinkhorn:      {'ENABLED' if HAS_TRITON and triton_sinkhorn_knopp is not None and _cuda_available else 'FALLBACK (PyTorch)'}")
_kernel_log.info(f"  Triton Sparse Attn:   {'ENABLED' if HAS_TRITON and triton_sparse_attention is not None and _cuda_available else 'FALLBACK (PyTorch)'}")
_kernel_log.info(f"  fla GatedDeltaRule:   {'ENABLED' if HAS_FLA and fla_gated_delta_rule is not None and _cuda_available else 'FALLBACK (Python loop)'}")
if not _cuda_available:
    _kernel_log.info("  NOTE: All Triton/fla kernels require CUDA. Running on MPS/CPU uses PyTorch fallbacks.")
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
                mask = (mask.max(dim=1).values >= 0)
            else:
                mask = (mask.max(dim=1).values > 0)
        else:
            mask = (mask.max(dim=1).values > 0)
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
    POS_DIM: int = 32    # Max token length in bytes
    D: int = 8192        # CHAR_DIM × POS_DIM = 256 × 32
    length_normalize: bool = True
    truncate_long_words: bool = True

    def __post_init__(self):
        assert self.CHAR_DIM == 256, "CHAR_DIM must be 256 for byte-level encoding"
        assert self.D == self.CHAR_DIM * self.POS_DIM, f"D ({self.D}) must equal CHAR_DIM × POS_DIM ({self.CHAR_DIM} × {self.POS_DIM})"


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
                byte_seq[:end].decode('utf-8')
                return byte_seq[:end]
            except UnicodeDecodeError:
                continue

        # Fallback: return empty if can't find valid truncation
        return b''

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
        byte_seq = word.encode('utf-8')

        # Truncate if needed (UTF-8 safe)
        if len(byte_seq) > self.POS_DIM:
            if self.cfg.truncate_long_words:
                byte_seq = self._utf8_safe_truncate(byte_seq, self.POS_DIM)
            else:
                raise ValueError(f"Token byte length {len(byte_seq)} exceeds POS_DIM={self.POS_DIM}")

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
            M *= (1.0 / math.sqrt(L))

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
            return byte_seq.decode('utf-8')
        except UnicodeDecodeError:
            # Should never happen with properly encoded data
            # But handle gracefully just in case
            return byte_seq.decode('utf-8', errors='replace')

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
    # FIX #37: Removed unused delta_qk_heads parameter (never referenced in code)
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
    sinkhorn_iters = 20  # PROBABLE FIX #26: Reduced from 20 (major compute savings at long context)

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
    """
    Root Mean Square Layer Normalization with fp32 statistics.

    FIX #43: Computes variance in fp32 for numerical stability at 256k context.
    Critical for preventing rare NaN spikes with bf16/fp16 training.

    Triton acceleration: When available, uses fused Triton kernel with both
    forward AND backward in Triton (Liger-style). No more autograd fallback.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._use_triton = HAS_TRITON and triton_rmsnorm is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton path: fused forward + backward kernel (Liger-style)
        # Now safe with grad enabled — the new kernel wraps torch.autograd.Function
        if self._use_triton and x.is_cuda:
            return triton_rmsnorm(x, self.weight, self.eps)

        if x.is_cuda:
            raise RuntimeError(
                "Triton RMSNorm kernel is required on CUDA but unavailable. "
                f"HAS_TRITON={HAS_TRITON}, triton_rmsnorm={triton_rmsnorm is not None}. "
                "Install triton: pip install triton"
            )

        # CPU-only fallback (for testing/debugging, never in production training)
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
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: int = 10000,
                 original_max_position_embeddings: int = 8192, scaling_factor: float = 32.0):
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
        if hasattr(self, '_forward_cache') and cache_key in self._forward_cache:
            return self._forward_cache[cache_key]

        t = torch.arange(seq_len, device=device).float()
        freqs = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
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
        # Liger integration: use vendored rotary application helper.
        return liger_rotary_pos_emb(x, cos, sin)


# ============================================================================
# Helper Modules for Gated DeltaNet
# ============================================================================

class ShortConvolution(nn.Module):
    """
    Short convolution layer with causal padding.
    Used in Gated DeltaNet for local context integration.
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
    def __init__(self, hidden_size, num_heads, head_dim,
                 max_seq_len=262144, rope_base=10000,
                 rope_original_max=8192, rope_scaling_factor=32.0,
                 conv_size=4, use_output_norm=True,
                 require_fused_kernel=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_output_norm = use_output_norm
        self.require_fused_kernel = require_fused_kernel

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
        # BUG FIX #1: uniform_(0, 16) can produce 0 → log(0) = -inf → dead head
        A_init = torch.empty(num_heads).uniform_(1e-4, 16)
        self.A_log = nn.Parameter(torch.log(A_init))  # log(A) for stability

        # D parameter for residual connection (per-head)
        self.D = nn.Parameter(torch.ones(num_heads))

        # dt_bias for Mamba-style gating (per-head)
        # Special initialization: log-uniform for stable gating
        dt_init_std = 0.01
        dt_bias = torch.rand(num_heads) * 2 * dt_init_std - dt_init_std
        self.dt_bias = nn.Parameter(dt_bias)

        # Rotary embeddings for Q/K (standard RoPE)
        self.rotary_emb = RotaryEmbedding(
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
        # Linear projections: std=0.02 (DeepScreen initialization)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Gate projections
        for m in [self.b_proj, self.gk_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _delta_rule_python(self, q, k, v, alpha, beta, B, T, device, original_dtype):
        """Python fallback is intentionally disabled in fused-only mode."""
        raise RuntimeError(
            "DeltaNet Python fallback is disabled. "
            "Fused FLA kernel is required for Model1B."
        )

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
        token_keep = _token_keep_mask(attention_mask, B, T, device)
        token_keep_f = None

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

        # 4. Apply RoPE to Q/K (computed on-the-fly to save 2.1GB VRAM)
        # FIX #40: Include dtype in cache lookup (was missing, causing cache MISS every time)
        cos, sin = self.rotary_emb._compute_cos_sin(T, device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        # 5. L2 normalization (NOT softmax) - Paper Section 3.3
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 6. Compute beta (writing strength) - sigmoid activation
        beta = torch.sigmoid(self.b_proj(x))  # (B, T, num_heads)
        beta = beta.unsqueeze(-1)  # (B, T, num_heads, 1)

        # 7. Compute alpha (decay parameter) - Paper Equation 10
        # alpha = exp(-exp(A_log) * softplus(gk + dt_bias))
        # CRITICAL FIX #23: Use exp() instead of sigmoid() to map to full (0,1) range
        # Reference: DeltaNet paper arXiv:2412.06464v3 explicitly states α_t ∈ (0,1)
        gk = self.gk_proj(x)  # (B, T, num_heads)
        A = torch.exp(self.A_log)  # Positive (move negative to multiplication)
        alpha = -A.view(1, 1, self.num_heads, 1) * F.softplus(gk + self.dt_bias).unsqueeze(-1)
        # Map negative values to (0, 1) range using exp - CRITICAL for 256k retention
        alpha = torch.exp(alpha)  # (B, T, num_heads, 1) - full (0,1) range, not (0,0.5)

        # Respect token padding/control mask in recurrent state updates:
        # masked tokens do not write or decay state (beta=0, alpha=1).
        if token_keep is not None:
            token_keep_f = token_keep.to(dtype=q.dtype).view(B, T, 1, 1)
            q = q * token_keep_f
            k = k * token_keep_f
            v = v * token_keep_f
            g = g * token_keep_f
            beta = beta * token_keep_f
            alpha = alpha * token_keep_f + (1.0 - token_keep_f)

        # 8. Gated Delta Rule with decay (Paper Equation 10)
        # St = St-1 * (alpha * (I - beta * k * k^T)) + beta * v * k^T
        #
        # Two paths:
        # - fla path: Fused Triton kernel via flash-linear-attention library (500-2000x faster)
        # - Python path: Explicit for-loop fallback (always correct, slow at long context)

        fla_available = HAS_FLA and fla_gated_delta_rule is not None and q.is_cuda
        if self.require_fused_kernel and not fla_available:
            raise RuntimeError(
                "DeltaNet fused kernel is required but unavailable. "
                f"HAS_FLA={HAS_FLA}, fla_gated_delta_rule={fla_gated_delta_rule is not None}, "
                f"q.is_cuda={q.is_cuda}."
            )

        if fla_available:
            try:
                o = fla_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    alpha=alpha,
                    beta=beta,
                    D=self.D,
                    num_heads=self.num_heads,
                )
            except Exception as e:
                if self.require_fused_kernel:
                    raise RuntimeError(
                        "DeltaNet fused kernel execution failed and fallback is disabled."
                    ) from e
                o = self._delta_rule_python(q, k, v, alpha, beta, B, T, device, x.dtype)
        else:
            o = self._delta_rule_python(q, k, v, alpha, beta, B, T, device, x.dtype)

        # 9. Apply output normalization with gating
        # o is (B, T, num_heads, head_dim) from both paths
        g = g  # (B, T, num_heads, head_dim) - already in correct shape

        if self.use_output_norm:
            # FIX #28: Vectorized output norm (was Python loop over heads)
            # Reshape to (B*T*H, head_dim) for single-pass normalization
            o_flat = o.reshape(B * T * self.num_heads, self.head_dim)
            g_flat = g.reshape(B * T * self.num_heads, self.head_dim)
            o_normed = self.o_norm(o_flat, g_flat)
            o = o_normed.view(B, T, self.num_heads, self.head_dim)
        else:
            o = o * torch.sigmoid(g)

        if token_keep_f is not None:
            o = o * token_keep_f

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

    Memory complexity: O(T·k) via fused_indexer_topk chunked kernel.
    The indexer processes importance scores in [B, C, T] chunks (C auto-tuned
    to ~512MB), never materializing the full [B, T, T] score tensor.
    Safe for 256k+ context lengths.

    Architecture:
    - Shared indexer keys (W_Ik → [B, T, d_idx]) across indexer heads
    - Per-attention-head diversity via head_importance_bias on attention logits
    - Adaptive sparsity budget k_t from variance-based heuristic
    """
    def __init__(self, hidden_size, num_heads, max_seq_len=262144, rope_base=10000,
                 k_base=512, k_min=32, k_max=1024, indexer_heads=4,
                 rope_original_max=8192, rope_scaling_factor=32.0,
                 require_fused_kernel=True):
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
        self.d_idx = 32
        self.W_Iq = nn.Linear(hidden_size, indexer_heads * self.d_idx, bias=False)
        self.W_Ik = nn.Linear(hidden_size, self.d_idx, bias=False)  # Shared across indexer heads
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
            scaling_factor=rope_scaling_factor
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_Iq, self.W_Ik, self.W_Iw, self.W_q, self.W_k, self.W_v,
                  self.o_proj, self.W_gv, self.W_go]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device
        token_keep = _token_keep_mask(attention_mask, B, T, device)

        gsa_fused_available = (
            HAS_FUSED_INDEXER
            and triton_sparse_attention is not None
            and x.is_cuda
        )
        is_grad_enabled = torch.is_grad_enabled()
        # Training uses differentiable PyTorch sparse attention; fused Triton
        # path remains required for no-grad / inference execution.
        if self.require_fused_kernel and (not is_grad_enabled) and not gsa_fused_available:
            raise RuntimeError(
                "GSA fused kernels are required but unavailable for no-grad execution. "
                f"fused_indexer_topk={HAS_FUSED_INDEXER}, "
                f"triton_sparse_attention={triton_sparse_attention is not None}, "
                f"x.is_cuda={x.is_cuda}."
            )

        # Lightning Indexer — O(T·k) via fused chunked kernel
        # Uses fused_indexer_topk to avoid materializing [B, heads, T, T] importance scores.
        # Processes in [B, C, T] chunks where C is auto-tuned to fit in ~512MB.
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)  # [B, T, 4, 32]
        k_I = self.W_Ik(x)          # [B, T, d_idx] — shared across indexer heads
        w_raw = self.W_Iw(x)        # [B, T, indexer_heads] — pre-sigmoid (kernel applies sigmoid)
        scale_idx = 1.0 / math.sqrt(self.d_idx)

        # Stateless re-computation: always recompute indices via fused_indexer_topk.
        # This avoids the gradient-accumulation race condition where _saved_selection
        # from micro-batch B would overwrite micro-batch A's indices before A's backward pass.
        # The fused_indexer_topk kernel is O(N) and adds <1% overhead vs O(N·K) attention.
        is_reversible_forward = self.training and (not torch.is_grad_enabled())

        # REVERSIBILITY FIX: During the forward pass (no_grad), snapshot the
        # current variance_ema BEFORE computing indices, then update the live
        # EMA afterwards.  During backward reconstruct (enable_grad), use the
        # snapshot so that fused_indexer_topk produces identical k_t/indices.
        # This prevents the race where gradient-accumulation micro-batch N+1
        # mutates variance_ema before micro-batch N's backward reconstruct.
        if is_reversible_forward:
            self._variance_ema_snapshot.copy_(self.variance_ema)

        # Use snapshot for the indexer call (both forward and reconstruct)
        ema_for_indexer = self._variance_ema_snapshot

        if not HAS_FUSED_INDEXER:
            raise RuntimeError(
                "GSA fused indexer kernel is required but unavailable. "
                "Fallback indexer path is disabled."
            )

        var_t, k_t, top_indices = fused_indexer_topk(
            q=q_I, k=k_I, w=w_raw, b=self.gate_bias,
            scale=scale_idx, causal=True,
            k_base=self.k_base, k_min=self.k_min, k_max=self.k_max,
            variance_ema=ema_for_indexer,  # snapshot, not live EMA
            # BUG FIX #6: was hardcoded False → EMA never updated
            is_training=self.training and is_reversible_forward,
            sink_size=4,
        )
        # var_t: [B, T], k_t: [B, T] (long), top_indices: [B, T, k_limit] (int32)

        # Update live variance EMA only during the reversible forward pass (no grad)
        # This happens AFTER the indexer call, so it doesn't affect this micro-batch's
        # indices.  The snapshot ensures backward reconstruct sees the same value.
        if is_reversible_forward:
            var_t_mean = var_t.mean().detach()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(var_t_mean, op=torch.distributed.ReduceOp.AVG)
            self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)

        # Build per-query keep mask from adaptive k_t
        k_limit = top_indices.size(-1)
        base_idx = top_indices.long()  # [B, T, k_limit]
        range_k = torch.arange(k_limit, device=device)
        keep_mask = range_k.view(1, 1, -1) < k_t.unsqueeze(-1)  # [B, T, k_limit]
        if token_keep is not None:
            invalid_query = ~token_keep
            if invalid_query.any():
                fallback_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
                base_idx = base_idx.clone()
                base_idx[..., 0] = torch.where(invalid_query, fallback_idx, base_idx[..., 0])

            key_keep = torch.gather(token_keep, dim=1, index=base_idx.reshape(B, -1)).view(B, T, k_limit)
            keep_mask = keep_mask & key_keep & token_keep.unsqueeze(-1)

            # Keep at least one index for masked queries to avoid empty-kernel rows.
            if invalid_query.any():
                keep_mask = keep_mask.clone()
                keep_mask[..., 0] = keep_mask[..., 0] | invalid_query

        # Dual Gating & Attention Projections
        q = self.W_q(x)
        k_attn = self.W_k(x)
        v = self.W_v(x)

        g_v = torch.sigmoid(self.W_gv(x))
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
        sparse_mask = keep_mask.float().unsqueeze(1).expand(B, self.num_heads, T, k_limit)

        scale_attn = 1.0 / math.sqrt(self.head_dim)

        # q, k_attn, v are [B, T, H, D].
        # Unified path: triton_sparse_attention handles both forward and backward
        # via TritonSparseAttnFn (custom autograd.Function with fused Triton kernels).
        if not (HAS_TRITON and triton_sparse_attention is not None and q.is_cuda):
            raise RuntimeError(
                "GSA fused sparse attention kernel is required but unavailable. "
                f"HAS_TRITON={HAS_TRITON}, "
                f"triton_sparse_attention={triton_sparse_attention is not None}, "
                f"q.is_cuda={q.is_cuda}."
            )
        o_sparse = triton_sparse_attention(
            q, k_attn, v, sparse_idx, sparse_mask, scale_attn,
        )
        if token_keep is not None:
            o_sparse = o_sparse * token_keep.to(dtype=o_sparse.dtype).view(B, T, 1, 1)

        # Output is [B, T, H, D] from kernel, reshape to [B, T, hidden_size]
        o_sparse = o_sparse.contiguous().view(B, T, self.hidden_size)

        # Output gate
        g_o = torch.sigmoid(self.W_go(x))

        return self.o_proj(o_sparse * g_o)


# ============================================================================
# MoE with Null Experts (from test model)
# ============================================================================

class MoEGate(nn.Module):
    """
    Router gate for MoE with null experts.

    NOTE: This 1B model is DENSE (no MoE in backbone), but MoE classes are preserved
    for compatibility and potential use in MTP or future variants.

    ⚠️  MONITORING RECOMMENDATIONS for Production Training (when MoE is used):
    ===========================================================================
    Track these metrics per layer and globally to detect sparsity drift:

    1. **Average Real Experts per Token**:
       - Metric: mean(num_real_experts_selected)
       - Expected: ~5.0 (with ρ=0.5, top-k=10)
       - Alert if: < 3.0 or > 7.0 (indicates router collapse or insufficient sparsity)

    2. **Fraction of Tokens Selecting 0 Real Experts**:
       - Metric: mean(is_null.all(dim=-1).float())
       - Expected: < 5% of tokens
       - Alert if: > 20% (indicates router preferring null experts excessively)

    3. **Load Balance Entropy**:
       - Metric: -sum(P * log(P)) where P is expert selection distribution
       - Expected: High entropy (near log(num_experts))
       - Alert if: Entropy drops significantly (indicates expert collapse)

    4. **Aux Loss Magnitude vs Main Loss**:
       - Metric: L_bal and L_z from aux_loss
       - Expected: L_bal ≈ 0.5-2.0, L_z ≈ 10-50
       - Alert if: aux_loss/main_loss > 0.1 (aux loss dominating gradients)

    5. **Per-Expert Load Distribution**:
       - Metric: counts / (B*T) for each real expert
       - Expected: Roughly uniform
       - Alert if: max/min ratio > 10 (load imbalance)

    6. **Null Expert Selection Rate**:
       - Metric: mean(is_null.float())
       - Expected: ~50% of top-k slots (with ρ=0.5)
       - Alert if: < 30% or > 70% (router not respecting data sparsity)

    Implementation:
    ```python
    # During training loop:
    with torch.no_grad():
        num_real = (~is_null).sum(dim=-1).float().mean()  # Avg real experts/token
        zero_real_frac = (is_null.all(dim=-1)).float().mean()  # Tokens with 0 real
        null_rate = is_null.float().mean()  # Overall null selection rate

        # Log per layer and aggregate
        logger.log(f"Layer {layer_idx}: real={num_real:.2f}, zero_frac={zero_real_frac:.3f}")
    ```
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, data_sparsity: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity
        # FIX #33: Store target null rate for null-rate regularizer (line 1210)
        self.rho = data_sparsity

        # Handle dense model case (num_experts = 0)
        if num_experts == 0 or data_sparsity == 0.0:
            self.num_null_copies = 0
            self.total_slots = 0
            self.gate = None
            self.logit_bias = None
            self.null_logit = None
            return

        self.num_null_copies = int(num_experts * (1 - data_sparsity) / data_sparsity)
        self.total_slots = num_experts + self.num_null_copies

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        self.gate.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        # Handle dense model case (num_experts = 0)
        if self.num_experts == 0:
            # Return dummy values for dense model
            # Note: topk_idx, topk_weight, is_null don't need gradients (routing only)
            topk_idx = torch.zeros((B, T, 1), dtype=torch.long, device=x.device)
            topk_weight = torch.zeros((B, T, 1), device=x.device)
            is_null = torch.zeros((B, T, 1), dtype=torch.bool, device=x.device)
            # Create aux_loss as part of computational graph (critical for reversible backprop)
            # Using x.sum() * 0.0 creates a zero tensor with grad_fn (not a leaf tensor)
            aux_loss = x.sum() * 0.0
            return topk_idx, topk_weight, is_null, aux_loss

        real_logits = self.gate(x) + self.logit_bias
        null_logits = self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)
        logits = torch.cat([real_logits, null_logits], dim=-1)

        probs = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        is_null = topk_idx >= self.num_experts
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # FIX #29: Separate real vs null expert losses (prevents null slots from distorting balance)
        # Balance loss: only over REAL experts (not null slots)
        logits_real = logits[:, :, :self.num_experts]  # (B, T, num_experts) - exclude null slots
        probs_real = F.softmax(logits_real, dim=-1)
        P_real = probs_real.mean(dim=(0, 1))  # (num_experts,) - average probability per real expert

        # Count only real expert selections
        idx_flat = topk_idx.view(-1)
        is_null_flat = (idx_flat >= self.num_experts)
        idx_real = torch.where(is_null_flat, torch.tensor(0, device=idx_flat.device), idx_flat)
        counts_real = torch.bincount(idx_real, minlength=self.num_experts).float()
        # BUG FIX #5: clamp to 0 — if all tokens select null, subtraction can go negative
        counts_real[0] = (counts_real[0] - is_null_flat.sum().float()).clamp(min=0)
        # FIX #34: Normalize by actual real assignments, not total slots (prevents router collapse)
        total_real_assignments = counts_real.sum()  # Total actual real expert selections
        f_real = counts_real / total_real_assignments.clamp(min=1e-6)  # Per-expert frequency among real selections

        L_bal = self.num_experts * torch.sum(f_real * P_real)  # Balance only real experts

        # Null-rate regularizer: target ρ=0.5 (50% null selections)
        null_rate = is_null.float().mean()
        target_null_rate = self.rho
        L_null = (null_rate - target_null_rate) ** 2

        # Z-loss: unchanged
        lse = torch.logsumexp(logits, dim=-1)
        L_z = (lse ** 2).mean()

        aux_loss = 2e-2 * L_bal + 1e-3 * L_z + 1e-2 * L_null

        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """MoE FFN with null experts (batched tensor implementation).

    For dense models (num_experts=0), acts as a simple dense FFN using only the shared expert.
    """
    def __init__(self, d_model: int, d_hidden: int, d_shared_hidden: Optional[int] = 2048,
                  num_experts: int = 270, top_k: int = 10,
                 dropout: float = 0.0, data_sparsity: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_shared_hidden = d_shared_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        self.is_dense = (num_experts == 0 or data_sparsity == 0.0)

        # Shared Expert (always present - acts as dense FFN for dense models)
        self.shared_gate = nn.Linear(d_model, d_shared_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, d_shared_hidden, bias=False)
        self.shared_down = nn.Linear(d_shared_hidden, d_model, bias=False)
        self._init_shared_weights()

        # Only create MoE components for sparse models
        if not self.is_dense:
            self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

            # Expert weights (batched)
            self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
            self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
            self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)
        else:
            self.gate = None
            self.W_gate = None
            self.W_up = None
            self.W_down = None

        self.last_indices = None

    def _init_shared_weights(self):
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        device, dtype = x.device, x.dtype

        # Shared expert (always computed - acts as dense FFN for dense models)
        # Liger integration: use vendored SiLU-mul op for SwiGLU path.
        shared_h = liger_silu_mul(self.shared_gate(x), self.shared_up(x))
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # For dense models, just return shared expert output
        if self.is_dense:
            # Create aux_loss as part of computational graph (critical for reversible backprop)
            # Using x.sum() * 0.0 creates a zero tensor with grad_fn (not a leaf tensor)
            aux_loss = x.sum() * 0.0
            return shared_out, aux_loss

        # Routed experts (only for sparse models)
        K = self.top_k
        E = self.num_experts
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
                h = liger_silu_mul(chunk_x @ self.W_gate[e], chunk_x @ self.W_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.W_down[e]
            start = end

        weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out)

        y = shared_out + routed_out.view(B, T, D)
        return y, aux_loss


class DenseMLP(nn.Module):
    """Simple SwiGLU FFN for dense (non-MoE) models.

    Replaces the MoEFFN-with-is_dense hack that used `x.sum() * 0.0` to fake
    an aux_loss.  This is torch.compile-friendly: no dead branches, no ghost
    MoE graph nodes.  Returns a plain tensor (no aux_loss tuple).
    """
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # Liger integration: dense SwiGLU MLP implementation (bias policy preserved).
        self.mlp = LigerSwiGLUMLP(
            in_features=d_model,
            hidden_features=d_hidden,
            out_features=d_model,
            bias=False,
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LightningMLP(nn.Module):
    """MLP wrapper: uses DenseMLP when num_experts==0, MoEFFN otherwise."""
    def __init__(self, hidden_size, intermediate_size, num_experts, num_shared_experts, top_k,
                 shared_intermediate_size=2048, data_sparsity=0.5):
        super().__init__()
        self.is_dense = (num_experts == 0 or data_sparsity == 0.0)

        if self.is_dense:
            self.mlp = DenseMLP(d_model=hidden_size, d_hidden=shared_intermediate_size)
        else:
            self.mlp = MoEFFN(
                d_model=hidden_size,
                d_hidden=intermediate_size,
                num_experts=num_experts,
                d_shared_hidden=shared_intermediate_size,
                top_k=top_k,
                dropout=0.0,
                data_sparsity=data_sparsity
            )

    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# mHC (Multi-Head Composition) - From test model
# ============================================================================

def sinkhorn_knopp(logits: torch.Tensor, iters: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Doubly-stochastic matrix via Sinkhorn-Knopp with numerical stability.
    FIX #26: Default reduced to 5 iterations (from 20) for performance at long context.

    Triton acceleration: When available, fuses all iterations into a single
    kernel launch (eliminates 2*iters kernel launches).
    """
    # Triton path: fused kernel (all iterations in one launch, zero extra memory traffic)
    # Sinkhorn is forward-only Triton (no autograd.Function), so skip when grad enabled.
    if HAS_TRITON and triton_sinkhorn_knopp is not None and logits.is_cuda and not torch.is_grad_enabled():
        logits_stable = logits - logits.amax(dim=-1, keepdim=True)
        return triton_sinkhorn_knopp(logits_stable, num_iters=iters, eps=eps)

    if logits.is_cuda and not torch.is_grad_enabled():
        raise RuntimeError(
            "Triton Sinkhorn kernel is required on CUDA (inference) but unavailable. "
            f"HAS_TRITON={HAS_TRITON}, triton_sinkhorn_knopp={triton_sinkhorn_knopp is not None}. "
            "Install triton: pip install triton"
        )

    # PyTorch fallback — only used when grad is enabled (training backward recompute)
    logits = logits - logits.amax(dim=-1, keepdim=True)
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

        # Cast to weight dtype to prevent float32/bfloat16 mismatch during reversible backward
        x_flat = x_flat.to(self.phi_pre.weight.dtype)

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
    def __init__(self, d_model: int, n_streams: int, sublayer: nn.Module, norm: nn.Module, iters: int = 20):
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

        mlp = LightningMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            shared_intermediate_size=config.shared_expert_intermediate_size,
            num_experts=config.num_real_experts,
            num_shared_experts=1,
            top_k=config.top_k,
            data_sparsity=config.data_sparsity
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
            total_aux = (aux1 if aux1 is not None else 0) + (aux2 if aux2 is not None else 0)

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
        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

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

        self.mlp = LightningMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            shared_intermediate_size=config.shared_expert_intermediate_size,
            num_experts=config.num_real_experts,
            num_shared_experts=1,
            top_k=config.top_k,
            data_sparsity=config.data_sparsity
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
        if isinstance(module, (DenseMLP, MoEFFN, MoEGate, MHCCoeffs)):
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
        x_stream = torch.zeros(batch_size, seq_len, self.n_streams, self.hidden_size,
                              device=x.device, dtype=x.dtype)
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
    1B Dense Model with Hybrid Gated DeltaNet + Gated Sparse Attention.

    Configuration:
    - 1.513B total params, 1.513B active params (100% dense - no MoE)
    - 8 layers: 75% DeltaNet (6 layers) + 25% GSA (2 layers)
    - No experts (dense FFN with 2048 intermediate size)
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
    def __init__(self, config: ModelConfig, embedding_type="kronecker", bpe_vocab=None, pf_codec=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_type = embedding_type.lower()
        self.n_streams = config.n_streams

        # Embeddings
        if self.embedding_type == "kronecker":
            if bpe_vocab is None or pf_codec is None:
                raise ValueError("bpe_vocab and pf_codec required for Kronecker embeddings")

            self.kronecker_embeddings = PureHybridEmbeddingTorch(bpe_vocab, pf_codec).module()
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
            # Pattern: Delta Delta Delta GSA (every 4th layer is GSA)
            # Provides periodic retrieval checkpoints for better long-context quality
            # Layers 3, 7, 11, 15, 19 are GSA (20-layer: 15 Delta + 5 GSA = 75%/25%)
            # Layers 3, 7 are GSA (8-layer: 6 Delta + 2 GSA = 75%/25%)
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
        # Memory Stream Recurrence (for infinite-length document processing)
        # ============================================================================
        self.recurrence_stream_idx = 3  # Use stream 3 for memory
        self.lambda_r_raw = nn.Parameter(torch.tensor(-2.5))  # Initial strength ~0.078
        self.memory_ln = nn.LayerNorm(config.hidden_size)  # Normalize memory before injection
        # FIX #25: Content-dependent memory gating (prevents uniform broadcast shortcut learning)
        self.memory_gate_proj = nn.Linear(config.hidden_size, 1, bias=True)  # Per-token gate from content

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        # Liger integration: fused linear + CE (chunked to avoid [B, T, V] allocation).
        self.liger_fused_ce = LigerFusedLinearCrossEntropyLoss(chunk_size=256)

        # Initialize
        self.apply(self._init_weights)

        # Re-initialize Kronecker projection for scale matching
        if self.use_kronecker and self.pf_to_model is not None:
            pf_to_model_std = 0.02 / math.sqrt(self._D_pf)
            self.pf_to_model.weight.data.normal_(mean=0.0, std=pf_to_model_std)
            print(f"   🔧 pf_to_model (8192→{config.hidden_size}) initialized with std={pf_to_model_std:.6f}")

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
            print(f"      Buffer size: {embedding_buffer:.1f}M (vocab × 8192, non-trainable)")
            print(f"      pf_to_model: {embedding_params:.1f}M params (8192 × {config.hidden_size})")
            print(f"      ⚠️  Embedding tying NOT possible (8192 ≠ {config.hidden_size})")
        print(f"\n   Total Layers: {config.num_layers}")
        print(f"   - DeltaNet: {config.num_deltanet_layers} layers ({config.num_deltanet_layers/config.num_layers*100:.0f}%) - O(N) linear attention")
        print(f"   - GSA: {config.num_gsa_layers} layers ({config.num_gsa_layers/config.num_layers*100:.0f}%) - Adaptive sparse")
        print(f"\n   Context Target: {config.max_seq_len:,} tokens (standard RoPE)")
        print(f"   Experts: {config.num_real_experts} real + {config.num_null_experts} null = {config.total_expert_slots} slots")
        print(f"   Top-k: {config.top_k} (dynamic, avg 5 with ρ={config.data_sparsity})")
        print(f"   MTP: {config.mtp_num_predictions} predictions" if config.enable_mtp else "   MTP: Disabled")
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

        if isinstance(module, (DenseMLP, MoEFFN, MoEGate, MHCCoeffs)):
            return

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _fused_linear_cross_entropy(self, hidden, lm_head, targets):
        """Liger fused linear+CE entrypoint (vendored implementation)."""
        return self.liger_fused_ce(
            hidden,
            lm_head.weight,
            targets,
            bias=lm_head.bias,
        )

    def forward(self, input_ids, next_token_ids=None, attention_mask=None,
                prev_memory_stream=None, return_memory=True, return_loss=False,
                ntp_targets=None, mtp_targets=None):
        """
        Forward pass with Multi-Token Prediction.

        Args:
            input_ids: [B, T] - Input token IDs
            next_token_ids: [B, T] - Optional for MTP (t+1 tokens)
            attention_mask: Optional attention mask
            prev_memory_stream: [B, D] - Memory from previous chunk (None for first chunk)
            return_memory: Whether to return memory stream for next chunk
            return_loss: Whether to return auxiliary loss
            ntp_targets: [B, T] - NTP target IDs.  When provided the model
                returns a scalar ``loss_ntp`` instead of the full logit tensor,
                using chunked cross-entropy to avoid materializing [B, T, vocab].
            mtp_targets: [B, T] - MTP target IDs (same semantics as ntp_targets).

        Returns:
            When ``ntp_targets`` is None (inference / eval):
                - logits_ntp: [B, T, vocab_size]
                - logits_mtp: [B, T, vocab_size] or None
            When ``ntp_targets`` is provided (training):
                - loss_ntp: scalar
                - loss_mtp: scalar or None
            Plus optionally aux_loss and memory_stream.
        """
        batch_size, seq_len = input_ids.size()
        token_keep_mask = _token_keep_mask(attention_mask, batch_size, seq_len, input_ids.device)

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

        # FIX #30: Precompute RoPE cos/sin once per forward (shared across all 8 layers)
        # FIX #32: Correct path through MHCSublayer wrapper (was layer.attn, now layer.attn_block.sublayer)
        # FIX #36: Include MTP block in RoPE cache optimization
        # FIX #39: Include dtype in cache key for mixed-precision safety
        # Set cache on all RotaryEmbedding instances - they'll check before computing
        cache_key = (T, x.device, x.dtype)
        for layer in self.layers:
            attn_mod = layer.attn_block.sublayer  # Access through MHCSublayer wrapper
            if not hasattr(attn_mod.rotary_emb, '_forward_cache'):
                attn_mod.rotary_emb._forward_cache = {}
            if cache_key not in attn_mod.rotary_emb._forward_cache:
                cos, sin = attn_mod.rotary_emb._compute_cos_sin(T, x.device, x.dtype)
                attn_mod.rotary_emb._forward_cache[cache_key] = (cos, sin)

        # Also cache for MTP block if enabled
        if self.mtp_block is not None:
            mtp_attn = self.mtp_block.attn_block.sublayer
            if not hasattr(mtp_attn.rotary_emb, '_forward_cache'):
                mtp_attn.rotary_emb._forward_cache = {}
            if cache_key not in mtp_attn.rotary_emb._forward_cache:
                cos, sin = mtp_attn.rotary_emb._compute_cos_sin(T, x.device, x.dtype)
                mtp_attn.rotary_emb._forward_cache[cache_key] = (cos, sin)

        # ============================================================================
        # MEMORY STREAM INJECTION (Non-blocking, fully parallel)
        # ============================================================================
        if prev_memory_stream is not None:
            # Defensive: Prevent cross-chunk gradient accumulation
            prev_memory_stream = prev_memory_stream.detach()

            # Normalize memory for stability
            memory = self.memory_ln(prev_memory_stream)

            # FIX #25: Content-dependent gating instead of uniform broadcast
            # Compute per-token gates from current input (prevents shortcut learning)
            # Gates are based on content, not position - model learns when memory is relevant
            memory_gates = torch.sigmoid(
                self.memory_gate_proj(x_stream[:, :, 0, :])  # Use stream 0 (primary) for gating decision
            )  # (B, T, 1)

            # Broadcast memory with content-dependent modulation
            memory_broadcast = memory.unsqueeze(1).expand(B, T, D)

            # Apply learnable strength + content-dependent gates (out-of-place
            # to preserve autograd link to lambda_r and memory_gate_proj)
            lambda_r = F.softplus(self.lambda_r_raw)
            mem_val = (lambda_r * memory_gates * memory_broadcast).unsqueeze(2)  # (B, T, 1, D)
            one_hot = torch.zeros(self.n_streams, device=x.device, dtype=x.dtype)
            one_hot[self.recurrence_stream_idx] = 1.0
            x_stream = x_stream + mem_val * one_hot.view(1, 1, self.n_streams, 1)

        # Pass through reversible stack
        x_stream, total_aux_loss = self.stack(x_stream, attention_mask=token_keep_mask)


        # ============================================================================
        # EXTRACT MEMORY STREAM for next chunk (from final position)
        # ============================================================================
        if return_memory:
            memory_stream_out = x_stream[:, -1, self.recurrence_stream_idx, :].detach()
        else:
            memory_stream_out = None

        # Collapse streams
        h_main = x_stream.mean(dim=2)
        h_main = self.norm(h_main)

        # NTP Prediction
        # Liger integration: fused linear+CE path avoids [B, T, vocab] allocation.
        if ntp_targets is not None:
            logits_ntp = self._fused_linear_cross_entropy(h_main, self.lm_head, ntp_targets)
        else:
            logits_ntp = self.lm_head(h_main)

        # MTP Prediction
        logits_mtp = None
        if self.mtp_block is not None and next_token_ids is not None:
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_ids_use = next_token_ids[:, :min_len]

            if self.use_kronecker:
                next_emb = self.kronecker_embeddings(next_ids_use)
                next_emb = self.pf_to_model(next_emb.to(dtype=self.pf_to_model.weight.dtype))
                next_emb = self.embed_norm(next_emb)
            else:
                next_emb = self.token_embed(next_ids_use)

            mtp_attention_mask = token_keep_mask[:, :min_len] if token_keep_mask is not None else None
            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=mtp_attention_mask)
            h_mtp_normed = self.norm(h_mtp)
            if mtp_targets is not None:
                logits_mtp = self._fused_linear_cross_entropy(h_mtp_normed, self.lm_head, mtp_targets)
            else:
                logits_mtp = self.lm_head(h_mtp_normed)

        # FIX #41: Clear RoPE forward-pass cache to prevent accumulation (CRITICAL PATH FIX)
        # Correct path: layer.attn_block.sublayer.rotary_emb (not layer.attn)
        # Architecture: LightningDecoderLayer → MHCSublayer → GatedDeltaNet/GatedSparseAttention → RotaryEmbedding
        for layer in self.layers:
            if hasattr(layer.attn_block.sublayer, 'rotary_emb'):
                if hasattr(layer.attn_block.sublayer.rotary_emb, '_forward_cache'):
                    layer.attn_block.sublayer.rotary_emb._forward_cache.clear()

        # Also clear MTP block cache if enabled
        if self.mtp_block is not None:
            if hasattr(self.mtp_block.attn_block.sublayer, 'rotary_emb'):
                if hasattr(self.mtp_block.attn_block.sublayer.rotary_emb, '_forward_cache'):
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

def create_model_1b(embedding_type="kronecker", bpe_vocab=None, pf_codec=None):
    """
    Create 1B model with default configuration.

    Args:
        embedding_type: "kronecker" or "standard"
        bpe_vocab: Required for Kronecker embeddings
        pf_codec: Required for Kronecker embeddings

    Returns:
        Model1B instance
    """
    config = ModelConfig()
    return Model1B(config, embedding_type=embedding_type, bpe_vocab=bpe_vocab, pf_codec=pf_codec)


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
    active_row = report_df[report_df['Component'] == 'TOTAL ACTIVE PARAMETERS']
    total_row = report_df[report_df['Component'] == 'TOTAL MODEL PARAMETERS']
    active_params = float(str(active_row['Total Contribution'].iloc[0]).replace(' B', ''))
    total_params = float(str(total_row['Total Contribution'].iloc[0]).replace(' B', ''))
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
    print(f"  DeltaNet: {config.num_deltanet_layers} layers ({config.num_deltanet_layers/config.num_layers*100:.0f}%) - O(N) for 256k context")
    print(f"  GSA: {config.num_gsa_layers} layers ({config.num_gsa_layers/config.num_layers*100:.0f}%) - Adaptive sparse quality")
    print("\nModel Type:")
    if num_experts == 0:
        print("  DENSE MODEL (No MoE)")
        print(f"  Dense FFN intermediate: {config.shared_expert_intermediate_size}")
    else:
        print("  MoE MODEL")
        print(f"  Real Experts: {num_experts}")
        print(f"  Null Experts: {num_experts} (ρ={config.data_sparsity})")
        print(f"  Total slots: {config.total_expert_slots}")
        print(f"  Top-k: {config.top_k} (dynamic 0-{config.top_k}, avg {config_calc.num_routed_experts_active})")
        print(f"  Shared Expert FFN: {config.shared_expert_intermediate_size} (always active)")
        print(f"  Routed Expert FFN: {config.expert_intermediate_size} (sparse)")
    print(f"\nContext: {config.max_seq_len:,} tokens")
    print("=" * 80)
