"""
Kernel Library for Hybrid DeltaNet + GSA Architecture
=====================================================

Centralized Triton kernels and PyTorch fallbacks for:
- Sparse Attention (GSA)
- Gated Lightning Indexer (GSA)
- Fused Sinkhorn-Knopp (mHC routing)
- Fused RMSNorm (all layers)
- DeltaNet fla wrapper (fused linear attention)

Usage:
    from kernels import (
        HAS_TRITON, HAS_FLA,
        triton_sparse_attention, pytorch_sparse_attention,
        triton_gated_indexer, pytorch_gated_indexer,
        fused_indexer_topk, streaming_indexer_variance,
        triton_sinkhorn_knopp, pytorch_sinkhorn_knopp,
        triton_rmsnorm, pytorch_rmsnorm, TritonRMSNorm,
        fla_gated_delta_rule,
    )
"""

# ── Triton availability flag ──────────────────────────────────────────
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ── fla availability flag ─────────────────────────────────────────────
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _check_fla
    HAS_FLA = True
    del _check_fla
except ImportError:
    HAS_FLA = False


# ── Sparse Attention ──────────────────────────────────────────────────
from .triton_sparse_attn import (
    triton_sparse_attention,
    pytorch_sparse_attention,
)

# ── Gated Lightning Indexer ───────────────────────────────────────────
from .triton_indexer import (
    triton_gated_indexer,
    pytorch_gated_indexer,
)

# ── Streaming Indexer (memory-efficient variance + chunked topk) ─────
from .triton_indexer_streaming import (
    fused_indexer_topk,
    streaming_indexer_variance,
)

# ── Sinkhorn-Knopp ───────────────────────────────────────────────────
from .triton_sinkhorn import (
    triton_sinkhorn_knopp,
    pytorch_sinkhorn_knopp,
)

# ── RMSNorm ───────────────────────────────────────────────────────────
from .triton_rmsnorm import (
    triton_rmsnorm,
    pytorch_rmsnorm,
    TritonRMSNorm,
)

# ── fla DeltaNet ──────────────────────────────────────────────────────
from .fla_deltanet import (
    fla_gated_delta_rule,
)

__all__ = [
    "HAS_TRITON",
    "HAS_FLA",
    "triton_sparse_attention",
    "pytorch_sparse_attention",
    "triton_gated_indexer",
    "pytorch_gated_indexer",
    "triton_sinkhorn_knopp",
    "pytorch_sinkhorn_knopp",
    "triton_rmsnorm",
    "pytorch_rmsnorm",
    "TritonRMSNorm",
    "fla_gated_delta_rule",
    "fused_indexer_topk",
    "streaming_indexer_variance",
]
