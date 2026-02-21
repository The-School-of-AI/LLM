"""
Kernel library for Test 13 (GSA-only).

Centralized Triton kernels and PyTorch fallbacks for:
- Sparse Attention (GSA)
- Gated Lightning Indexer (GSA)
- Fused Sinkhorn-Knopp (mHC routing)
- Fused RMSNorm (all layers)
"""

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from .triton_sparse_attn import (
    triton_sparse_attention,
    pytorch_sparse_attention,
)

from .triton_indexer import (
    triton_gated_indexer,
    pytorch_gated_indexer,
)

from .triton_indexer_streaming import (
    fused_indexer_topk,
    streaming_indexer_variance,
)

from .triton_sinkhorn import (
    triton_sinkhorn_knopp,
    pytorch_sinkhorn_knopp,
)

from .triton_rmsnorm import (
    triton_rmsnorm,
    pytorch_rmsnorm,
    TritonRMSNorm,
)

from .moe_grouped_gemm import (
    HAS_MOE_GROUPED_GEMM,
    moe_grouped_gemm,
)

__all__ = [
    "HAS_TRITON",
    "HAS_MOE_GROUPED_GEMM",
    "triton_sparse_attention",
    "pytorch_sparse_attention",
    "triton_gated_indexer",
    "pytorch_gated_indexer",
    "triton_sinkhorn_knopp",
    "pytorch_sinkhorn_knopp",
    "triton_rmsnorm",
    "pytorch_rmsnorm",
    "TritonRMSNorm",
    "fused_indexer_topk",
    "streaming_indexer_variance",
    "moe_grouped_gemm",
]
