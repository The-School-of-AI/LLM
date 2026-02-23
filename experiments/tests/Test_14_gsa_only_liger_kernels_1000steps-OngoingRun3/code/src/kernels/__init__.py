"""
Kernel library for Test 14 (DeltaNet + GSA, no fused CE).

Centralized Triton kernels and PyTorch fallbacks for:
- Sparse Attention (GSA)
- Gated Lightning Indexer (GSA)
- Fused Sinkhorn-Knopp (mHC routing)
- Fused RMSNorm (all layers)
- DeltaNet fla wrapper (fused linear attention)
"""

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# fla (flash-linear-attention) for DeltaNet
try:
    from .fla_deltanet import HAS_FLA, fla_gated_delta_rule
except (ImportError, AttributeError):
    HAS_FLA = False
    fla_gated_delta_rule = None

# Swap to V2 (key-major dK/dV backward, no atomics)
from .triton_sparse_attn_v2 import triton_sparse_attention_v2 as triton_sparse_attention

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
    "HAS_FLA",
    "fla_gated_delta_rule",
    "HAS_MOE_GROUPED_GEMM",
    "triton_sparse_attention",
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
