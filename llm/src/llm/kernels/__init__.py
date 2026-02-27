"""
Kernel library for Test 14 (DeltaNet + GSA, no fused CE).

Centralized Triton kernels and PyTorch fallbacks for:
- Sparse Attention (GSA)
- Gated Lightning Indexer (GSA)
- Fused Sinkhorn-Knopp (mHC routing)
- Fused RMSNorm (all layers)
- DeltaNet fla wrapper (fused linear attention)
"""

from .compat import HAS_FLA, HAS_TRITON
from .fla_deltanet import fla_gated_delta_rule
from .triton_indexer import (
    pytorch_gated_indexer,
    triton_gated_indexer,
)
from .triton_indexer_streaming import (
    fused_indexer_topk,
    streaming_indexer_variance,
)
from .triton_rmsnorm import (
    TritonRMSNorm,
    pytorch_rmsnorm,
    triton_rmsnorm,
)
from .triton_sinkhorn import (
    pytorch_sinkhorn_knopp,
    triton_sinkhorn_knopp,
)
from .triton_sparse_attn import triton_sparse_attention_v2 as triton_sparse_attention

__all__ = [
    "HAS_FLA",
    "HAS_TRITON",
    "fla_gated_delta_rule",
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
]
