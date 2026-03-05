"""
Kernel library for Test 14 (DeltaNet + GSA, no fused CE).

Centralized Triton kernels and PyTorch fallbacks for:
- Sparse Attention (GSA)
- Gated Lightning Indexer (GSA)
- Fused Sinkhorn-Knopp (mHC routing)
- Fused RMSNorm (all layers)
- DeltaNet fla wrapper (fused linear attention)
"""

from .compat import HAS_TRITON
from .triton_cross_entropy import FusedLinearCrossEntropyLoss
from .triton_delta_entrance import fused_delta_entrance
from .triton_indexer import pytorch_gated_indexer, triton_gated_indexer
from .triton_indexer_streaming import fused_indexer_topk, streaming_indexer_variance
from .triton_rmsnorm import (
    TritonRMSNorm,
    pytorch_rmsnorm,
    triton_rmsnorm,
    triton_rmsnorm_fwd_only,
)
from .triton_sinkhorn import pytorch_sinkhorn_knopp, triton_sinkhorn_knopp
from .triton_sparse_attn import triton_sparse_attention_v2 as triton_sparse_attention

__all__ = [
    "HAS_TRITON",
    "triton_sparse_attention",
    "triton_gated_indexer",
    "pytorch_gated_indexer",
    "triton_sinkhorn_knopp",
    "pytorch_sinkhorn_knopp",
    "triton_rmsnorm",
    "triton_rmsnorm_fwd_only",
    "pytorch_rmsnorm",
    "TritonRMSNorm",
    "fused_indexer_topk",
    "streaming_indexer_variance",
    "fused_delta_entrance",
    "FusedLinearCrossEntropyLoss",
]
