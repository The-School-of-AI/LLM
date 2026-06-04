"""
Kernel library for Test 14 (DeltaNet + GSA, no fused CE).

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

from .moe_grouped_gemm import HAS_MOE_GROUPED_GEMM, moe_grouped_gemm
from .triton_delta_entrance import fused_delta_entrance
from .triton_deltanet_post_train import triton_deltanet_post_fused
from .triton_fused_proj import (
    fused_beta_gk_proj_triton,
    fused_dual_proj_sigmoid,
    fused_multi_proj,
    fused_qkv_proj,
    fused_qkvg_proj,
)
from .triton_fused_rope import fused_qk_rope
from .triton_indexer import pytorch_gated_indexer, triton_gated_indexer
from .triton_indexer_streaming import fused_indexer_topk
from .triton_mhc_stream import fused_mhc_collapse, fused_mhc_expand_residual
from .triton_moe_fused_gate_up import fused_moe_gate_up_silu
from .triton_moe_grouped_gemm import triton_grouped_gemm
from .triton_moe_weighted_scatter import fused_weighted_scatter_add
from .triton_rmsnorm import (
    TritonRMSNorm,
    pytorch_rmsnorm,
    triton_rmsnorm,
    triton_rmsnorm_fwd_only,
)
from .triton_sigmoid_gate import fused_scaled_sigmoid, fused_sigmoid_gate
from .triton_sinkhorn import pytorch_sinkhorn_knopp, triton_sinkhorn_knopp
from .triton_sparse_attn import triton_sparse_attention_v2 as triton_sparse_attention

# --- Phase 2: New fused kernels ---


# --- Phase 3: MoE expert kernels ---


__all__ = [
    "HAS_TRITON",
    "HAS_MOE_GROUPED_GEMM",
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
    "moe_grouped_gemm",
    "fused_delta_entrance",
    # Phase 2: New fused kernels
    "fused_multi_proj",
    "fused_qkvg_proj",
    "fused_qkv_proj",
    "fused_dual_proj_sigmoid",
    "fused_beta_gk_proj_triton",
    "fused_mhc_collapse",
    "fused_mhc_expand_residual",
    "triton_deltanet_post_fused",
    "fused_qk_rope",
    "fused_sigmoid_gate",
    "fused_scaled_sigmoid",
    # Phase 3: MoE expert kernels
    "triton_grouped_gemm",
    "fused_moe_gate_up_silu",
    "fused_weighted_scatter_add",
]
