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

from .triton_sparse_attn import (
    triton_sparse_attention_v2 as triton_sparse_attention,
)

from .triton_indexer import (
    triton_gated_indexer,
    pytorch_gated_indexer,
)

from .triton_indexer_streaming import (
    fused_indexer_topk,
)

from .triton_sinkhorn import (
    triton_sinkhorn_knopp,
    pytorch_sinkhorn_knopp,
)

from .triton_rmsnorm import (
    triton_rmsnorm,
    triton_rmsnorm_fwd_only,
    pytorch_rmsnorm,
    TritonRMSNorm,
)

from .moe_grouped_gemm import (
    HAS_MOE_GROUPED_GEMM,
    moe_grouped_gemm,
)

from .triton_delta_entrance import (
    fused_delta_entrance,
)

from .triton_silu_mul import (
    fused_silu_mul,
)

from .triton_rope import (
    fused_rope,
)

from .fused_swiglu import (
    FusedSwiGLUForward,
    _FusedSwiGLUFunc,
)

from .fused_moe_expert import (
    fused_moe_expert_forward,
    has_fused_moe_expert_triton,
)

from .fused_qkv_proj import (
    fused_qkv_proj_forward,
    has_fused_qkv_proj,
    fused_qkvg_proj_forward,
    has_fused_qkvg_proj,
    fused_o_gate_proj_forward,
    has_fused_o_gate_proj,
)

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
    "fused_silu_mul",
    "fused_rope",
    "FusedSwiGLUForward",
    "fused_moe_expert_forward",
    "has_fused_moe_expert_triton",
    "fused_qkv_proj_forward",
    "has_fused_qkv_proj",
    "fused_qkvg_proj_forward",
    "has_fused_qkvg_proj",
    "fused_o_gate_proj_forward",
    "has_fused_o_gate_proj",
]
