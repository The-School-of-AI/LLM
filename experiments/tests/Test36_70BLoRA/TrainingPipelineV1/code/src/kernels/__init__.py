"""
Kernel library for 70B MoE LoRA training.

Centralized Triton kernels and PyTorch fallbacks from Test28 (pre-V2, production-validated).

Phase 1: Foundation kernels (Sparse Attention, Indexer, RMSNorm, Sinkhorn, DeltaNet)
Phase 2: Fused projection kernels (RoPE, Sigmoid Gate, Multi-Proj, mHC Stream, DeltaNet Post)
Phase 3: MoE expert kernels (Grouped GEMM V1, Fused Gate+Up+SiLU, Weighted Scatter)
Phase 4: LoRA kernels (Fused LoRA Linear, Fused LoRA + Grouped GEMM)
Phase 5: MoE gate kernel (Fused Softmax-TopK-Index)
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


# --- Phase 2: Fused kernels ---

from .triton_fused_proj import (
    fused_multi_proj,
    fused_qkvg_proj,
    fused_qkv_proj,
    fused_dual_proj_sigmoid,
    fused_beta_gk_proj_triton,
)

from .triton_mhc_stream import (
    fused_mhc_collapse,
    fused_mhc_expand_residual,
)

from .triton_deltanet_post_train import (
    triton_deltanet_post_fused,
)

from .triton_fused_rope import (
    fused_qk_rope,
)

from .triton_sigmoid_gate import (
    fused_sigmoid_gate,
    fused_scaled_sigmoid,
)


# --- Phase 3: MoE expert kernels (V1, pre-adaptive, from Test28) ---

from .triton_moe_grouped_gemm import (
    triton_grouped_gemm,
)

from .triton_moe_fused_gate_up import (
    fused_moe_gate_up_silu,
)

from .triton_moe_weighted_scatter import (
    fused_weighted_scatter_add,
)


# --- Phase 4: LoRA kernels ---

from .fused_lora_linear import (
    fused_lora_linear,
    FusedLoRALinear,
)

from .fused_lora_grouped_gemm import (
    fused_lora_grouped_gemm,
    fused_lora_gate_up_silu,
)


# --- Phase 5: MoE gate kernel ---

from .triton_moe_softmax_topk import (
    fused_softmax_topk,
)


# --- Phase 6: NF4 Quantization kernels ---

from .nf4_quantize import (
    NF4QuantConfig,
    NF4Parameter,
    NF4_LEVELS,
    quantize_tensor_nf4,
)

from .triton_nf4_grouped_gemm import (
    nf4_lora_grouped_gemm,
    pytorch_nf4_lora_grouped_gemm,
)

# --- Phase 7: Manual backward LoRA ---

from .manual_lora_backward import (
    ManualLoRALinear,
    ManualLoRALinearFn,
)


# --- Upgraded Cross Entropy: Cut CE (V-tiled, no [BT,V] materialization) ---

from .cut_cross_entropy import (
    CutCrossEntropyLoss,
)


__all__ = [
    "HAS_TRITON",
    "HAS_MOE_GROUPED_GEMM",
    # Phase 1
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
    # Phase 2
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
    # Phase 3: MoE expert kernels (V1)
    "triton_grouped_gemm",
    "fused_moe_gate_up_silu",
    "fused_weighted_scatter_add",
    # Phase 4: LoRA kernels
    "fused_lora_linear",
    "FusedLoRALinear",
    "fused_lora_grouped_gemm",
    "fused_lora_gate_up_silu",
    # Phase 5: MoE gate kernel
    "fused_softmax_topk",
    # Phase 6: NF4 Quantization
    "NF4QuantConfig",
    "NF4Parameter",
    "NF4_LEVELS",
    "quantize_tensor_nf4",
    "nf4_lora_grouped_gemm",
    "pytorch_nf4_lora_grouped_gemm",
    # Phase 7: Manual backward LoRA
    "ManualLoRALinear",
    "ManualLoRALinearFn",
    # Upgraded Cross Entropy
    "CutCrossEntropyLoss",
]
