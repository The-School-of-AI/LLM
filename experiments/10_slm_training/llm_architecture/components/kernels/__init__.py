"""
Optimized Triton kernels for Gated Sparse Attention.

These kernels provide efficient GPU implementations for:
1. Sparse attention computation (avoiding O(L^2) memory)
2. Gated indexer computation
3. Fused gated attention operations
4. Fused Sinkhorn-Knopp (all iterations in 1 kernel launch)
5. Fused RMSNorm + SiLU + Gate (3 ops -> 1 kernel launch)
6. RMSNorm with optional residual fusion

Usage:
    from components.kernels import triton_sparse_attention, HAS_TRITON

    if HAS_TRITON:
        output, lse = triton_sparse_attention(q, k, v, indices, mask)
    else:
        # Fall back to PyTorch implementation
        output = pytorch_sparse_attention(q, k, v, indices, mask)
"""

# Check if Triton is available
try:
    __import__("triton")
    __import__("triton.language")

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from .triton_fused_norm_gate import (
        FusedRMSNormSiLUGate,
        pytorch_fused_norm_silu_gate,
        triton_fused_norm_silu_gate,
    )
    from .triton_indexer import triton_gated_indexer
    from .triton_sinkhorn import pytorch_sinkhorn_knopp, triton_sinkhorn_knopp
    from .triton_sparse_attn import pytorch_sparse_attention, triton_sparse_attention
else:
    # Provide fallback functions that raise helpful errors
    def triton_sparse_attention(*args, **kwargs):
        raise ImportError("Triton is not installed. Install with: pip install triton")

    def triton_gated_indexer(*args, **kwargs):
        raise ImportError("Triton is not installed. Install with: pip install triton")

    def triton_sinkhorn_knopp(*args, **kwargs):
        raise ImportError("Triton is not installed. Install with: pip install triton")

    def triton_fused_norm_silu_gate(*args, **kwargs):
        raise ImportError("Triton is not installed. Install with: pip install triton")

    # Import PyTorch fallbacks
    from .triton_fused_norm_gate import (
        FusedRMSNormSiLUGate,
        pytorch_fused_norm_silu_gate,
    )
    from .triton_sinkhorn import pytorch_sinkhorn_knopp
    from .triton_sparse_attn import pytorch_sparse_attention

__all__ = [
    "HAS_TRITON",
    "triton_sparse_attention",
    "triton_gated_indexer",
    "pytorch_sparse_attention",
    "triton_sinkhorn_knopp",
    "pytorch_sinkhorn_knopp",
    "triton_fused_norm_silu_gate",
    "pytorch_fused_norm_silu_gate",
    "FusedRMSNormSiLUGate",
]
