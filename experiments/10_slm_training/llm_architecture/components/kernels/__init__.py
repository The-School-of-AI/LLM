"""
Optimized Triton kernels for Gated Sparse Attention.

These kernels provide efficient GPU implementations for:
1. Sparse attention computation (avoiding O(L^2) memory)
2. Gated indexer computation
3. Fused gated attention operations

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
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from .triton_sparse_attn import (
        triton_sparse_attention,
        pytorch_sparse_attention,
    )
    from .triton_indexer import (
        triton_gated_indexer,
    )
else:
    # Provide fallback functions that raise helpful errors
    def triton_sparse_attention(*args, **kwargs):
        raise ImportError(
            "Triton is not installed. Install with: pip install triton"
        )

    def triton_gated_indexer(*args, **kwargs):
        raise ImportError(
            "Triton is not installed. Install with: pip install triton"
        )

    # Import PyTorch fallback
    from .triton_sparse_attn import pytorch_sparse_attention

__all__ = [
    "HAS_TRITON",
    "triton_sparse_attention",
    "triton_gated_indexer",
    "pytorch_sparse_attention",
]
