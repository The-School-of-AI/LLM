# Kernels Directory

Triton fused kernels and PyTorch fallbacks for the Hybrid DeltaNet + GSA architecture.

## Installation

```bash
pip install triton   # Triton kernels (optional — PyTorch fallbacks used if missing)
pip install fla      # flash-linear-attention for DeltaNet fused kernel
```

## Integration into Other Branches

1. **Copy the `kernels/` directory** into your branch root (same level as `model_1b.py`)

2. **Add imports** at the top of your model file:
```python
from kernels import (
    HAS_TRITON, HAS_FLA,
    triton_sparse_attention, pytorch_sparse_attention,
    triton_gated_indexer, pytorch_gated_indexer,
    triton_sinkhorn_knopp, pytorch_sinkhorn_knopp,
    triton_rmsnorm, pytorch_rmsnorm, TritonRMSNorm,
    fla_gated_delta_rule,
)
```

3. **Update flag imports** in training scripts:
```python
# Old: from model_1b import HAS_TRITON, HAS_FLA
from kernels import HAS_TRITON, HAS_FLA
```

## Kernel Reference

### GSA Sparse Attention (`triton_sparse_attn.py`)
- **What**: Computes attention only over selected token indices (from indexer)
- **Complexity**: O(T*k) instead of O(T^2)
- **Usage**: `triton_sparse_attention(q, k, v, indices, mask, scale)`
- **Layout**: q/k/v `[B, T, H, D]`, indices/mask `[B, H, T, k_sel]`
- **Fallback**: `pytorch_sparse_attention(...)` — chunked gather-based

### GSA Gated Indexer (`triton_indexer.py`)
- **What**: Computes importance scores for adaptive sparse token selection
- **Key optimization**: Shared-K design avoids (B, H, T, T) intermediates
  - Old: Q=[B,T,H,d], K=[B,T,H,d] -> scores=[B,H,T,T] (~6 GB at T=4096)
  - New: Q=[B,T,H,d], K=[B,T,d]   -> scores=[B,T,T]   (~134 MB at T=4096)
- **Usage**: `triton_gated_indexer(q, k, w, bias, scale, causal=True)`
- **Fallback**: `pytorch_gated_indexer(...)` — einsum-based

### Sinkhorn-Knopp (`triton_sinkhorn.py`)
- **What**: Projects matrices onto Birkhoff polytope (doubly stochastic)
- **Used by**: mHC (Multi-Head Composition) routing coefficients
- **Key optimization**: All iterations in single kernel launch (40 launches -> 1)
- **Usage**: `triton_sinkhorn_knopp(H, num_iters=20, eps=1e-8)`
- **Fallback**: `pytorch_sinkhorn_knopp(...)` — iterative row/col normalization

### RMSNorm (`triton_rmsnorm.py`)
- **What**: Fused RMSNorm with optional residual addition
- **Key optimization**: 50% less memory bandwidth, 3-4x fewer kernel launches
- **Usage**: `triton_rmsnorm(x, weight, eps, residual)` or `TritonRMSNorm(dim)`
- **Fallback**: `pytorch_rmsnorm(...)` — standard variance+rsqrt+mul

### DeltaNet fla Wrapper (`fla_deltanet.py`)
- **What**: Wraps fla library's `chunk_gated_delta_rule` kernel
- **Handles**: float32 upcast, layout conversion, alpha->log(alpha), D residual
- **Usage**: `fla_gated_delta_rule(q, k, v, alpha, beta, D, num_heads)`
- **Requirements**: `pip install fla`

## Reversibility Safety

All kernels are **pure functions** (no side effects, no saved state). They are called
inside `force()` which is invoked by the reversible midpoint integrator. Safety guarantees:

1. **Deterministic**: Same inputs always produce same outputs
2. **No saved state**: Kernels don't modify module attributes
3. **`_saved_selection` protocol unchanged**: GSA topk indices are still saved/replayed
   by the model code, not by the kernels
4. **Sinkhorn determinism**: Triton kernel runs fixed iterations (no early stopping),
   PyTorch fallback has GPU-only convergence tracking (no host sync)

## Enabling/Disabling Triton

- Triton kernels are auto-detected via `HAS_TRITON` flag
- If Triton is not installed, PyTorch fallbacks are used automatically
- To force PyTorch fallback even with Triton installed, pass `use_triton_kernel=False` to GSA
- fla kernels require `pip install fla` and `delta_recurrence_mode="fla"` in config

## File Structure

```
kernels/
  __init__.py            # Central imports + HAS_TRITON/HAS_FLA flags
  triton_sparse_attn.py  # GSA sparse attention kernel
  triton_indexer.py      # GSA gated lightning indexer kernel
  triton_sinkhorn.py     # Fused Sinkhorn-Knopp kernel
  triton_rmsnorm.py      # Fused RMSNorm kernel + TritonRMSNorm module
  fla_deltanet.py        # fla wrapper for DeltaNet recurrence
  README.md              # This file
```
