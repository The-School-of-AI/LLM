# V2 Optimization: Native Precision Loading

## What V2 Changes

**One change**: Remove `.to(tl.float32)` casts when loading Q, K, V, dO inside Triton kernels.

### V1 (Rohan's baseline)
```python
# Loads bf16/fp16 data, immediately casts to fp32 (4 bytes per element)
k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
```

### V2 (native precision)
```python
# Loads bf16/fp16 data, keeps in native precision (2 bytes per element)
k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)
```

## Why This Helps

The kernel is **memory-bandwidth bound** — the bottleneck is loading K/V data from GPU memory, not the arithmetic.

- V1 casts every element to fp32 before processing → **4 bytes per element loaded**
- V2 keeps elements in bf16/fp16 → **2 bytes per element loaded = 50% less bandwidth**
- `tl.sum(q * k)` with bf16 inputs still produces fp32 results (Triton auto-promotes)
- Softmax accumulators (`m_i`, `l_i`, `acc`) remain fp32 for numerical stability

## What Stays the Same

Everything else is identical to V1:
- All safety features (NaN prevention, bounds checking, input sanitization)
- Kernel structure (online softmax, atomic scatter for dK/dV)
- Block sizes, grid dimensions
- Public API

## Files

| File | Description |
|------|-------------|
| `triton_sparse_attn.py` | Rohan's V1 (unchanged baseline) |
| `triton_sparse_attn_v2.py` | V1 + native precision loading |
| `test_sparse_attn_correctness.py` | Correctness: V1 vs V2 vs PyTorch |
| `benchmark_sparse_attn.py` | 4-way benchmark (Dense/Sparse/V1/V2) |
