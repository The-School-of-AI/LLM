# CHANGELOG

All performance fixes are tagged `FIX-PERF-XX` in the code for traceability.

---

## 2026-02-22 — Throughput Bottleneck Fixes (14k → expected 60k–100k tok/s)

Root cause analysis was performed against `T14-4096-B32 LONG.log`.
The ~14k tok/s throughput (vs expected 500k+) was caused by three compounding issues:

---

### FIX-PERF-01 · DeepSpeed: Remove CPU optimizer offload + ZeRO-2 → ZeRO-1

**Files changed:** `deepspeed/zero-2-dense-bf16-test14-1000steps.json`

**Problem:**
- `offload_optimizer.device = "cpu"` was enabled, meaning **ALL Adam optimizer states**
  (1st moment, 2nd moment, params = ~3× model size ≈ 10 GB) lived on CPU RAM.
- Every optimizer step required PCIe transfers:
  gradients GPU→CPU, Adam update on CPU, weights CPU→GPU.
  At 1.65 B params (BF16 = 3.3 GB), that's ≥ 6.6 GB of PCIe traffic per step.
  With p4de NVLink bandwidth ~64 GB/s host-to-GPU, this adds **~100–300 ms per step**.
- ZeRO-2 shards gradients + optimizer states. With 80 GB × 8 = 640 GB available GPU
  memory and only a ~10 GB Adam state budget, ZeRO-2 is unnecessary overhead.

**Fix:**
- Removed `offload_optimizer` entirely — optimizer states now stay on GPU.
- Downgraded ZeRO stage `2 → 1` (shards optimizer states across ranks; no gradient
  partitioning overhead).
- Removed unused `activation_checkpointing` block (all fields were `false`).

**Expected impact:** ~10–20% step time reduction (biggest wins with many gradient-accumulation steps or large batch sizes).

---

### FIX-PERF-02 · ModelConfig: Lower GSA sparsity budget (k_base 512→128, k_max 1024→256)

**Files changed:** `code/src/models/recurrence_model_1b.py` (`ModelConfig`)

**Problem:**
- `gsa_k_base = 512`, `gsa_k_max = 1024`.
- At T=4096, k_base=512 means each query attends to **12.5–25% of the sequence**.
  This defeats the purpose of sparse attention at short contexts.
- Higher k_limit directly scales O(B·H·T·k_sel) work in:
  - The `fused_indexer_topk` score computation (T*k per query)
  - The `triton_sparse_attn` forward kernel inner loop
  - The `scatter_add` / old atomic dK/dV backward
- k_max=1024 is appropriate for the 256k context target, NOT for 4096 training steps.

**Fix:**
- `gsa_k_base = 128` — each query now attends to ~3% of sequence at T=4096.
- `gsa_k_max = 256` — hard cap limits backward scatter work.
- `gsa_k_min = 32` unchanged — ensures meaningful coverage for short queries.

**Expected impact:** ~4× reduction in sparse attention backward work.
At 256k context, these values should be restored to k_base=512, k_max=1024.

---

### FIX-PERF-03 · Sparse attention kernel: BLOCK_K + atomic-free dK/dV backward

**Files changed:** `code/src/kernels/triton_sparse_attn.py`

#### FIX-PERF-03a · BLOCK_K: 64 → 128 (forward kernel)

**Problem:** `BLOCK_K = triton.next_power_of_2(min(64, k_sel))`.
At k_sel=512 (old) or k_sel=256 (new), BLOCK_K was 64. On A100 with BF16, larger tile
sizes (128+) have better L1/L2 cache reuse for the K/V gather pattern.

**Fix:** `BLOCK_K = triton.next_power_of_2(min(128, k_sel))`

**Expected impact:** Minor (~5–10% forward kernel speedup for the sparse attn op).

#### FIX-PERF-03b · dK/dV backward: atomic scatter → PyTorch scatter_add_ (major)

**Problem:** The original Triton dK/dV backward kernel used `tl.atomic_add`:
```
Grid: (B*H, T) = 4 × 16 × 4096 = 262,144 programs
Each program: k_sel=512 random atomic writes to dK, dV
Total atomics: 262,144 × 512 = ~134 million atomic_add calls per backward pass
```
The access pattern is completely non-coalesced (different queries scatter to different
random key positions), causing severe **L2 cache thrashing** on the A100.
This was the primary cause of the 496ms GSA layer latency vs the expected ~80ms.

**Fix:** Replaced the atomic Triton dKdV kernel with vectorised PyTorch `scatter_add_`:
1. Reorder Q/K/V/dO to head-major `[B, H, T, D]` layout.
2. Gather selected keys/values: `k_sel_t = k_bh[b_idx, h_idx, indices]` → `[B,H,T,k_sel,D]`.
3. Recompute attention weights `P` from saved LSE (same as FlashAttention pattern).
4. Compute `dK_contrib = dS × Q`, `dV_contrib = P × dO` → dense `[B,H,T*k_sel,D]` tensors.
5. `dk.scatter_add_(dim=2, ...)` and `dv.scatter_add_(dim=2, ...)`.

PyTorch's `scatter_add_` is backed by a highly optimized deterministic reduce kernel
(uses atomics internally but with far better coalescing than the query-major scatter).
The key advantage: the `dk_contrib`/`dv_contrib` tensors are dense and contiguous,
so the gather+scatter pattern is cache-friendly compared to per-program random atomics.

**Note:** The old `_sparse_attn_bwd_dkdv_kernel` Triton function remains in the file
(not deleted) but is no longer called. It can be removed after validation.

**Expected impact:** 3–5× backward speedup for GSA layers.
Combined with FIX-PERF-02 (4× fewer scatter operations), total backward improvement: **12–20×**.

---

## Projected Throughput After Fixes

| Metric | Before | After (projected) |
|---|---|---|
| tok/s | ~14,000 | ~60,000–100,000 |
| Step time | ~9.3s | ~1.5–2.5s |
| GSA backward | ~3.5s/layer | ~0.2–0.5s/layer |
| ZeRO offload | CPU (PCIe) | GPU (NVLink) |

> Note: The `lm_head` projection (`4096 × 131,075 = 537M params`) still consumes
> ~27% of forward time (595ms). This is a structural cost of the 131K vocabulary
> with untied embeddings. Consider `liger_kernel`'s fused CE loss to reduce the
> backward cost of this layer in a future fix.
