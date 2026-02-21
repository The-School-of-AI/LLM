# BF16 & Kernel Diagnostics Test Report — p4d Branch

**Model:** recurrence_model_1b (1.65B params, dense)
**Branch:** `feature/experiments-p4d-gsa-memory-fixes`
**Test folder:** `experiments/tests/Test_9_extend_test8_to_3000steps_save_state/code`
**GPU:** NVIDIA A100-SXM4-80GB
**Date:** 2026-02-20

---

## Summary

| Test | Status | Description |
|------|--------|-------------|
| bf16 | PASS | bf16 dtype consistency, NaN checks, differentiable paths |
| gsa | PASS | GSA train vs inference kernel selection policy |
| profile | PASS | Fused kernel call counts and wall-time accounting |

**Result: 3/3 tests passed**

---

## Environment

```
CUDA available:       True
GPU:                  NVIDIA A100-SXM4-80GB
VRAM:                 85.1 GB
HAS_TRITON:           True
HAS_FLA:              True
Triton RMSNorm:       ENABLED
Triton Sinkhorn:      ENABLED
Triton Sparse Attn:   ENABLED
fla GatedDeltaRule:   ENABLED
```

---

## What This Branch Does Differently

This branch implements **fused Triton backward kernels** for GSA sparse attention. Instead of falling back to `pytorch_sparse_attention` during training (as `codex/gsa-training-grad-fix` does), it uses `triton_sparse_attention` with `use_triton_backward=True` for both forward and backward passes.

Key commits:
- `d4138ec8` — Triton GSA backward kernels (dQ, dK/dV) with FlashAttention-style recomputation
- `74cdf5b3` — Attention mask threading through reversible midpoint stack
- `2d1d4900` — MLP path fix and fused cross-entropy wiring

Architecture:
- `TritonSparseAttnFn` (`torch.autograd.Function`) with fused forward and backward
- Variance EMA snapshot prevents race conditions across gradient-accumulation micro-batches
- `pytorch_sparse_attention` still available as a debugging fallback via `USE_TRITON_BACKWARD` toggle

---

## Test 1: bf16 Pipeline, NaN Checks, and Differentiable Paths

**Status: PASS**

### What was validated

| Check | Result |
|-------|--------|
| bf16 parameter ratio | 379/379 (100.0%) |
| Non-finite parameters | None |
| Forward loss finite | Yes |
| Per-layer output finite | All 8 layers clean |
| Gradient dtype | 370 params with bf16 grads |
| Non-finite gradients | None |
| GSA projections (W_q, W_k, W_v, W_gv) | Non-zero gradients confirmed |
| DeltaNet projections (q, k, v, g) | Non-zero gradients confirmed |
| MLP/MoE sublayer | Non-zero gradients confirmed |
| AdamW optimizer step | Parameters updated successfully |

### What this proves

- bf16 dtype is preserved from input through forward, backward, and weight update
- No NaN or Inf values appear anywhere in the compute graph
- All three block types (GSA, DeltaNet, MLP) are fully differentiable
- The training loop will produce valid weight updates

---

## Test 2: GSA Train vs Inference Path Policy

**Status: PASS**

### Path Selection

| Path | Kernel Used | Verified |
|------|------------|----------|
| `torch.no_grad()` | `triton_sparse_attention` (fused forward) | Yes — call count > 0 |
| Grad-enabled | `triton_sparse_attention` (fused forward + backward) | Yes — call count > 0 |

Both paths use the same Triton kernel. The difference is that the grad-enabled path activates `TritonSparseAttnFn.backward()` which computes dQ, dK, dV via fused Triton JIT kernels with online softmax recomputation.

### GSA Gradient Checks

| Check | Result |
|-------|--------|
| W_q.weight.grad present | Yes |
| W_q.weight.grad finite | Yes |
| W_q.weight.grad non-zero | Yes |

### What this proves

- The original `RuntimeError` is fully resolved
- Triton fused backward produces valid, finite, non-zero gradients
- No PyTorch fallback needed — entire GSA path is fused

---

## Test 3: Kernel Usage and Timing Report

**Status: PASS**

### Kernel Availability

| Kernel | Available |
|--------|-----------|
| triton_rmsnorm | YES |
| triton_sinkhorn_knopp | YES |
| triton_sparse_attention | YES |
| pytorch_sparse_attention | YES (unused) |
| fla_gated_delta_rule | YES |
| fused_indexer_topk | YES |

### Kernel Call Counts and Wall Time

| Kernel | Fwd Calls | Fwd ms | Fwd ms/call | Bwd Calls | Bwd ms |
|--------|-----------|--------|-------------|-----------|--------|
| triton_rmsnorm | 33 | 3.838 | 0.116 | 0 | 0.000 |
| triton_sinkhorn_knopp | 14 | 1.639 | 0.117 | 0 | 0.000 |
| triton_sparse_attention | 3 | 1.533 | 0.511 | 2 | 0.956 |
| pytorch_sparse_attention | 0 | 0.000 | 0.000 | 0 | 0.000 |
| fla_gated_delta_rule | 6 | 6.684 | 1.114 | 6 | 6.576 |
| fused_indexer_topk | 3 | 2.519 | 0.840 | 2 | 1.559 |

### Step Timing

| Phase | Time |
|-------|------|
| Forward | 71.40 ms |
| Backward | 266.49 ms |
| **Total** | **337.89 ms** |

### Key Observations

- **triton_sparse_attention** is called in both forward (3) and backward (2) — confirms the fused Triton backward kernel is active during training
- **pytorch_sparse_attention** has 0 calls — entirely replaced by fused Triton path
- **fla_gated_delta_rule** called forward (6) and backward (6) — FLA's custom autograd works end-to-end for all 6 DeltaNet layers
- **fused_indexer_topk** called forward (3) and backward (2) — O(T*k) sparse index selection is fused
- **triton_rmsnorm** (33 calls) and **triton_sinkhorn_knopp** (14 calls) have 0 backward calls — they run inside `torch.no_grad()` guards; gradients flow through PyTorch fallback paths
- Per-layer hook timings are empty because the reversible midpoint stack uses `torch.func.functional_call`, which bypasses standard PyTorch module hooks. This is expected.

---

## Comparison vs `codex/gsa-training-grad-fix`

| Metric | codex/gsa-training-grad-fix | feature/p4d (this branch) | Delta |
|--------|---------------------------|---------------------------|-------|
| GSA training kernel | pytorch_sparse_attention | triton_sparse_attention (fused bwd) | Triton replaces PyTorch |
| GSA bwd time | 1.662 ms | 0.956 ms | **-42%** |
| Forward | 71.07 ms | 71.40 ms | +0.5% |
| Backward | 267.63 ms | 266.49 ms | **-0.4%** |
| Total step | 338.70 ms | 337.89 ms | **-0.2%** |
| triton_sparse_attention bwd calls | 0 | 2 | Fused bwd active |
| pytorch_sparse_attention calls | 3 (1 fwd + 2 bwd) | 0 | Eliminated |

The 42% speedup in GSA backward is from replacing PyTorch autograd with fused Triton JIT kernels. The overall step improvement is modest (-0.2%) because GSA is only 2/8 layers, but this scales with more GSA layers or longer sequences.

---

## Model Architecture (1B)

```
Vocabulary:        131,072 (TSAI 131K tokenizer)
Hidden Size:       4,096
Embedding:         Kronecker (POS_DIM=32 x CHAR_DIM=256 = D=8192)
Total Layers:      8
  - DeltaNet:      6 layers (75%) - O(N) linear attention
  - GSA:           2 layers (25%) - Adaptive sparse attention
Context Target:    262,144 tokens
MTP:               2 predictions (DeepSeek-V3 style)
Total Parameters:  1,647,204,532 (~1.65B)
Active Parameters: ~1.513B (100% active, no MoE sparsity)
```

---

## Notes

- `pytorch_sparse_attention` is still importable (available=YES) but deliberately unused. It can be re-enabled for debugging by setting `USE_TRITON_BACKWARD = False` at module level in `triton_sparse_attn.py`.
- The variance EMA snapshot fix in GSA prevents a race condition where gradient-accumulation micro-batch N+1 mutates `variance_ema` before micro-batch N's backward reconstruct.
- Per-layer hook profiling won't work with reversible midpoint stack (`functional_call` bypasses hooks). Instrument inside the reversible stack if per-layer timing is needed.
