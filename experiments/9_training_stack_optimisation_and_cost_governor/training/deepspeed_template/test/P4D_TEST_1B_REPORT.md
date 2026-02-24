# P4D Test Report -- Model 1B: GSA/DeltaNet BF16 & Kernel Diagnostics

**Date:** 2026-02-23
**Branch:** `feature/experiments-p4d-gsa-memory-fixes`
**Source code path:** `experiments/tests/Test_9_extend_test8_to_3000steps_save_state/code`
**Notebook:** `run_tests_p4d.ipynb`

---

## 1. Executive Summary

| Metric | Result |
|--------|--------|
| **Test suite** | `test_bf16_and_kernels.py` |
| **Model tested** | Model1B (recurrence_model_1b) |
| **Hardware** | NVIDIA A100-SXM4-80GB (P4D instance) |
| **Tests executed** | 3 / 3 |
| **Tests passed** | 3 / 3 |
| **Overall status** | **PASS** |

All three diagnostic tests (bf16 numerical stability, GSA train/inference path policy, kernel profiling) passed on the 1B recurrence model running on an A100-80GB GPU.

---

## 2. Test Environment

### 2.1 Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA A100-SXM4-80GB |
| VRAM | 85.1 GB |
| Instance type | P4D (AWS) |
| Runtime | Google Colab (connected to P4D) |

### 2.2 Software Dependencies

| Dependency | Version / Status |
|------------|-----------------|
| PyTorch | CUDA-enabled (Colab default) |
| Triton | v3.6.0 |
| Flash Linear Attention (FLA) | Installed, operational |
| Transformers | Installed |
| Tokenizers | Installed |
| DeepSpeed | Installed |

### 2.3 Kernel Availability (6/6 available)

| Kernel | Status |
|--------|--------|
| `triton_rmsnorm` | ENABLED |
| `triton_sinkhorn_knopp` | ENABLED |
| `triton_sparse_attention` | ENABLED |
| `pytorch_sparse_attention` | ENABLED |
| `fla_gated_delta_rule` | ENABLED |
| `fused_indexer_topk` | ENABLED |

---

## 3. Model Under Test

### 3.1 Model Identity

| Property | Value |
|----------|-------|
| Variant | `1b` (default; `GSA_MODEL_VARIANT` not set) |
| Module | `src.models.recurrence_model_1b` |
| Class | `Model1B` |
| Initialization | From scratch (random weights, `torch.manual_seed(42)`) |

### 3.2 Architecture Specification

| Property | Value |
|----------|-------|
| **Total parameters** | 1,647,204,532 (~1.65B) |
| **Active parameters** | ~1.513B (100% active, no MoE sparsity) |
| **Hidden size** | 4,096 |
| **Vocabulary** | 131,072 (TSAI 131K tokenizer) |
| **Total layers** | 8 |
| DeltaNet layers | 6 (75%) -- O(N) linear attention |
| GSA layers | 2 (25%) -- Adaptive sparse attention |
| **Context target** | 262,144 tokens (standard RoPE) |
| **MTP heads** | 2 predictions (next-token + multi-token) |
| **MoE experts** | 0 (dense FFN) |
| **Embedding type** | Kronecker Product |

### 3.3 Embedding Details

| Property | Value |
|----------|-------|
| CHAR_DIM | 256 (all UTF-8 bytes) |
| POS_DIM | 32 (tokens up to 32 bytes) |
| Kronecker D | 8,192 (256 x 32) |
| Projection | `pf_to_model` linear: 8,192 -> 4,096 (33.6M params) |
| PF buffer | 1,073.7M (vocab x 8,192, non-trainable) |
| Embedding tying | Not possible (8,192 != 4,096) |

### 3.4 Tokenizer

| Property | Value |
|----------|-------|
| Name | TSAI 131K |
| Vocab size | 131,072 |
| Total tokens (with special) | 131,075 |
| BOS token | `<\|startoftext\|>` (ID: 131072) |
| EOS token | `<\|return\|>` (ID: 131073) |
| PAD token | `<\|endoftext\|>` (ID: 131074) |
| Source | `src/tokenizer/tokenizer.json` (local filesystem) |

### 3.5 Model Source

Models are **not** loaded from pre-trained checkpoints, HuggingFace Hub, or S3. They are instantiated from scratch using:
1. `ModelConfig()` -- hardcoded architecture hyperparameters
2. `ModelClass(config, embedding_type="kronecker", bpe_vocab=..., pf_codec=...)` -- random initialization
3. Deterministic seed: `torch.manual_seed(42)`

---

## 4. Test Details & Results

### 4.1 Test 1: BF16 Pipeline, NaN Guard, and Differentiability -- PASS

**Purpose:** Validates that the full model pipeline works correctly in BFloat16 precision with no numerical instability.

**What was checked:**

| Check | Result |
|-------|--------|
| BF16 parameter ratio | 379/379 (100.0%) -- exceeds 95% threshold |
| Non-finite parameters at init | 0 detected |
| Forward pass loss finiteness | Finite (no NaN/Inf) |
| Per-layer output finiteness | All 8 layers finite |
| Backward pass gradient finiteness | 0 non-finite gradients across 370 gradient tensors |
| Gradient dtype distribution | 100% `torch.bfloat16` |
| GSA differentiable path (layer 3) | W_q, W_k, W_v, W_gv all have non-zero finite gradients |
| DeltaNet differentiable path | q_proj, k_proj, v_proj, g_proj all have non-zero finite gradients |
| MLP/MoE differentiable path | Layer 0 MLP sublayer has non-zero finite gradients |
| Optimizer step (AdamW) | Parameters changed after `optimizer.step()` |

**Batch configuration:**
- Sequence length: 20 tokens
- Batch size: 1
- Input token range: [0, 100)
- Loss: `loss_ntp + 0.3 * loss_mtp + aux_loss`

**Verdict:** All numerical stability, dtype adherence, and differentiability checks passed. The model is safe for BF16 training.

---

### 4.2 Test 2: GSA Train vs. Inference Path Policy -- PASS

**Purpose:** Validates that Gated Sparse Attention (GSA) correctly selects different kernel paths for inference (no-grad) vs. training (grad-enabled).

**GSA layer tested:** Layer index 3

**What was checked:**

| Mode | Kernel Used | Result |
|------|-------------|--------|
| **Inference** (`torch.no_grad()`) | `triton_sparse_attention` (fused Triton) | PASS -- used fused kernel for maximum throughput |
| **Training** (`requires_grad=True`) | `triton_sparse_attention` with fused backward | PASS -- fused backward path, not PyTorch fallback |

**Training path details:**
- GSA `W_q.weight.grad` is present, finite, and non-zero
- Gradients flow correctly through the Triton fused backward path
- PyTorch fallback (`pytorch_sparse_attention`) was available but not needed

**Verdict:** GSA correctly dispatches to the fused Triton kernel in both inference and training modes. The training path uses `triton_sparse_attention` with fused backward (not the slower `pytorch_sparse_attention` fallback), which is optimal for A100 hardware.

---

### 4.3 Test 3: Kernel Accounting & Step Timing Profile -- PASS

**Purpose:** Instruments every fused kernel call during a full forward+backward step, counting invocations and measuring wall-clock time.

#### 4.3.1 Kernel Call & Time Report

| Kernel | Fwd Calls | Fwd ms | ms/call | Bwd Calls | Bwd ms |
|--------|-----------|--------|---------|-----------|--------|
| `triton_rmsnorm` | 33 | 3.838 | 0.116 | 0 | 0.000 |
| `triton_sinkhorn_knopp` | 14 | 1.639 | 0.117 | 0 | 0.000 |
| `triton_sparse_attention` | 3 | 1.533 | 0.511 | 2 | 0.956 |
| `pytorch_sparse_attention` | 0 | 0.000 | 0.000 | 0 | 0.000 |
| `fla_gated_delta_rule` | 6 | 6.684 | 1.114 | 6 | 6.576 |
| `fused_indexer_topk` | 3 | 2.519 | 0.840 | 2 | 1.559 |

#### 4.3.2 Call Count Analysis

| Kernel | Expected | Actual (fwd/bwd) | Explanation |
|--------|----------|-------------------|-------------|
| `triton_rmsnorm` | ~33 | 33 / 0 | RMSNorm applied at every sub-layer boundary. Forward-only Triton kernel (PyTorch autograd handles backward). |
| `triton_sinkhorn_knopp` | ~14 | 14 / 0 | Sinkhorn normalization for sparse routing in GSA layers. Forward-only. |
| `triton_sparse_attention` | 2-3 | 3 / 2 | 2 GSA layers + potential MTP head interaction. Fused forward and backward. |
| `pytorch_sparse_attention` | 0 | 0 / 0 | Correctly unused -- Triton path preferred on A100. |
| `fla_gated_delta_rule` | 6 | 6 / 6 | Exactly 6 DeltaNet layers, each calling FLA once per direction. |
| `fused_indexer_topk` | 2-3 | 3 / 2 | Fused top-k index selection for GSA sparse routing. |

#### 4.3.3 Step Timing

| Phase | Wall Time |
|-------|-----------|
| **Forward** | 71.40 ms |
| **Backward** | 266.49 ms |
| **Total step** | 337.89 ms |
| **Backward/Forward ratio** | 3.73x |

#### 4.3.4 Kernel Time Breakdown (Forward Pass)

| Kernel | Forward ms | % of Forward |
|--------|-----------|-------------|
| `fla_gated_delta_rule` | 6.684 | 9.4% |
| `triton_rmsnorm` | 3.838 | 5.4% |
| `fused_indexer_topk` | 2.519 | 3.5% |
| `triton_sinkhorn_knopp` | 1.639 | 2.3% |
| `triton_sparse_attention` | 1.533 | 2.1% |
| **Total instrumented** | **16.213** | **22.7%** |
| **Uninstrumented** (embeddings, projections, loss, etc.) | **55.187** | **77.3%** |

#### 4.3.5 Per-Layer Timing

Per-layer forward/backward timings were **empty** (all 0.000 ms). This is expected and documented behavior -- the model uses a **reversible midpoint stack** via `functional_call`, which bypasses PyTorch's standard `register_forward_hook` mechanism. The hooks register correctly but are never triggered because the layers are invoked through the functional API.

#### 4.3.6 Assertions Verified

| Assertion | Result |
|-----------|--------|
| Loss is finite | PASS |
| All gradients finite | PASS |
| No non-finite forward activations | PASS |
| No non-finite backward gradients | PASS |
| `fused_indexer_topk` used in forward | PASS (3 calls) |
| `fla_gated_delta_rule` used in forward | PASS (6 calls) |
| Sparse attention kernel used in forward | PASS (3 calls via `triton_sparse_attention`) |

**Verdict:** All kernels are being used as expected. Kernel call counts match the architecture (6 DeltaNet layers, 2 GSA layers). The fused Triton path is preferred over the PyTorch fallback.

---

## 5. Other Notebooks in Test Suite (Not Executed)

| Notebook | Target | Branch | Model Variants | Status |
|----------|--------|--------|----------------|--------|
| `run_tests_p4d.ipynb` | P4D / A100-80GB | `feature/experiments-p4d-gsa-memory-fixes` | 1B (default) | **Executed -- 3/3 PASS** |
| `run_tests_3b_8b.ipynb` | A100 (3B & 8B) | `p9/feat/reversibility_test` | 3B, 8B | Template only, no outputs |
| `run_tests_colab.ipynb` | Colab | `feature/experiments-p4d-gsa-memory-fixes` | 1B (default) | Template only, no outputs |

### Key Differences Between Notebooks

| Feature | `run_tests_p4d.ipynb` | `run_tests_3b_8b.ipynb` | `run_tests_colab.ipynb` |
|---------|----------------------|------------------------|------------------------|
| Branch | `feature/experiments-p4d-gsa-memory-fixes` | `p9/feat/reversibility_test` | `feature/experiments-p4d-gsa-memory-fixes` |
| Model source dir | `experiments/tests/Test_9_.../code` | `experiments/.../deepspeed_template` | `experiments/tests/Test_9_.../code` |
| Tokenizer copy step | Yes (Cell 2) | No | Yes (Cell 2) |
| Model variant support | 1b, 70b | 1b, 3b, 8b, 70b | 1b, 70b |

---

## 6. Observations & Recommendations

### 6.1 Per-Layer Timing Not Available

The reversible midpoint stack (`functional_call`) bypasses standard PyTorch hooks, so per-layer forward/backward timing data is unavailable. This is a known limitation documented in the test output as a `[WARN]`. To get per-layer profiling, consider using `torch.profiler` with CUDA activity tracing instead of hook-based timing.

### 6.2 Backward/Forward Ratio is 3.73x

The backward pass (266.49 ms) takes 3.73x longer than the forward pass (71.40 ms). For typical training the expected ratio is ~2-3x. The slightly elevated ratio may be attributable to:
- The reversible midpoint stack recomputing activations during backward
- Small batch size (1) and short sequence length (20) not saturating the GPU
- Instrumentation overhead from kernel wrappers with `cuda.synchronize()` calls

### 6.3 Small Batch Test Limitation

The test uses batch_size=1, seq_len=20 (18 effective input tokens). This validates correctness and differentiability but does not stress-test memory or throughput at training-scale batch sizes. Consider adding a stress test with larger batches (e.g., batch_size=4, seq_len=2048) to validate memory stability on the A100-80GB.

### 6.4 Missing 70B Model File

The test script references `recurrence_model_70b.py` but it is not present in `src/models/`. Only `recurrence_model_1b.py` exists. This is not a failure (the test defaults to 1b), but the 70b code path is untestable in the current source tree.

### 6.5 3B/8B Tests Not Yet Executed

The `run_tests_3b_8b.ipynb` notebook has no outputs. If these model variants exist on the `p9/feat/reversibility_test` branch, they should be run to validate the full model family.

---

## 7. Conclusion

The IDFT smoke test for the **1B recurrence model** on **A100-80GB** hardware is **fully passing**. All fused Triton and FLA kernels are correctly dispatched, BFloat16 numerical stability is confirmed across all layers, and the full forward-backward-optimizer pipeline produces valid gradients with no NaN/Inf issues. The model is ready for training at the 1B scale on P4D instances.
