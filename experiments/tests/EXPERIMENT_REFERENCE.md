# Experiment Reference: Test Configurations vs Graph Series & Code

This document maps **Tests.md** to the **graph series (T1–T8)**, summarizes **config and code differences** across `experiments/tests`, and ties them to the **throughput/loss** insights from the run results.

---

## 1. Graph series (T1–T8) ↔ Test mapping

| Graph label | Test | Config / variant | Steps |
|-------------|------|------------------|-------|
| **T1 New Recurrence** | Test 1 | `test1_diff_rec.yaml` — embedding-space recurrence (`diff_rec`) | 100 |
| **T1 Old Recurrence** | Test 1 | `test1_lead_wo_rev.yaml` — lead-aligned, stream-3 recurrence (`lead_wo_rev`) | 100 |
| **T3 Standard Emb** | Test 3 | Standard embedding, `diff_rec`, init from Test 2 | 1000 |
| **T4 Kronecker** | Test 4 | Kronecker embedding, `diff_rec` (non-reversible backbone) | 1000 |
| **T5 With Reversibility** | Test 5 | Kronecker + **reversible** backbone (`ReversibleMidpointStack`) | 1000 |
| **T6 With Fused GSA FB** | Test 6 | Test 5 + **Triton GSA** (`require_gsa_triton`) | 500 |
| **T7 With Fused GSA DeltaNet** | Test 7 | Test 6 + **Fused DeltaNet** (FLA `fla_gated_delta_rule`) | 1000 |
| **T8 With Fused MLP CE** | Test 8 | Test 7 + **fused CE** (`use_fused_ce`) + **additional fused** (Liger MLP/CE assertion) | 1000 |

**Note:** Test 2 is not plotted; it is the 20-step run that produces `model_init.pt` used by Tests 3–8. Test 9 (3000-step resume), Test 10 (3B MoE), and Test 11 (Fused MoE) are not in the T1–T8 graphs.

---

## 2. Config knobs that change per test

All tests share the same data (e.g. wikitext-103-raw-v1, max_length 512, pack_into_blocks). Differences are in **model** and **training**:

| Test | `embedding_type` | `model_variant` | `require_gsa_triton` | `require_deltanet_fused` / `require_fused_kernels` | `require_fused_deltanet_kernel` | `use_fused_ce` | `require_additional_fused_kernels` | `max_train_steps` |
|------|------------------|-----------------|----------------------|---------------------------------------------------|----------------------------------|----------------|-------------------------------------|-------------------|
| 1 | standard | `diff_rec` or `lead_wo_rev` | — | — | false | — | — | 100 |
| 3 | **standard** | diff_rec | — | — | false | — | — | 1000 |
| 4 | **kronecker** | diff_rec | — | — | false | — | — | 1000 |
| 5 | kronecker | **reversible** | — | — | false | — | — | 1000 |
| 6 | kronecker | reversible | **true** | — | false | — | — | **500** |
| 7 | kronecker | reversible | true | **true** | **true** | — | — | 1000 |
| 8 | kronecker | reversible | true | true | true | **true** | **true** | 1000 |

- **Test 3 vs 4:** Only `embedding_type` changes (standard → kronecker); same `diff_rec` backbone.
- **Test 4 vs 5:** Same kronecker; backbone switches from `diff_rec` to `reversible` (ReversibleMidpointStack).
- **Test 5 vs 6:** GSA path uses Triton (`require_gsa_triton`); Test 6 runs 500 steps.
- **Test 6 vs 7:** Fused DeltaNet (FLA) and `require_fused_kernels` / `require_fused_deltanet_kernel` enabled.
- **Test 7 vs 8:** Training uses fused CE (`use_fused_ce`) and asserts additional fused kernels (Liger MLP + CE).

---

## 3. Code differences between tests

### 3.1 Model backbone and entrypoint

- **Test 1** (`Test 1/code/main.py`): Dispatches by `model_variant`:
  - `lead_wo_rev` / `wo_rev` → `Model1B_WoRev` (from `recurrence_model_1b_wo_rev.py`) — “old” recurrence.
  - `diff_rec` → `Model1B_DiffRec` (from `different_recurrence_model_1b_wo_rev.py`) — “new” recurrence.
- **Tests 3 & 4**: Use **non-reversible** backbone only:
  - `different_recurrence_model_1b_wo_rev.py` (same module name in both); Test 3 uses standard embedding, Test 4 kronecker.
- **Tests 5, 6, 7, 8**: Use **reversible** backbone:
  - `recurrence_model_1b.py` with `ReversibleMidpointStack`; all use kronecker in the graphed configs.

### 3.2 Embedding

- **Standard** (`embedding_type: "standard"`): Conventional token embedding table.
- **Kronecker** (`embedding_type: "kronecker"`): `KroneckerEmbeddings` (byte-level, POS_DIM×CHAR_DIM); used from Test 4 onward in the progression.

### 3.3 Fused kernels (in model)

- **GSA (Triton):** In `recurrence_model_1b.py`, GatedSparseAttention uses:
  - `fused_indexer_topk` (chunked O(T·k) importance scores),
  - `triton_sparse_attention` when Triton is available.
  - Test 6 enables **require_gsa_triton** so the run environment must provide these (asserted in save_init_model).
- **DeltaNet (FLA):** In `GatedDeltaNet.forward()`, when `config.require_fused_deltanet_kernel` is True and FLA is available, the recurrence uses `fla_gated_delta_rule` instead of the Python loop. **Test 7** sets `require_fused_deltanet_kernel: true` (and related flags); Tests 5–6 leave it false.
- **Liger MLP & CE:** `recurrence_model_1b.py` (Tests 5–8) already defines `LigerSwiGLUMLP` and `LigerFusedLinearCrossEntropyLoss`. The **training** path difference is:
  - **Test 8:** `use_fused_ce: true` and `require_additional_fused_kernels: true` → forward is called with `ntp_targets`/`mtp_targets`, loss computed inside the model (fused CE path in `train.py`), and save_init_model asserts Liger CE + first-layer Liger MLP.
  - **Test 7:** Same model classes but training uses the non-fused CE path (logits + F.cross_entropy in train.py); no `use_fused_ce` or additional-fused assertion.

### 3.4 Training script (loss path)

- **Test 8 (and Test 9):** `train.py` reads `use_fused_ce`; when True, calls `model(..., return_loss=True, ntp_targets=y_ntp, mtp_targets=y_mtp)` and treats returned `logits_ntp`/`logits_mtp` as scalar losses (fused CE path).
- **Tests 5, 6, 7:** No `use_fused_ce`; standard forward then `F.cross_entropy` on logits.

---

## 4. How this matches the graphs (from your results)

- **Throughput (Tokens/sec):**  
  T6/T8 (fused GSA + fused CE path) > T7 (fused GSA + DeltaNet) > T5 (reversible only) > T1 (recurrence) > T4 (Kronecker, non-reversible) > T3 (standard emb, non-reversible).  
  So: **Fused GSA and fused DeltaNet** explain most of the gain; **fused CE** adds a bit more. **Kronecker + non-reversible** (T4) and **standard emb** (T3) are slow and oscillatory.

- **Loss:**  
  T8 ≈ T7 (best), then T6, T5, T1 (all good), while T4 and T3 stay high. So **reversibility + Kronecker** (T5) recovers loss; **fused kernels (T6–T8) keep or improve it**. The **T5 loss spike around step ~500** is worth checking in CleanedLogs (LR, data order, or restarts).

- **T1 New vs Old:** Both recurrence variants (diff_rec vs lead_wo_rev) perform similarly in your plots; no clear winner.

---

## 5. Tests not in the T1–T8 graphs

- **Test 2:** 20-step init; produces `model_init.pt`.
- **Test 9:** Resumes Test 8 to 3000 steps (validation of checkpoint resume).
- **Test 10:** 3B-class MoE, 100 steps (architectural check).
- **Test 11:** Fused MoE (grouped GEMM), 1000 steps (throughput scaling).

---

## 6. Quick reference: where to look in code

| What you want | Where to look |
|---------------|----------------|
| Config for a test | `experiments/tests/Test_<N>_.../configs/*.yaml` |
| Model variant dispatch (T1) | `Test 1/code/main.py` (variant → Model1B_WoRev vs Model1B_DiffRec) |
| Non-reversible backbone (T3, T4) | `Test_3_.../code/src/models/different_recurrence_model_1b_wo_rev.py` |
| Reversible backbone (T5–T8) | `Test_5_.../code/src/models/recurrence_model_1b.py` (and same in 6,7,8) |
| GSA Triton / fused indexer | `recurrence_model_1b.py` (GatedSparseAttention), `kernels/triton_indexer*.py`, `triton_sparse_attn.py` |
| Fused DeltaNet (FLA) | `recurrence_model_1b.py` (GatedDeltaNet), `kernels/fla_deltanet.py` |
| Fused CE and Liger MLP | `train.py` (`use_fused_ce` path), `models/liger_ops.py`, `recurrence_model_1b.py` (LigerSwiGLUMLP, LigerFusedLinearCrossEntropyLoss) |
| Init-model assertions | Each test’s `scripts/save_init_model.py` (embedding, GSA Triton, deltanet fused, additional fused) |

This file is the single reference tying **Tests.md**, **configs**, **code paths**, and **graph series (T1–T8)** together.
