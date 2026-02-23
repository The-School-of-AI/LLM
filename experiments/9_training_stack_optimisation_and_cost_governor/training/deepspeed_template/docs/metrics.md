# Training Metrics Reference

Everything logged to `checkpoints_*/metrics.jsonl` — what each field means, what value to expect, and what the actual observed values were during the MoE test run.

---

## Model Used for Testing

### Architecture: Model3B (`src/models/recurrence_model_3b.py`)

| Property | Value |
|---|---|
| Total parameters | ~3.9B |
| Active parameters | ~1.74B (sparse MoE routing) |
| Hidden size | 4096 |
| Number of layers | 8 (6 DeltaNet + 2 GSA) |
| Attention type | 75% Gated DeltaNet + 25% Gated Sparse Attention (GSA) |
| FFN type | MoE (LightningMLP → MoEFFN) on every layer |
| Reversible integration | ReversibleMidpointStack — enables memory-efficient backprop |
| MTP head | 2-token multi-token prediction (loss_ntp + loss_mtp) |
| Max sequence length | 262,144 (256k) |
| Vocab size | 131,072 (2^17, Kronecker embeddings) |

### MoE Configuration

| Property | Value |
|---|---|
| Real experts | 20 |
| Null experts (slots) | 4 → computed as `round(20 * (1−5/6) / (5/6)) = 4` |
| Total router slots | 24 (20 real + 4 null) |
| Top-k | 2 (dynamic; each token picks 2 slots from 24) |
| Data sparsity ρ | 5/6 ≈ 0.833 — target fraction of selections that are null |
| Shared expert | 1 always-active expert per layer (intermediate_size = 2048) |
| Routed expert size | 1024 intermediate per expert |
| Null logit | Single shared learnable scalar, broadcast to all null slots |
| Balance loss weight | 2e-2 |
| Z-loss weight | 1e-3 |
| Null-rate loss weight | 1e-2 |

**Why 4 null experts?** With ρ=5/6, the router is trained to route ~83% of its top-k selections to null slots, meaning only ~17% hit real experts on average. 4 null copies give enough "null surface area" for balanced null-slot coverage while keeping the total slot count small (24 vs the previous 40).

### Training Setup for This Run

| Property | Value |
|---|---|
| Dataset | wikitext-2-raw-v1 (tiny, for smoke-testing) |
| Max sequence length | 512 |
| Optimizer steps | 50 (`max_train_steps`) |
| GPUs | 4 × (A100/H100 class) |
| DeepSpeed stage | ZeRO-2 (`zero-2-moe-oom-fixed.json`) |
| Gradient accumulation | Managed by DeepSpeed engine |
| LR schedule | Linear warmup to 1e-5, then cosine decay |

---

## Metrics Reference

### Loss Metrics

| Field | What it measures | Expected range | Observed (steps 1–24) |
|---|---|---|---|
| `loss` | Total combined loss = loss_ntp + 0.3×loss_mtp + loss_aux | Decreases over time | 18.04 → 0.96 (fast drop on tiny dataset) |
| `loss_ntp` | Next-token prediction cross-entropy (primary task) | Decreases, should dominate | 13.67 → 0.31 |
| `loss_mtp` | Multi-token prediction cross-entropy t+2 (auxiliary task, weight 0.3) | Should track loss_ntp | 11.99 → 0.31 |
| `loss_aux` | Combined MoE auxiliary loss (L_bal + L_z + L_null, all scaled) | 0.1–1.5 — should slowly decrease | 0.39–0.81 |
| `loss_null_router` | NULL — see note below | — | always null |
| `loss_moe_router` | NULL — see note below | — | always null |

> **Why `loss_null_router` and `loss_moe_router` are always null:**
> These fields are only populated if the model's `forward()` returns `aux_loss` as a **2-element tuple** `(null_component, moe_component)`. The current model returns a single merged scalar from `MoEGate`: `2e-2 * L_bal + 1e-3 * L_z + 1e-2 * L_null`. The split is available in the `moe_L_*` fields below instead.

---

### MoE Router Metrics

All `moe_*` fields are averaged across all 8 MoE gate modules (one per decoder layer), collected after each optimizer step from `MoEGate.last_*` attributes.

#### `moe_null_rate`

- **What it is:** Fraction of all top-k slot selections (across B×T×top_k picks) that landed on a null slot
- **Formula:** `is_null.float().mean()` where `is_null = (topk_idx >= num_real_experts)`
- **Target:** ρ = 5/6 ≈ **0.833** (83.3% of selections should be null)
- **Healthy range:** 0.70–0.95
- **Alert if:** < 0.30 (router ignoring null slots entirely) or > 0.98 (router routing everything to null = collapsed)
- **Observed:** **0.000** for steps 1–21, then 0.0007, 0.0003, 0.0001 — the router has not learned to use null slots yet. This is **normal at training start** and should increase gradually as `L_null` gradient accumulates
- **Interpretation of 0.0:** With ρ=0.833 as target, the router starts at ~0 and must learn that routing to null is preferred. Both top-k selections go to real experts every token. This will converge toward 83% null over thousands of steps.

#### `moe_avg_real_experts`

- **What it is:** Average number of real expert selections per token (across B×T tokens)
- **Formula:** `(~is_null).sum(dim=-1).float().mean()` — counts non-null slots per token, then averages
- **Target:** `top_k × (1 − ρ) = 2 × (1 − 5/6) = 2 × 1/6 ≈ 0.33` real expert assignments per token (at convergence)
- **Healthy range:** 0.2–0.7
- **Alert if:** = top_k (2.0) consistently — means null rate is 0, router not sparsifying
- **Observed:** **2.0 constant** — confirms null_rate=0, every token routes both top-k picks to real experts. Expected to decrease as training learns sparsity.

#### `moe_zero_real_frac`

- **What it is:** Fraction of tokens where ALL top-k selections were null (no real expert received this token)
- **Formula:** `is_null.all(dim=-1).float().mean()` — True when every top-k pick is a null slot
- **Target:** With ρ=0.833 and top_k=2: P(both null) = ρ² = 0.694 = **69.4%** of tokens should be "all-null" at convergence
- **Healthy range:** 50–85% at convergence
- **Alert if:** > 95% (tokens never reach real experts) or = 0.0 early + stays 0 after many steps
- **Observed:** **0.0 throughout** — consistent with null_rate=0. All tokens are hitting real experts. This will rise toward ~69% as the router learns sparsity.

#### `moe_L_bal`

- **What it is:** Load balance loss over real experts only — penalizes uneven token distribution across the 20 real experts
- **Formula:** `num_experts × sum(f_real × P_real)` where `f_real` = per-expert selection frequency and `P_real` = average softmax probability. For perfect balance, this equals 1.0.
- **Expected:** Should approach ~1.0 as routing balances
- **Alert if:** > 5.0 (severe imbalance — some experts seeing 5× more data than others) or < 0.5 (suspicious)
- **Observed:** **1.39–4.04** — oscillates across steps, with some very uneven steps (e.g. step 6: E12 got 1949 tokens vs E13 with 18 tokens = 108× ratio). This is normal for early training but should improve.

#### `moe_L_null`

- **What it is:** Null-rate regularizer — squared deviation of actual null rate from target ρ
- **Formula:** `(null_rate − ρ)²`
- **Expected at convergence:** Near 0.0
- **Maximum possible:** ρ² = (5/6)² = 0.6944 (when null_rate=0, i.e. router never picks null)
- **Alert if:** Stays at 0.6944 after >1000 steps (router failed to learn sparsity) or oscillates wildly (unstable routing)
- **Observed:** **0.6944 constant** = maximum value, confirming null_rate=0 throughout. The `1e-2 × L_null` gradient is pushing toward null usage but hasn't moved the router yet. This is expected in early training — the balance loss dominates initially because all tokens hitting real experts creates a strong L_bal gradient.

#### `moe_L_z`

- **What it is:** Z-loss on router logits — penalizes large logit magnitudes to keep softmax well-calibrated
- **Formula:** `mean((logsumexp(logits, dim=-1))²)` across all B×T positions
- **Expected:** 10–50 throughout training (grows slowly as model learns)
- **Alert if:** > 100 (logits exploding — router confidence saturating) or < 5 (unusually small)
- **Observed:** **12.85–15.04** — stable within the healthy range across all 24 steps. Good sign.

#### `moe_expert_counts`

- **What it is:** Array of 20 floats — total token-routing count per real expert, summed across all 8 MoE layers, for the most recent micro-batch
- **Formula:** `bincount(real_expert_indices, minlength=num_experts)` summed across layers
- **Expected at balance:** Roughly equal across all 20 experts
- **Alert if:** max/min ratio > 10 (severe collapse to a few experts) or any expert always near 0 (dead expert)
- **Observed:** Strong imbalance visible in actual data. Example from step 6:
  ```
  E0:1109  E1:341  E2:56   E3:1002  E4:28   E5:22   E6:53   E7:316
  E8:1080  E9:627  E10:483 E11:478  E12:1949 E13:18  E14:51  E15:188
  E16:29   E17:405 E18:858 E19:87
  ```
  E12 received 1949 tokens while E5 received only 22 — an 88× ratio. This shows the router strongly preferring certain experts. **This is the primary issue to fix** — the `L_bal` loss should correct this over more training steps, but the extremely high initial imbalance suggests some experts may need load-balancing interventions.

---

### GPU Metrics

#### `gpu_util_pct` (single device — local rank 0)

- **What it is:** GPU SM utilization % for the rank-0 device, sampled at log time via NVML
- **Expected:** 80–100% during active training
- **Alert if:** < 50% consistently (GPU is waiting for CPU/data/communication)
- **Observed:** 72–100% — generally healthy, dips on steps with very small batches (e.g. step 14: 129 tokens)

#### `gpu_util_all_pct`

- **What it is:** Dict `{"0": %, "1": %, "2": %, "3": %}` — SM utilization per GPU, all 4 devices
- **Expected:** All 4 GPUs at similar utilization (balanced multi-GPU training)
- **Alert if:** One GPU consistently 20+ points below others (communication bottleneck or load imbalance)
- **Observed:**
  | Step | GPU0 | GPU1 | GPU2 | GPU3 |
  |---|---|---|---|---|
  | 1 | 74% | 75% | 74% | 71% |
  | 5 | 100% | 100% | 100% | 98% |
  | 10 | 100% | 100% | 100% | 97% |
  | 21 | 100% | 97% | 73% | 60% |

  Generally good. Some steps show one GPU lagging (variable batch sizes from wikitext-2 cause imbalance).

#### `gpu_idle_pct`

- **What it is:** `100 - gpu_util_pct` for the local rank device
- **Expected:** < 20%
- **Alert if:** > 50% (GPU mostly idle)
- **Observed:** 0–28% — acceptable

#### `gpu_mem_gb` (local rank 0)

- **What it is:** GPU VRAM currently allocated on rank-0 device in GB
- **Observed:** Stable at **22.28 GB** after step 1 warmup

#### `gpu_mem_all_gb`

- **What it is:** Dict `{"0": GB, "1": GB, "2": GB, "3": GB}` — VRAM per GPU
- **Observed:**
  ```
  GPU0: 22.28 GB    GPU1: 17.28 GB
  GPU2: 17.28 GB    GPU3: 22.28 GB
  ```
  GPU0 and GPU3 use ~5GB more than GPU1/GPU2. This is **expected under ZeRO-2**: optimizer states (Adam m/v) are sharded across ranks, but GPU0 and GPU3 hold the embedding table and output projection respectively, which are larger.

---

### Throughput Metrics

| Field | What it measures | Expected | Observed |
|---|---|---|---|
| `tokens_per_sec` | Total tokens processed per second across all 4 GPUs | > 1000 for full training data | 1.3 (step 1 JIT warmup) → 4–29 (tiny wikitext-2 batches) |
| `batches_per_sec` | Optimizer steps per second | Depends on batch size | 0.0376–0.0380 (stable ~26s/step) |
| `step_time_s` | Wall-clock seconds for one optimizer step (all micro-batches) | Varies by batch size | 26.3–33.3s. Step 1 is 33s due to JIT/CUDA graph warmup; all others ~26s |
| `tokens` | Total tokens in this optimizer step across all ranks | Varies with dynamic batching | 43–799 (wikitext-2 has very variable sequence lengths) |
| `total_tokens_processed` | Cumulative tokens since training started | Increases monotonically | 43 → 10,313 after 24 steps |

---

### Training Progress Metrics

| Field | What it measures | Notes |
|---|---|---|
| `epoch` | Current epoch (0-indexed) | 0 throughout (max_train_steps=50 ends before epoch completes) |
| `global_step` | Optimizer step count | 1-indexed; increments only at gradient accumulation boundaries |
| `lr` | Current learning rate from DeepSpeed scheduler | Warmup to 1e-5 (step 10), then cosine decay to ~6.75e-6 by step 24 |
| `timestamp` | Unix timestamp at log time | Useful for computing wall-clock gaps between steps |

---

### System Metrics

| Field | What it measures | Observed |
|---|---|---|
| `cpu_util_pct` | CPU utilization % (all cores, psutil) | 18.5–23.1% — light CPU use; data loading not a bottleneck |
| `cpu_idle_pct` | 100 − cpu_util_pct | 77–81.5% |
| `cpu_mem_used` | *(not logged, only in print output)* | — |

---

## What to Watch During Training

### Router Health (priority order)

1. **`moe_null_rate` not rising** — The most critical issue in this run. After >500 steps it should be approaching 0.3+. If it stays at 0.0 after 1000+ steps, the `L_null` weight (currently `1e-2`) may need to be increased.

2. **`moe_L_bal` staying above 3.0** — Means experts are very unbalanced. Watch `moe_expert_counts` for "dead experts" (count near 0 every step). If certain experts consistently get 5× the average, consider increasing the balance loss weight (`2e-2 → 4e-2`).

3. **`moe_L_z` > 50** — Would indicate router logits growing too large. Currently stable at ~14.

4. **`moe_zero_real_frac` rising** — Once `moe_null_rate` starts increasing, `zero_real_frac` should also rise (toward ~70% at ρ=5/6 convergence). If null_rate rises but zero_real_frac doesn't, the routing is concentrating null picks instead of distributing them.

### Loss Health

- **loss** should decrease overall with occasional oscillation (tiny dataset = high variance)
- **loss_aux / loss** ratio should stay < 0.1 — currently 0.39/18 = 2.2% at step 1, rising to 0.78/0.96 = 81% by step 12 when NTP/MTP converge. On a small dataset this is expected; on real data loss_aux will be a much smaller fraction.
- **loss_mtp ≈ loss_ntp** — They should track each other. If mtp diverges upward, the model is failing to predict 2 tokens ahead.

### GPU Health

- All 4 GPUs should be at 80%+ utilization. Dips below 50% on individual steps are acceptable if batch sizes are variable.
- Memory should be stable. A growing `gpu_mem_gb` across steps indicates a memory leak.

---

## Metric Source Map

| Where it's computed | Where it's stored | Where it's logged |
|---|---|---|
| `MoEGate.forward()` → `L_bal, L_null, L_z, null_rate, counts_real` | `gate.last_*` attributes | `_collect_moe_stats()` in `train.py` → JSONL `moe_*` fields |
| `pynvml.nvmlDeviceGetUtilizationRates()` | `gpu_rows` list in `train_epoch()` | JSONL `gpu_util_all_pct`, `gpu_mem_all_gb` |
| `psutil.cpu_percent()` | local variable | JSONL `cpu_util_pct` |
| `ReversibleMidpointStack.forward()` → accumulated aux | `total_aux_loss` scalar | JSONL `loss_aux` (combined, not split) |
| `model_engine.get_lr()` | DeepSpeed scheduler | JSONL `lr` |
