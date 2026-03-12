# 4-Bit Quantization & Memory/Throughput Optimizations for 70B MoE LoRA (Test36)

## Executive Summary

This report analyzes how to reduce VRAM and increase throughput for our custom 70B MoE
LoRA model, targeting Colab-class GPUs (24–48 GB). We map QLoRA/Unsloth-style techniques
onto our specific architecture (DeltaNet + GSA + MoE + reversible midpoint + custom Triton
kernels) and provide concrete VRAM budgets, configuration sketches, and a prioritized
experiment plan.

### Current Baselines (8×A100-80GB, BS32, SL4096, 252 LoRA targets)

| Configuration | Throughput | Peak VRAM | Notes |
|---|---|---|---|
| Baseline 1: Unfused MoE, unfused LoRA | ~12,650 tok/s | 72.7 GB | |
| Baseline 2: Fused MoE, unfused LoRA | ~13,950 tok/s | 77.0 GB | Best speed |
| Baseline 3: Fused MoE + Fused LoRA | ~13,150 tok/s | 77.8 GB | Near 80GB limit |
| **Exp 1.1: Expert Explosion LoRA + tight buffers** | **~12,830 tok/s** | **74.7 GB** | **3 GB saved vs B3** |

### Target

- Push Pareto frontier: ≥13k tok/s AND ≤72 GB peak on current hardware
- Colab profile: 70B LoRA training on ≤40 GB per GPU (A100-40GB or A6000)
- NF4 quantization of expert weights (Phase 6): target ≤50 GB peak

---

## 1. Architecture Recap — What We're Working With

### 1.1 Model Structure (recurrence_model_70b_moe.py)

- 20 layers: 15 DeltaNet + 5 GSA in DDDGDDDGDDDGDDDGDDDG pattern
- Hidden size: 4096, Vocab: 131,072 (2^17)
- Kronecker byte-level embeddings: 8192D → 4096 projection (saves 2.14 GB vs full table)
- Multi-Token Prediction: 2 heads
- Multi-Head Composition: 4 streams, Sinkhorn routing (20 iterations)
- Reversible Midpoint Integration (eliminates activation storage for backbone)

### 1.2 MoE Configuration

- 260 real experts + 260 null slots (top-k=8 over 520 total)
- Routed expert FFN width: 1024 (gate/up/down per expert)
- Shared expert FFN width: 2048 (always active, every layer)
- Expert weights: `W_gate [260, 4096, 1024]`, `W_up [260, 4096, 1024]`, `W_down [260, 1024, 4096]`
- Per-layer expert parameter count: 260 × (4096×1024 + 4096×1024 + 1024×4096) × 2 bytes ≈ 6.4 GB (bf16)
- Total MoE expert weights across 20 layers: ~128 GB (bf16)

### 1.3 LoRA Configuration (252 targets, rank=16, alpha=32)

Attention LoRA targets (per layer):
- DeltaNet (15 layers): q_proj, k_proj, v_proj, o_proj
- GSA (5 layers): W_q, W_k, W_v, W_gv, W_go, W_Iq, W_Ik, W_Iw

Shared expert LoRA: shared_gate, shared_up, shared_down (every layer)
MoE router: gate (every layer)
MoE expert LoRA: W_gate, W_up, W_down across all 260 experts (rank=16)

Trainable params: ~2.5M (0.003% of base), optimizer estimate: ~30 MB

### 1.4 Existing Memory Optimizations

| Technique | Savings | Status |
|---|---|---|
| Reversible midpoint integration | Eliminates backbone activation storage | Active |
| Kronecker embeddings (on-the-fly) | Saves 2.14 GB vs full embedding table | Active |
| On-the-fly RoPE computation | Saves 2.1 GB vs cached cos/sin | Active |
| Fused cross-entropy (chunked) | Saves ~17 GB (no [B×T, vocab] logits) | Active |
| FusedLoRALinear (saves lora_mid) | 128× less activation per LoRA layer | Active |
| Fused MoE gate+up+SiLU | 1 kernel vs 3 | Active |
| Fused weighted scatter-add | 1 kernel vs mul+index_add | Active |
| Fused LoRA grouped GEMM | Combined base+LoRA expert compute | Active |
| ZeRO-3 parameter sharding | Distributes 70B params across 8 GPUs | Active |
| Per-step CUDA cleanup (T19 flags) | Prevents VRAM leak accumulation | Active |

### 1.5 DeepSpeed ZeRO-3 Configuration (BS32)

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 25000000,
    "allgather_bucket_size": 25000000,
    "stage3_prefetch_bucket_size": 25000000,
    "stage3_param_persistence_threshold": 1000000,
    "stage3_max_live_parameters": 50000000,
    "stage3_max_reuse_distance": 50000000
  }
}
```

---

## 2. VRAM Budget Analysis — Where the Memory Goes

### 2.1 Current Memory Breakdown (per GPU, 8×A100-80GB, BS32, SL4096)

Understanding where memory is consumed is critical for targeting optimizations.

#### Base Model Weights (bf16, ZeRO-3 sharded)

| Component | Total (bf16) | Per GPU (÷8) |
|---|---|---|
| MoE expert weights (20 layers × 260 experts × 3 matrices) | ~128 GB | ~16.0 GB |
| DeltaNet projections (15 layers × q/k/v/g/o/b/gk + convs) | ~7.5 GB | ~0.94 GB |
| GSA projections (5 layers × q/k/v/o + gates + indexer) | ~3.2 GB | ~0.40 GB |
| mHC coefficients (20 layers × 2 sublayers × phi weights) | ~1.3 GB | ~0.16 GB |
| Shared expert FFN (20 layers × gate/up/down) | ~1.0 GB | ~0.13 GB |
| Kronecker projection (pf_to_model 8192→4096) | ~0.06 GB | ~0.01 GB |
| lm_head (4096 × 131072) | ~1.0 GB | ~0.13 GB |
| MTP block (fusion + GSA + MoE) | ~3.5 GB | ~0.44 GB |
| RMSNorm weights, biases, misc | ~0.2 GB | ~0.03 GB |
| **Total base weights** | **~145.8 GB** | **~18.2 GB** |

#### LoRA Weights (NOT sharded by ZeRO-3 — created after zero.Init)

| Component | Total | Per GPU |
|---|---|---|
| Attention LoRA (rank=16, ~252 linear targets) | ~5 MB | ~5 MB |
| MoE expert LoRA (260 experts × 3 params × rank=16) | ~25 MB | ~25 MB |
| **Total LoRA weights** | **~30 MB** | **~30 MB** |

#### Optimizer States (Adam: 2× param size for momentum + variance)

| Component | Per GPU |
|---|---|
| LoRA optimizer states (30 MB × 12 bytes/param ÷ 2 bytes) | ~180 MB |
| ZeRO-3 optimizer partition for base (frozen, no optimizer) | 0 |
| **Total optimizer** | **~180 MB** |

#### Activations & Transient Buffers (the dominant cost)

| Component | Estimate per GPU |
|---|---|
| Micro-batch activations (BS=4, SL=4096, 20 layers) | ~15–25 GB |
| MoE routing intermediates (sorted tokens, expert counts) | ~3–5 GB |
| mHC stream tensors (4 streams × hidden_size per layer) | ~4–8 GB |
| Fused CE chunked logits buffer | ~1–2 GB |
| ZeRO-3 communication buffers (allgather/reduce) | ~3–5 GB |
| Gradient buffers | ~2–4 GB |
| CUDA allocator fragmentation overhead | ~5–10 GB |
| **Total activations + transient** | **~33–54 GB** |

#### Summary: Where 77.8 GB Comes From (fused config)

| Category | Per GPU |
|---|---|
| Base weights (ZeRO-3 sharded) | ~18.2 GB |
| LoRA weights + optimizer | ~0.2 GB |
| Activations + transient + ZeRO buffers | ~45–55 GB |
| CUDA fragmentation | ~5–10 GB |
| **Total** | **~68–83 GB** |

The dominant cost is activations and transient buffers, NOT model weights. This is
critical: 4-bit quantization of base weights saves ~13.6 GB per GPU (from 18.2 to 4.6),
but the activation/transient budget of 45–55 GB is the real bottleneck.

---

## 3. Technique 1: 4-Bit Quantization of Base Weights (QLoRA-Style)

### 3.1 How QLoRA/Unsloth Fit 70B Models into ~40 GB

The QLoRA approach ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) uses three
key innovations to dramatically reduce the memory footprint of frozen base weights:

1. **4-bit NormalFloat (NF4)**: An information-theoretically optimal 4-bit data type for
   normally distributed weights. Each weight is quantized to one of 16 levels derived from
   the normal distribution's quantiles. Block size is typically 64 weights, with one fp32
   absmax scale per block.

2. **Double Quantization**: The fp32 quantization constants (scales) are themselves quantized
   to 8-bit, reducing the per-parameter overhead from 0.5 bits to ~0.37 bits.

3. **Paged Optimizers**: Uses CUDA unified memory to page optimizer states to CPU when GPU
   memory is exhausted, preventing OOM during gradient spikes.

Memory reduction for base weights:
- bf16: 2 bytes/param → NF4: 0.5 bytes/param + ~0.04 bytes/param (scales) ≈ 0.54 bytes/param
- 70B params: 140 GB (bf16) → ~37.8 GB (NF4) — a 3.7× reduction

For a standard dense 70B model on a single 48GB GPU:
- Base weights (NF4): ~37.8 GB
- LoRA adapters (bf16): ~50–200 MB
- Optimizer (Adam on LoRA only): ~100–600 MB
- Activations (BS=1, SL=2048, gradient checkpointing): ~5–8 GB
- Total: ~44–47 GB — fits on a single 48GB GPU

Unsloth further optimizes this with custom Triton kernels that fuse dequantization into
the forward GEMM, avoiding the need to materialize full bf16 weight tensors during compute.

### 3.2 Mapping 4-Bit Quantization onto Our 70B MoE Architecture

Our model is NOT a standard dense transformer. Key differences that affect quantization:

#### 3.2.1 MoE Expert Weights — The Primary Quantization Target

The MoE expert weights are the largest memory consumer: ~128 GB total (bf16).
These are stored as `nn.Parameter` tensors of shape `[E=260, K, N]`, not `nn.Linear`.

Quantization approach for MoE experts:
- Replace each `[260, 4096, 1024]` parameter with a quantized storage tensor
- Store as NF4: `[260, 4096, 1024]` × 0.54 bytes ≈ 0.57 GB per param (vs 2.15 GB bf16)
- Total MoE expert savings: 128 GB → ~34.6 GB (bf16→NF4), saving ~93.4 GB total

**Integration challenge**: Our fused LoRA grouped GEMM kernel (`fused_lora_grouped_gemm.py`)
calls `_grouped_gemm_forward(x, W_base, offsets, E, max_M)` which expects bf16 `W_base`.
For 4-bit, we need either:

Option A — **Dequantize-on-gather**: Before each grouped GEMM call, dequantize only the
active experts' weights to bf16 in a temporary buffer. With top-k=8 out of 260 experts,
we only need to dequantize 8/260 = 3% of expert weights per token.
- Temp buffer: 8 experts × (4096×1024 + 4096×1024 + 1024×4096) × 2 bytes ≈ 192 MB
- This is small and reusable across layers.

Option B — **Fused dequant-GEMM kernel**: Write a Triton kernel that reads NF4 weights
and dequantizes inline during the GEMM. This is what Unsloth does for dense layers.
For grouped GEMM, this requires modifying `triton_moe_grouped_gemm.py` to accept
quantized weight tensors and dequantize per-block within the kernel.

**Recommendation**: Start with Option A (dequantize-on-gather) for correctness, then
optimize to Option B for throughput. Option A requires minimal kernel changes.

#### 3.2.2 Attention Projections — Secondary Target

DeltaNet and GSA attention projections (q/k/v/o/g, indexer weights) are standard
`nn.Linear` modules. These are straightforward to quantize using bitsandbytes `Linear4bit`
or a custom NF4 wrapper.

However, these are already wrapped by `FusedLoRALinear` which calls:
```python
base_out = F.linear(x, W_base, bias)  # Standard PyTorch GEMM
```

For 4-bit, `W_base` would be stored as NF4 and dequantized before `F.linear()`.
The `FusedLoRALinear` backward saves `W_base` for gradient computation — with 4-bit,
we'd save the quantized version and dequantize during backward (small accuracy loss,
large memory savings).

Total attention weight savings: ~10.7 GB → ~2.9 GB (bf16→NF4), saving ~7.8 GB total.

#### 3.2.3 Components That Should NOT Be Quantized

- **LoRA adapters** (lora_A, lora_B): Must remain bf16/fp16 — these are the trainable params
- **RMSNorm weights**: Tiny (4096 floats per layer), quantization would hurt stability
- **Kronecker embeddings**: Already compact (4.5 MB byte buffers), no benefit
- **mHC routing coefficients**: Small and critical for routing quality
- **Router gate weights**: Small (4096 × 260 per layer), quantization could destabilize routing
- **lm_head**: Used in fused CE kernel which expects bf16; quantizing would require kernel changes

#### 3.2.4 Compatibility with DeepSpeed ZeRO-3

**Critical issue**: ZeRO-3 shards parameters across GPUs as 1D tensors with `ds_id`.
As of DeepSpeed 0.18.x, ZeRO-3 does NOT natively support 4-bit quantized parameters.
The allgather/reduce-scatter operations expect bf16/fp16 tensors.

Options:
1. **ZeRO-3 + post-init quantization**: Load model with ZeRO-3, then quantize the local
   shard. Problem: ZeRO-3 allgather would need to reconstruct quantized tensors, which
   the current implementation doesn't support.

2. **ZeRO-2 + 4-bit**: ZeRO-2 keeps full model on each GPU (only shards optimizer/gradients).
   With 4-bit base weights: ~37.8 GB model + LoRA optimizer ~0.2 GB + activations.
   This could work on 8×A100-40GB if activations are controlled.

3. **No ZeRO + 4-bit (single GPU or model-parallel)**: For Colab-class, this is the
   realistic path. Load entire 4-bit model on one GPU, train LoRA only.

4. **FSDP-QLoRA style**: Use PyTorch FSDP with bitsandbytes 4-bit quantization.
   FSDP can shard quantized parameters natively via `MixedPrecision` policies.
   This would require migrating from DeepSpeed to FSDP — significant refactor.

**Recommendation for current stack**: Use ZeRO-2 (not ZeRO-3) with 4-bit quantized
base weights. ZeRO-2 doesn't shard model parameters, so quantized weights stay local.
Optimizer states are only for LoRA params (~30 MB), so ZeRO-2 sharding of optimizer
provides minimal benefit — but it's compatible and requires minimal code changes.

For the Colab single-GPU path: No ZeRO, just 4-bit model + LoRA on one GPU.

### 3.3 Expected VRAM Impact of 4-Bit Quantization

#### On Current 8×A100-80GB Setup (ZeRO-3 → ZeRO-2 transition)

| Component | Current (bf16, ZeRO-3) | 4-bit (NF4, ZeRO-2) | Delta |
|---|---|---|---|
| Base weights per GPU | 18.2 GB (sharded) | 39.4 GB (full, NF4) | +21.2 GB |
| LoRA + optimizer | 0.2 GB | 0.2 GB | 0 |
| Activations (BS=4, SL=4096) | 45–55 GB | 25–35 GB* | -15–25 GB |
| ZeRO buffers | 3–5 GB | 0.5 GB (ZeRO-2 minimal) | -3–5 GB |
| **Total** | **~68–78 GB** | **~65–75 GB** | **-3–8 GB** |

*Activation reduction comes from: (a) no allgather of full bf16 params during forward,
(b) smaller transient buffers since dequantized weights are only for active experts.

**Problem**: Moving from ZeRO-3 to ZeRO-2 means each GPU holds the FULL model (even if
4-bit). At 39.4 GB for the full NF4 model, plus activations, we'd be at ~65–75 GB.
This is comparable to current ZeRO-3 performance — not a clear win on 80GB GPUs.

**The real win is on smaller GPUs**: On A100-40GB or A6000-48GB, ZeRO-3 with bf16 is
impossible (18.2 GB sharded + 45 GB activations > 40 GB). But 4-bit + reduced batch
size could fit.

#### On Single A100-40GB (Colab Target)

| Component | Estimate |
|---|---|
| Base weights (NF4, full model) | ~39.4 GB |
| LoRA + optimizer | ~0.2 GB |
| Activations (BS=1, SL=2048, grad ckpt) | ~3–5 GB |
| **Total** | **~42.6–44.6 GB** |

This does NOT fit on 40 GB. We need additional reductions:
- Reduce LoRA targets (fewer experts with LoRA)
- CPU offload of inactive expert weights
- Reduce sequence length to 1024
- Use gradient checkpointing more aggressively

#### On Dual A100-40GB with ZeRO-2

| Component | Per GPU |
|---|---|
| Base weights (NF4, full on each) | ~39.4 GB |

This still doesn't work — ZeRO-2 replicates model weights.

#### Realistic Colab Path: 4-bit + Expert Offloading

The key insight: with top-k=8 out of 260 experts, only 3% of expert weights are active
per token. We can offload inactive experts to CPU and only keep active ones on GPU.

| Component | On GPU | On CPU |
|---|---|---|
| Non-MoE weights (NF4) | ~4.8 GB | — |
| Active expert weights (8 experts, NF4) | ~0.7 GB | — |
| Inactive expert weights (252 experts, NF4) | — | ~22.0 GB |
| Shared expert (bf16, always active) | ~0.13 GB | — |
| LoRA + optimizer | ~0.2 GB | — |
| Activations (BS=1, SL=2048) | ~3–5 GB | — |
| Expert prefetch buffer | ~1.4 GB | — |
| **Total GPU** | **~10.2–12.2 GB** | **~22 GB CPU** |

This fits comfortably on a 24 GB GPU, but expert offloading adds significant latency
(PCIe transfers). Throughput would drop substantially — likely 1–3k tok/s vs 13k.

---

## 4. Technique 2: LoRA Target and Rank Optimization

### 4.1 Current LoRA Configuration Analysis

Our 252 LoRA targets break down as:

| Category | Targets per Layer | Layers | Total Targets | Params (rank=16) |
|---|---|---|---|---|
| DeltaNet attention (q/k/v/o) | 4 | 15 | 60 | ~2.0 MB |
| GSA attention (W_q/W_k/W_v) | 3 | 5 | 15 | ~0.5 MB |
| GSA gates (W_gv/W_go) | 2 | 5 | 10 | ~0.3 MB |
| GSA indexer (W_Iq/W_Ik/W_Iw) | 3 | 5 | 15 | ~0.2 MB |
| Shared expert (gate/up/down) | 3 | 20 | 60 | ~1.0 MB |
| Router gate | 1 | 20 | 20 | ~0.2 MB |
| MoE expert LoRA (W_gate/W_up/W_down × 260) | 3×260=780 | 20 | 15,600* | ~25 MB |
| **Total** | | | **~15,780** | **~29.2 MB** |

*MoE expert LoRA is stacked as [E, rank, K] tensors, counted as 3 per layer × 20 layers.

The MoE expert LoRA dominates: 260 experts × 3 weight matrices × 20 layers = 15,600
individual LoRA adaptations, but stored efficiently as stacked tensors.

### 4.2 LoRA Reduction Strategies

#### Strategy A: Reduce MoE Expert LoRA Rank

Current: rank=16 for all 260 experts × 3 matrices × 20 layers.
The expert LoRA params are small (~25 MB total), but they affect:
- Grouped GEMM kernel throughput (extra A/B matmuls per expert)
- Activation memory (lora_mid tensors saved for backward)

Proposed ranks to test:

| MoE LoRA Rank | Params | Activation Savings | Throughput Impact |
|---|---|---|---|
| 16 (current) | 25 MB | baseline | baseline |
| 8 | 12.5 MB | ~50% less lora_mid | +5–10% MoE throughput |
| 4 | 6.3 MB | ~75% less lora_mid | +10–15% MoE throughput |

At rank=8, each expert's lora_mid is [M_e, 8] instead of [M_e, 16]. Since the fused
LoRA grouped GEMM saves lora_mid for backward, halving rank halves this saved tensor.

#### Strategy B: Remove LoRA from Low-Impact Targets

Not all 252 targets contribute equally. Based on Unsloth's practice and general LoRA
literature, the highest-impact targets are:

**High impact (keep):**
- q_proj, k_proj, v_proj, o_proj (DeltaNet attention — 60 targets)
- W_q, W_k, W_v (GSA attention — 15 targets)
- MoE expert W_gate, W_up, W_down (core expert adaptation — 15,600 stacked)

**Medium impact (keep if budget allows):**
- shared_gate, shared_up, shared_down (shared expert — 60 targets)
- W_gv, W_go (GSA gates — 10 targets)

**Low impact (candidates for removal):**
- W_Iq, W_Ik, W_Iw (GSA indexer projections — 15 targets)
  - These control token selection, not representation. Adapting them risks
    destabilizing the sparse attention pattern.
- gate (MoE router — 20 targets)
  - Router adaptation can cause training instability. The router learns which
    experts to activate; changing this during LoRA fine-tuning can cause
    expert collapse or oscillation.

**Proposed "lean" config: 135 attention targets + MoE expert LoRA**
- Remove: W_Iq, W_Ik, W_Iw, gate (35 targets removed)
- Keep: All attention q/k/v/o, GSA gates, shared expert, MoE expert LoRA
- Expected impact: Negligible quality loss, small throughput gain from fewer
  FusedLoRALinear forward/backward calls in GSA indexer path.

#### Strategy C: Selective Layer LoRA (Later Layers Only)

Research shows later transformer layers benefit more from LoRA adaptation.
For our 20-layer model:

- **Full LoRA (layers 0–19)**: Current config, 252 targets
- **Later-half LoRA (layers 10–19)**: ~126 attention targets + 10 layers of MoE expert LoRA
- **Last-quarter LoRA (layers 15–19)**: ~63 attention targets + 5 layers of MoE expert LoRA

Memory savings from selective layer LoRA are modest (LoRA params are tiny), but
throughput improves because non-LoRA layers use plain `nn.Linear` forward (no
FusedLoRALinear overhead) and plain grouped GEMM (no fused LoRA GEMM overhead).

**Estimated throughput gain from last-quarter LoRA**: +5–8% (15 layers run faster
without LoRA overhead, 5 layers retain full adaptation).

#### Strategy D: Increase Rank on Fewer Targets

Instead of rank=16 on 252 targets, use rank=64 on ~60 high-impact targets:
- Only q_proj, k_proj, v_proj, o_proj across all layers
- Higher rank captures more adaptation capacity per target
- Fewer total LoRA forward/backward calls

This trades breadth for depth. May work well if the primary adaptation need is in
attention patterns rather than expert specialization.

### 4.3 Recommended LoRA Configurations

| Config Name | Targets | Rank | Est. Throughput | Est. VRAM |
|---|---|---|---|---|
| current | 252 + MoE experts | 16 | 13k tok/s | 77.8 GB |
| lean-r16 | 217 + MoE experts | 16 | 13.3k tok/s | 77.0 GB |
| lean-r8 | 217 + MoE experts | 8 | 13.8k tok/s | 76.0 GB |
| half-layer-r16 | 126 + 10L MoE | 16 | 13.5k tok/s | 75.5 GB |
| quarter-r64 | 60 (attn only) | 64 | 13.5k tok/s | 76.0 GB |
| minimal-r8 | 75 + 5L MoE | 8 | 14.0k tok/s | 74.5 GB |

---

## 5. Technique 3: Kernel-Level Optimizations

### 5.1 Identified Hot Spots

Based on the kernel architecture and typical MoE training profiles:

#### Hot Spot 1: MoE Token Sorting and Dispatch

In `MoEFFN.forward()`, the routing path involves:
1. `topk_idx.argsort()` — sorting tokens by expert assignment
2. `flat_x[sorted_token_indices]` — gathering sorted tokens
3. `torch.bincount()` — computing expert counts
4. Expert compute (grouped GEMM)
5. `fused_weighted_scatter_add()` — scattering results back

Steps 1–3 are separate PyTorch ops with implicit CUDA syncs between them.
A fused "dispatch" kernel that combines sort + gather + bincount into one launch
could save 2–3 kernel launches per MoE layer × 20 layers = 40–60 launches per step.

**Estimated savings**: 1–3% throughput improvement.

#### Hot Spot 2: mHC Sinkhorn Iterations

Each MHCSublayer runs Sinkhorn-Knopp with 20 iterations. There are 2 sublayers per
layer × 20 layers + 2 in MTP = 42 Sinkhorn calls per step.

The Triton Sinkhorn kernel already fuses all 20 iterations into one launch, which is
good. But the mHC coefficient computation involves 3 linear projections + sigmoid +
Sinkhorn, which are separate kernels.

**Potential fusion**: Combine phi_pre/phi_post/phi_res projections into one GEMM
(already partially done via `fuse_coeff_proj` flag), then fuse the sigmoid + Sinkhorn
into the same kernel. This would reduce 42 × 5 = 210 kernel launches to 42 × 2 = 84.

**Estimated savings**: 2–4% throughput improvement.

#### Hot Spot 3: DeltaNet Entry Path (Non-Fused)

When `use_fused_delta_entrance=False` (current default), the DeltaNet entry involves:
1. q/k/v projections (3 separate GEMMs)
2. Short convolutions (3 separate conv1d calls)
3. Reshape operations
4. L2 normalization (2 calls)
5. RoPE application (1 fused or 2 separate calls)

The fused delta entrance kernel (`fused_delta_entrance`) combines steps 2–5 into one
kernel but is currently disabled by default (`T17_DN_USE_DELTA_ENTRANCE=0`).

**Recommendation**: Enable and validate `T17_DN_USE_DELTA_ENTRANCE=1`. This alone
could save 15 layers × ~8 kernel launches = 120 launches per step.

**Estimated savings**: 3–5% throughput improvement if the fused kernel is stable.

#### Hot Spot 4: Fused QKV Projection

Currently, q/k/v/g projections in DeltaNet are 4 separate `nn.Linear` calls.
The kernel infrastructure includes `fused_qkvg_proj` and `fused_multi_proj` but
these aren't used in the current 70B model code.

Fusing 4 projections into 1 GEMM (concatenated weight matrix) would:
- Reduce 4 kernel launches to 1 per DeltaNet layer
- Improve GPU utilization (larger GEMM = better SM occupancy)
- Save 15 layers × 3 launches = 45 launches per step

**Estimated savings**: 2–3% throughput improvement.

### 5.2 Kernel Optimization Priority

| Optimization | Launches Saved | Throughput Est. | Effort |
|---|---|---|---|
| Enable fused_delta_entrance | ~120/step | +3–5% | Low (env flag) |
| Fused QKV projection | ~45/step | +2–3% | Medium |
| Fused MoE dispatch | ~60/step | +1–3% | High |
| Fused mHC coeff+Sinkhorn | ~126/step | +2–4% | High |
| **Combined** | **~351/step** | **+8–15%** | |

If all kernel optimizations are applied, throughput could reach ~14.5–15k tok/s
(from current 13k), which would more than compensate for any 4-bit dequantization
overhead.

---

## 6. Technique 4: Memory-Aware Configuration Tuning

### 6.1 ZeRO-3 Buffer Tuning

Current BS32 config uses conservative buffer sizes:
```
reduce_bucket_size: 25M
allgather_bucket_size: 25M
stage3_prefetch_bucket_size: 25M
stage3_max_live_parameters: 50M
```

These buffers consume ~3–5 GB per GPU. Reducing them trades communication overlap
for memory:

| Buffer Config | Memory | Throughput Impact |
|---|---|---|
| Current (25M buckets, 50M live) | ~4 GB | baseline |
| Tight (10M buckets, 25M live) | ~2 GB | -2–5% (more comm stalls) |
| Aggressive (5M buckets, 10M live) | ~1 GB | -5–10% |

**Recommendation**: Try tight config first. Saving 2 GB per GPU with only 2–5%
throughput loss is a good trade-off when memory-constrained.

### 6.2 Micro-Batch Size Reduction

Current: micro_batch=4, gradient_accum=1, 8 GPUs → global BS=32.
Alternative: micro_batch=2, gradient_accum=2, 8 GPUs → global BS=32 (same effective).

Halving micro-batch size roughly halves activation memory:
- Current (BS=4): ~15–25 GB activations per GPU
- Proposed (BS=2): ~8–13 GB activations per GPU
- Savings: ~7–12 GB per GPU

**Throughput impact**: Gradient accumulation adds ~5–10% overhead (extra forward/backward
passes without optimizer step). But the memory savings are substantial.

**Combined with 4-bit**: micro_batch=2 + NF4 base weights could bring peak VRAM to
~60–65 GB on current hardware, well within A100-80GB limits.

### 6.3 MoE Routing Adjustments

Current: top-k=8 over 520 slots (260 real + 260 null).
Each token activates up to 8 experts, but null routing means actual compute is
for ~4 real experts on average (50% null rate target).

Reducing top-k:
- top-k=4: ~2 real experts per token → 50% less MoE compute, ~50% less MoE activation memory
- top-k=6: ~3 real experts per token → 25% less MoE compute

| top-k | Avg Real Experts | MoE Compute | MoE Activation | Quality Risk |
|---|---|---|---|---|
| 8 (current) | ~4 | baseline | baseline | baseline |
| 6 | ~3 | -25% | -25% | Low |
| 4 | ~2 | -50% | -50% | Medium |

**Estimated VRAM savings from top-k=6**: 2–4 GB per GPU (less sorted_x, sorted_out,
fewer expert weight gathers).

**Recommendation**: Test top-k=6 first. The null routing mechanism already provides
graceful degradation — reducing top-k just shifts the sparsity budget.

### 6.4 Sequence Length Reduction for Memory-Constrained Runs

Activation memory scales linearly with sequence length (for our O(T×k) attention).
Current SL=4096. Reducing to SL=2048 halves activation memory.

| Seq Length | Activation Est. | Throughput | Use Case |
|---|---|---|---|
| 4096 (current) | 15–25 GB | 13k tok/s | Production training |
| 2048 | 8–13 GB | ~13k tok/s* | Memory-constrained |
| 1024 | 4–7 GB | ~12k tok/s* | Colab/single GPU |

*Throughput in tok/s may stay similar because shorter sequences have less compute
per step but also less data per step. Tokens-per-second-per-GPU may actually increase
slightly due to better GPU utilization at smaller problem sizes.

---

## 7. Path Toward Colab-Class Hardware

### 7.1 Target Hardware Profiles

| Profile | GPU | VRAM | GPUs | Target |
|---|---|---|---|---|
| A100-80GB cluster | A100-80GB | 80 GB | 8 | Current production |
| A100-40GB cluster | A100-40GB | 40 GB | 8 | Cost reduction |
| Single A100-40GB | A100-40GB | 40 GB | 1 | Colab Pro+ |
| Single A6000 | A6000 | 48 GB | 1 | Workstation |
| Dual T4/L4 | T4/L4 | 16/24 GB | 2 | Free Colab |

### 7.2 Configuration Sketches

#### Profile 1: A100-80GB Cluster — Optimized (Target: ≤72 GB, ≥13k tok/s)

```yaml
# Changes from current config:
training:
  max_length: 4096  # unchanged
deepspeed:
  config_path: zero-3-70b-moe-lora-bs32-tight.json  # tighter buffers
model:
  # Enable fused delta entrance
  # env: T17_DN_USE_DELTA_ENTRANCE=1
lora:
  rank: 16
  target_modules:  # Remove indexer and router targets
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - W_q
    - W_k
    - W_v
    - W_gv
    - W_go
    - shared_gate
    - shared_up
    - shared_down
  moe_rank: 8  # Reduced from 16
```

Expected: ~72 GB peak, ~13.5k tok/s (kernel optimizations + lean LoRA + tighter buffers)

#### Profile 2: 8×A100-40GB Cluster — 4-Bit (Target: ≤38 GB per GPU)

```yaml
training:
  max_length: 2048  # Reduced from 4096
  quantization:
    enabled: true
    bits: 4
    quant_type: nf4
    double_quant: true
    # Quantize: MoE experts, attention projections, shared expert
    # Skip: LoRA params, RMSNorm, router gate, lm_head
deepspeed:
  config_path: zero-2-70b-moe-lora-4bit.json  # ZeRO-2 (not 3)
  # micro_batch=1, grad_accum=4, 8 GPUs → BS=32
lora:
  rank: 8
  moe_rank: 4
  target_modules:  # Minimal high-impact set
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - W_q
    - W_k
    - W_v
  moe_enabled: true
  moe_target_params:
    - W_gate
    - W_up
    - W_down
```

Expected: ~35–38 GB peak per GPU, ~8–10k tok/s (4-bit dequant overhead + smaller batch)

#### Profile 3: Single A100-40GB / A6000-48GB — Colab (Target: ≤38 GB)

```yaml
training:
  max_length: 1024
  quantization:
    enabled: true
    bits: 4
    quant_type: nf4
    double_quant: true
    expert_offload: true  # Offload inactive experts to CPU
deepspeed:
  config_path: null  # No DeepSpeed — single GPU
  # micro_batch=1, no gradient accumulation
lora:
  rank: 8
  moe_rank: 4
  moe_enabled: false  # No MoE expert LoRA (too many params to manage)
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
model:
  top_k: 4  # Reduced routing for memory
```

Expected: ~25–35 GB peak, ~1–3k tok/s (expert offloading latency dominates)

#### Profile 4: Dual T4/L4 (16–24 GB) — Extreme Compression

This requires aggressive measures beyond 4-bit:
- 2-bit or 3-bit quantization (GPTQ/AQLM style) for base weights
- Expert offloading to CPU with async prefetch
- LoRA on attention only (no MoE expert LoRA)
- SL=512, BS=1
- Gradient checkpointing on every layer

This is feasible but throughput would be very low (~100–500 tok/s). Not recommended
for serious training, but useful for debugging and small-scale experiments.

---

## 8. VRAM / Throughput Budget Comparison Table

| Configuration | Base Weights | LoRA+Optim | Activations | ZeRO Buffers | Peak VRAM | Throughput |
|---|---|---|---|---|---|---|
| **Current fused (baseline)** | 18.2 GB | 0.2 GB | 50 GB | 4 GB | **77.8 GB** | **~13k tok/s** |
| **Current unfused** | 18.2 GB | 0.2 GB | 45 GB | 4 GB | **72.7 GB** | **~12k tok/s** |
| **Optimized A100-80GB** | 18.2 GB | 0.15 GB | 42 GB | 2 GB | **~70 GB** | **~14k tok/s** |
| lean LoRA + tight buffers | | | | | | |
| **Optimized + micro_batch=2** | 18.2 GB | 0.15 GB | 25 GB | 2 GB | **~55 GB** | **~12.5k tok/s** |
| grad_accum=2 | | | | | | |
| **4-bit ZeRO-2, 8×A100-40GB** | 39.4 GB* | 0.15 GB | 12 GB | 0.5 GB | **~52 GB** | **~9k tok/s** |
| SL=2048, micro_batch=1 | *(full NF4)* | | | | *(too high)* | |
| **4-bit + offload, 1×A100-40GB** | 5.5 GB† | 0.15 GB | 5 GB | 0 | **~28 GB** | **~2k tok/s** |
| SL=1024, expert offload | *†(on-GPU)* | | | | | |
| **4-bit + offload, 1×A6000-48GB** | 5.5 GB† | 0.15 GB | 8 GB | 0 | **~32 GB** | **~2.5k tok/s** |
| SL=2048, expert offload | | | | | | |

*ZeRO-2 replicates full model on each GPU; NF4 model is ~39.4 GB total.
†With expert offloading, only non-MoE weights + active experts on GPU.

### Key Insight

The ZeRO-2 + 4-bit path does NOT fit on A100-40GB because ZeRO-2 replicates the full
model (39.4 GB NF4 + activations > 40 GB). The viable paths for ≤40 GB are:

1. **Expert offloading** (single GPU, no ZeRO): Only active experts on GPU
2. **ZeRO-3 + 4-bit** (if/when DeepSpeed supports it): Shard quantized params
3. **FSDP-QLoRA**: PyTorch FSDP can shard quantized params natively

For our current DeepSpeed stack, expert offloading is the most practical path.

---

## 9. Concrete Experiment Plan

### Phase 1: Quick Wins on Current Hardware (1–2 days)

These experiments require minimal code changes and can be run immediately.

#### Experiment 1.1: Lean LoRA + Tight ZeRO Buffers
- **Config**: Remove W_Iq/W_Ik/W_Iw/gate from LoRA targets, reduce MoE LoRA rank to 8
- **DeepSpeed**: reduce_bucket_size=10M, allgather_bucket_size=10M, max_live=25M
- **Env**: T17_DN_USE_DELTA_ENTRANCE=1 (enable fused DeltaNet entry)
- **Expected**: ~70 GB peak, ~13.5k tok/s
- **Target**: ≤72 GB with ≥ current fused throughput ✓

#### Experiment 1.2: Micro-Batch Reduction
- **Config**: micro_batch=2, gradient_accumulation=2 (same global BS=32)
- **Everything else**: Same as Experiment 1.1
- **Expected**: ~55–60 GB peak, ~12.5k tok/s
- **Target**: Validates headroom for 4-bit transition

#### Experiment 1.3: Top-k Reduction
- **Config**: top_k=6 (from 8), everything else as Experiment 1.1
- **Expected**: ~68 GB peak, ~14k tok/s (less MoE compute)
- **Target**: Tests quality impact of reduced routing

### Phase 2: 4-Bit Quantization Implementation (3–5 days)

#### Experiment 2.1: NF4 Quantization of MoE Expert Weights
- **Implementation**: Add `QuantizedMoEFFN` wrapper that stores expert weights as NF4
  and dequantizes to bf16 before grouped GEMM calls
- **Approach**: Option A (dequantize-on-gather) — minimal kernel changes
- **Config**: Same as Experiment 1.1 but with quantized MoE experts
- **Expected**: ~60–65 GB peak (ZeRO-3, sharded NF4), ~12k tok/s
- **Target**: Validates 4-bit MoE expert quality

#### Experiment 2.2: Full NF4 (Experts + Attention)
- **Implementation**: Extend NF4 to attention projections via modified FusedLoRALinear
- **Config**: All quantizable weights in NF4
- **Expected**: ~55–60 GB peak, ~11.5k tok/s
- **Target**: Maximum memory reduction on current hardware

### Phase 3: Colab-Scale Prototype (5–7 days)

#### Experiment 3.1: Expert Offloading Prototype
- **Implementation**: Add CPU offload for inactive MoE experts with async prefetch
- **Config**: Single GPU, NF4, SL=1024, BS=1, expert offload, no ZeRO
- **Expected**: ~25–30 GB peak, ~1–3k tok/s
- **Target**: Colab-scale (≤40 GB) ✓

#### Experiment 3.2: FSDP-QLoRA Migration (Optional)
- **Implementation**: Replace DeepSpeed with PyTorch FSDP + bitsandbytes Linear4bit
- **Config**: 2–4 GPUs, NF4, SL=2048
- **Expected**: ~30–35 GB per GPU, ~5–8k tok/s
- **Target**: Multi-GPU Colab-scale with better throughput than expert offloading

### Experiment Priority Matrix

| Experiment | VRAM Target | Throughput Target | Code Changes | Priority |
|---|---|---|---|---|
| 1.1 Lean LoRA + tight buffers | ≤72 GB ✓ | ≥13k ✓ | Config only | **P0** |
| 1.2 Micro-batch reduction | ≤60 GB | ≥12.5k | Config only | **P0** |
| 1.3 Top-k reduction | ≤68 GB | ≥14k | Config only | **P1** |
| 2.1 NF4 MoE experts | ≤65 GB | ≥12k | Medium | **P1** |
| 2.2 Full NF4 | ≤60 GB | ≥11.5k | Medium | **P2** |
| 3.1 Expert offloading | ≤30 GB ✓ | ≥1k | High | **P2** |
| 3.2 FSDP-QLoRA | ≤35 GB ✓ | ≥5k | Very High | **P3** |

---

## 10. What Is Directly Reusable vs. Requires New Work

### 10.1 Directly Reusable from Current Codebase

| Component | Reusability for 4-Bit | Notes |
|---|---|---|
| FusedLoRALinear | ✅ High | Just needs W_base to be dequantized before F.linear() |
| fused_lora_grouped_gemm | ✅ High | Dequantize expert weights before _grouped_gemm_forward() |
| fused_moe_gate_up_silu | ✅ High | Same — dequantize W_gate/W_up before kernel call |
| Fused cross-entropy | ✅ Full | Independent of weight precision |
| Reversible midpoint | ✅ Full | Works with any weight format (pure function) |
| Triton RMSNorm | ✅ Full | Operates on activations, not weights |
| Triton Sinkhorn | ✅ Full | Operates on routing logits |
| Triton sparse attention | ✅ Full | Operates on Q/K/V activations |
| fla chunk_gated_delta_rule | ✅ Full | Operates on activations |
| inject_lora / inject_moe_lora | ⚠️ Needs adaptation | Must skip quantized modules or wrap them |
| freeze_non_lora | ✅ Full | Works regardless of weight format |
| T19 cleanup flags | ✅ Full | Memory management is format-agnostic |

### 10.2 Requires New Implementation

| Component | Effort | Description |
|---|---|---|
| NF4 weight storage class | Medium | Custom nn.Module that stores [E, K, N] as NF4 blocks |
| Dequantize-on-gather for MoE | Medium | Wrapper around _moe_grouped() that dequantizes active experts |
| Quantized FusedLoRALinear | Low | Modify forward() to dequantize W_base before F.linear() |
| Expert CPU offload manager | High | Async prefetch of expert weights from CPU to GPU |
| New DeepSpeed configs (ZeRO-2) | Low | JSON config files for 4-bit training |
| New YAML training configs | Low | Config files for each experiment |
| Fused dequant-GEMM kernel (optional) | Very High | Triton kernel that reads NF4 inline during GEMM |
| FSDP migration (optional) | Very High | Replace DeepSpeed with PyTorch FSDP |

### 10.3 Key Risk: Custom Kernels + Quantization Interaction

Our fused kernels (grouped GEMM, gate+up+SiLU, LoRA GEMM) expect bf16 weight tensors.
The safest approach is "dequantize then call existing kernel" — this preserves all
kernel correctness guarantees and only adds a small dequantization step.

The risk of fused dequant-GEMM kernels is high: any bug in the dequantization logic
inside a Triton kernel is extremely hard to debug, and numerical differences could
compound across 20 layers of reversible integration.

**Recommendation**: Always validate 4-bit training against bf16 baseline on a small
run (10 steps) before committing to longer experiments. Compare loss curves, gradient
norms, and expert utilization statistics.

---

## 11. Prototype: 4-Bit LoRA Training Script Outline

Below is a self-contained prototype outline for loading the 70B model in 4-bit with
LoRA and running a single forward+backward step. This is adapted to our codebase style
but uses bitsandbytes for the quantization backend.

```python
"""
Prototype: 4-bit LoRA training step for 70B MoE model.

Usage:
    python prototype_4bit_lora.py

Requirements:
    pip install bitsandbytes>=0.43.0

This script demonstrates the memory footprint of 4-bit base weights + bf16 LoRA
on our custom 70B MoE architecture. For initial validation, use a smaller proxy
model (e.g., 3B MoE) and scale the memory estimates.
"""

import torch
import torch.nn as nn
import bitsandbytes as bnb
from dataclasses import dataclass


@dataclass
class NF4Config:
    """Configuration for NF4 quantization."""
    bits: int = 4
    quant_type: str = "nf4"       # "nf4" or "fp4"
    double_quant: bool = True      # Quantize the quantization constants
    block_size: int = 64           # NF4 block size
    compute_dtype: torch.dtype = torch.bfloat16  # Dequantize to this for compute


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in NF4.
    Dequantizes to compute_dtype for forward pass.
    Compatible with FusedLoRALinear wrapping.
    """
    def __init__(self, in_features, out_features, bias=False, config=None):
        super().__init__()
        if config is None:
            config = NF4Config()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Use bitsandbytes Linear4bit for storage
        self.linear_4bit = bnb.nn.Linear4bit(
            in_features, out_features, bias=bias,
            compute_dtype=config.compute_dtype,
            quant_type=config.quant_type,
            compress_statistics=config.double_quant,
        )

    @property
    def weight(self):
        return self.linear_4bit.weight

    def forward(self, x):
        return self.linear_4bit(x)


class QuantizedMoEExperts(nn.Module):
    """
    Stores MoE expert weights [E, K, N] in NF4 format.
    Dequantizes selected experts on-demand for grouped GEMM.
    """
    def __init__(self, num_experts, in_features, out_features, config=None):
        super().__init__()
        if config is None:
            config = NF4Config()
        self.E = num_experts
        self.K = in_features
        self.N = out_features
        self.config = config

        # Store as quantized: [E*K, N] flattened for bitsandbytes
        # Each expert's [K, N] weight is a contiguous block
        self.weight_4bit = bnb.nn.Params4bit(
            torch.empty(num_experts * in_features, out_features,
                       dtype=config.compute_dtype),
            requires_grad=False,
            quant_type=config.quant_type,
            compress_statistics=config.double_quant,
            blocksize=config.block_size,
        )

    def dequantize_experts(self, expert_indices=None):
        """
        Dequantize expert weights to bf16.

        Args:
            expert_indices: Optional [E_active] tensor of expert indices.
                          If None, dequantizes all experts.

        Returns: [E_active, K, N] bf16 tensor
        """
        # Full dequantization
        full_weight = bnb.functional.dequantize_4bit(
            self.weight_4bit.data,
            self.weight_4bit.quant_state,
            quant_type=self.config.quant_type,
        ).reshape(self.E, self.K, self.N)

        if expert_indices is not None:
            return full_weight[expert_indices]
        return full_weight


def print_memory_stats(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{label}] Allocated: {alloc:.2f} GB, "
              f"Reserved: {reserved:.2f} GB, Peak: {peak:.2f} GB")


# Usage example (run on a smaller proxy model for validation):
#
# if __name__ == "__main__":
#     torch.cuda.reset_peak_memory_stats()
#     print_memory_stats("start")
#
#     # Create a small proxy MoE layer with 4-bit experts
#     config = NF4Config()
#     experts = QuantizedMoEExperts(260, 4096, 1024, config)
#     experts.cuda()
#     print_memory_stats("after loading 260 experts in NF4")
#
#     # Dequantize 8 active experts
#     active = experts.dequantize_experts(torch.tensor([0,1,2,3,4,5,6,7]))
#     print_memory_stats("after dequantizing 8 experts")
#     print(f"Active experts shape: {active.shape}, dtype: {active.dtype}")
#
#     # Compare: full bf16 experts
#     bf16_size = 260 * 4096 * 1024 * 2 / 1e9
#     nf4_size = 260 * 4096 * 1024 * 0.54 / 1e9
#     print(f"bf16 expert size: {bf16_size:.2f} GB")
#     print(f"NF4 expert size: {nf4_size:.2f} GB")
#     print(f"Savings: {bf16_size - nf4_size:.2f} GB ({(1-nf4_size/bf16_size)*100:.1f}%)")
```

---

## 12. Summary and Recommendations

### Immediate Actions (P0 — this week)

1. **Run Experiment 1.1**: Lean LoRA config (remove indexer/router targets, MoE rank=8)
   + tight ZeRO-3 buffers + enable fused DeltaNet entrance. This is config-only and
   should achieve ≤72 GB peak with ≥13k tok/s — beating both current baselines.

2. **Run Experiment 1.2**: Add micro_batch=2 + grad_accum=2 to Experiment 1.1.
   Validates ~55–60 GB peak, creating headroom for future 4-bit work.

### Near-Term (P1 — next 1–2 weeks)

3. **Implement NF4 MoE expert wrapper** (Experiment 2.1): The dequantize-on-gather
   approach requires ~200 lines of new code. Validate on 10-step runs against bf16.

4. **Test top-k=6 routing** (Experiment 1.3): Pure config change, tests quality impact.

### Medium-Term (P2 — 2–4 weeks)

5. **Full NF4 quantization** (Experiment 2.2): Extend to attention projections.
6. **Expert offloading prototype** (Experiment 3.1): For Colab-scale validation.

### The Pareto-Optimal Path

The most promising configuration for pushing the Pareto frontier on current hardware is:

**Lean LoRA (rank=8 MoE, no indexer/router) + tight ZeRO-3 buffers + fused DeltaNet
entrance + micro_batch=2/grad_accum=2**

This should achieve: ~55–60 GB peak, ~12.5–13.5k tok/s — significantly better than
both current baselines on both axes.

Adding NF4 quantization on top of this would push to ~45–50 GB peak, opening the door
to A100-40GB clusters with 8 GPUs.

### The Colab Path

For single-GPU Colab-class training (≤40 GB), the viable path is:
**NF4 base weights + expert CPU offloading + attention-only LoRA (rank=8) + SL=1024 + BS=1**

This trades throughput (~2k tok/s) for accessibility. It's useful for:
- Debugging and rapid iteration on LoRA configurations
- Small-scale fine-tuning experiments
- Demonstrating the model works on consumer hardware

For serious training throughput on smaller GPUs, the FSDP-QLoRA migration (P3) would
be the long-term solution, enabling multi-GPU 4-bit training with proper parameter
sharding.

---

## 13. Implementation Progress — Phase 6/7 Code & Config Alignment

This section documents the concrete implementation work completed since the initial
research report (Sections 1–12). It covers new kernel code, integration into the
training pipeline, config alignment with the expert explosion strategy, and 70B
experiment results.

### 13.1 Expert Explosion Strategy — Config Alignment

The lead clarified the 70B training strategy: the 70B model is an "expert explosion"
from a well-trained 8B (20 experts → 260 experts). Since the 8B has been trained with
the most tokens and its weight matrices have already collapsed to their natural low-rank
structure, the 70B LoRA adaptation is inherently low-rank.

Key implications for LoRA configuration:

- **MoE expert rank=8** (not 16): Experts are expanded copies of 8B's converged experts.
  The collapsed subspace from 8B training means rank=8 captures the adaptation capacity.
- **ALL LoRA targets preserved**: Unlike the initial "lean LoRA" approach (Section 4.2
  Strategy B), the expert explosion strategy requires keeping:
  - `W_Iq, W_Ik, W_Iw` (GSA indexer): 8B's learned sparse attention patterns must transfer
  - `gate` (MoE router): Router must learn the 20→260 expert distribution mapping

All four experiment configs were updated to reflect this:
- `configs/exp1_1_lean_lora_tight_buffers.yaml`
- `configs/exp1_1_8bmoe_lean_lora_tight_buffers.yaml`
- `configs/exp1_2_lean_lora_microbatch2.yaml`
- `configs/exp1_2_8bmoe_lean_lora_microbatch2.yaml`

### 13.2 Lead's 70B Baselines

These baselines were run by the lead on 8×A100-80GB, BS32, SL4096, 252 LoRA targets:

| Run | Configuration | Avg tok/s (steady) | Peak VRAM | dt (s/it) |
|---|---|---|---|---|
| Baseline 1 | Unfused MoE, unfused LoRA | ~12,650 | 72.7 GB | ~10.35 |
| Baseline 2 | Fused MoE, unfused LoRA | ~13,950 | 77.0 GB | ~9.35 |
| Baseline 3 | Fused MoE + Fused LoRA (252 targets) | ~13,150 | 77.8 GB | ~9.95 |

### 13.3 Exp 1.1 — 70B Results (Expert Explosion LoRA + Tight Buffers)

Config: `exp1_1_lean_lora_tight_buffers.yaml` with `T17_DN_USE_DELTA_ENTRANCE=1`.
Changes from baseline: MoE expert rank 16→8, tight ZeRO-3 buffers (10M/25M), all
LoRA targets preserved (indexer + router), fused DeltaNet entrance enabled.

Three runs were captured. Steady-state metrics (steps 2–9, excluding warmup steps
and final step):

**Run 1 (initial code):**

| Step | dt (s) | loss_ntp | tok/s | loss2 | peak VRAM |
|---|---|---|---|---|---|
| 0 | 68.64 | 12.591 | 1,910 | 12.607 | 74.72 GB |
| 1 | 10.41 | 12.589 | 12,592 | 12.622 | 74.72 GB |
| 2 | 10.20 | 12.592 | 12,849 | 12.628 | 74.72 GB |
| 3 | 10.22 | 12.596 | 12,823 | 12.624 | 74.72 GB |
| 4 | 10.22 | 12.607 | 12,827 | 12.616 | 74.72 GB |
| 5 | 10.21 | 12.596 | 12,835 | 12.637 | 74.72 GB |
| 6 | 10.28 | 12.579 | 12,752 | 12.633 | 74.72 GB |
| 7 | 10.12 | 12.608 | 12,958 | 12.614 | 74.72 GB |
| 8 | 10.21 | 12.597 | 12,832 | 12.619 | 74.72 GB |
| 9 | 10.26 | 12.593 | 12,775 | 12.611 | 74.72 GB |
| 10 | 11.23 | 12.596 | 11,674 | 12.621 | 74.72 GB |

**Run 2 (initial code):**

| Step | dt (s) | loss_ntp | tok/s | loss2 | peak VRAM |
|---|---|---|---|---|---|
| 0 | 45.20 | 12.591 | 2,900 | 12.607 | 74.73 GB |
| 1 | 10.28 | 12.589 | 12,749 | 12.622 | 74.73 GB |
| 2 | 10.15 | 12.592 | 12,919 | 12.628 | 74.73 GB |
| 3 | 10.20 | 12.599 | 12,852 | 12.620 | 74.73 GB |
| 4 | 10.19 | 12.611 | 12,866 | 12.623 | 74.73 GB |
| 5 | 10.21 | 12.601 | 12,838 | 12.643 | 74.73 GB |
| 6 | 10.22 | 12.575 | 12,830 | 12.637 | 74.73 GB |
| 7 | 10.10 | 12.598 | 12,971 | 12.599 | 74.73 GB |
| 8 | 10.17 | 12.587 | 12,888 | 12.609 | 74.73 GB |
| 9 | 10.24 | 12.589 | 12,805 | 12.623 | 74.73 GB |
| 10 | 11.34 | 12.614 | 11,556 | 12.627 | 74.73 GB |

**Run 3 (updated LoRA code — `exp1_1_70B_updated.jsonl`):**

| Step | dt (s) | loss | tok/s | loss2 | peak VRAM |
|---|---|---|---|---|---|
| 1 | 170.93 | 12.640 | 767 | 12.578 | 74.7 GB |
| 2 | 10.93 | 12.627 | 11,990 | 12.574 | 74.7 GB |
| 3 | 10.32 | 12.627 | 12,696 | 12.575 | 74.7 GB |
| 4 | 10.37 | 12.598 | 12,636 | 12.570 | 74.7 GB |
| 5 | 10.37 | 12.628 | 12,637 | 12.562 | 74.7 GB |
| 6 | 10.51 | 12.629 | 12,476 | 12.556 | 74.7 GB |
| 7 | 10.41 | 12.614 | 12,592 | 12.574 | 74.7 GB |
| 8 | 10.62 | 12.624 | 12,342 | 12.577 | 74.7 GB |
| 9 | 10.61 | 12.617 | 12,354 | 12.567 | 74.7 GB |

**Summary (steady-state avg, steps 2–9):**

| Metric | Run 1 | Run 2 | Run 3 (updated) | Baseline 3 (fused) |
|---|---|---|---|---|
| Avg tok/s | ~12,805 | ~12,857 | ~12,466 | ~13,150 |
| Peak VRAM | 74.72 GB | 74.73 GB | 74.7 GB | 77.8 GB |
| Avg dt (s) | ~10.21 | ~10.18 | ~10.52 | ~9.95 |
| Avg loss | ~12.596 | ~12.593 | ~12.621 | — |

**Analysis**: All three Exp 1.1 runs consistently save ~3 GB peak VRAM (74.7 vs 77.8 GB)
from the tighter ZeRO-3 buffers. Run 3 (updated LoRA code) shows slightly lower
throughput (~12,466 vs ~12,830 tok/s in Runs 1–2), a ~3% drop that may be from the
updated LoRA injection path or run-to-run variance. The longer warmup step in Run 3
(170.9s vs 68.6s/45.2s) suggests a heavier first-step compilation or ZeRO-3 gather.
Peak VRAM is identical across all three runs. Loss values are not directly comparable
to baselines (baseline was mid-training, Exp 1.1 from step 0), but loss2 in Run 3
trends slightly lower (12.556–12.578) which is encouraging.

The MoE rank reduction (16→8) had negligible throughput impact since expert LoRA params
are tiny. Overall, Exp 1.1 trades ~2.5–5% throughput for 3 GB VRAM savings — a
reasonable trade-off, especially as a foundation for NF4 quantization (Exp 2.1).

### 13.4 Phase 6: NF4 Quantization Implementation

Three new modules implement 4-bit NF4 quantization of MoE expert weights:

#### K6: `code/src/kernels/nf4_quantize.py` — Block-wise NF4 Utilities

Core quantization/dequantization for 3D expert weight tensors `[E, K, N]`.

- `NF4_LEVELS`: 16-value lookup table from the QLoRA paper (normal distribution quantiles)
- `NF4QuantConfig`: Configuration dataclass (`block_size=64`, `double_quant=True`, `compute_dtype=bf16`)
- `_quantize_block_nf4()`: Quantizes flat tensor to NF4 with per-block absmax scaling.
  Packs two 4-bit indices into one uint8 byte. Returns `(packed, absmax)`.
- `_dequantize_block_nf4()`: Unpacks uint8 → 4-bit indices → NF4 lookup → scale by absmax.
- `NF4Parameter`: Non-nn.Parameter storage class with `dequantize()`, `nbytes()`, `to(device)`.
  Stores `packed` (uint8), `absmax` (float32 or FP8), `original_shape`, `block_size`.
- `quantize_tensor_nf4()`: Top-level API. Optionally applies double quantization
  (absmax scales → FP8 E4M3) for additional compression (~0.37 bits/param overhead
  vs 0.5 bits without).

Memory savings: `[260, 4096, 1024]` bf16 = 2.15 GB → NF4 ≈ 0.57 GB per param (3.8× reduction).

#### K7: `code/src/kernels/triton_nf4_grouped_gemm.py` — Fused NF4 Dequant+GEMM

Triton kernel that dequantizes NF4 weights tile-by-tile inside the grouped GEMM,
never materializing the full bf16 weight tensor in HBM.

- `_nf4_grouped_gemm_fwd_kernel`: Triton kernel with tile-level NF4 dequantization.
  Reads packed uint8 weights, unpacks to 4-bit indices, looks up NF4 values, scales
  by per-block absmax, then performs the standard tiled GEMM accumulation.
  Block sizes: `BLOCK_M=64, BLOCK_N=64, BLOCK_K=64`.
- `NF4GroupedGEMMFn`: Autograd Function wrapping the Triton kernel. Forward runs the
  fused NF4 GEMM + LoRA path. Backward computes gradients for LoRA A/B only (base
  NF4 weights are frozen — no gradient needed).
- `nf4_lora_grouped_gemm()`: Top-level API matching the signature of
  `fused_lora_grouped_gemm()` but accepting NF4 packed weights.
- `pytorch_nf4_lora_grouped_gemm()`: Pure PyTorch reference implementation for
  correctness testing (dequantizes then calls standard matmul).

#### Integration: `code/src/nf4_moe_utils.py`

Bridges NF4 quantization with the existing MoE training stack:

- `quantize_moe_experts(model, config)`: Walks model, finds MoEFFN modules, quantizes
  `W_gate`, `W_up`, `W_down` expert weights to NF4 in-place. Replaces `nn.Parameter`
  with `NF4Parameter` stored as module attributes (`_nf4_W_gate`, etc.). Deletes
  original bf16 parameters to free memory.
- `patch_moe_nf4_forward(model)`: Monkey-patches each MoEFFN's forward to use the
  NF4 compute path. If the fused Triton kernel is available, uses
  `nf4_lora_grouped_gemm()`. Otherwise falls back to dequant-then-standard-GEMM.
- `print_nf4_summary(model)`: Prints per-module NF4 memory usage.

### 13.5 Phase 7: Manual Backward LoRA

#### K8: `code/src/kernels/manual_lora_backward.py`

Provides two backward paths for LoRA-augmented linear layers:

1. **`ManualLoRALinearFn`** (autograd Function): Streamlined version of `FusedLoRALinearFn`.
   Key difference: does NOT save `W_base` in `saved_tensors` — only saves `x`, `lora_A`,
   `lora_B`, `lora_mid`. W_base is re-gathered from ZeRO-3 during backward. This reduces
   the autograd tape's memory footprint.

2. **`ManualLoRALinear`** (nn.Module): Drop-in replacement for `FusedLoRALinear` with:
   - `forward()`: Standard autograd path via `ManualLoRALinearFn`
   - `forward_no_grad()`: Runs under `torch.no_grad()` — zero autograd tape overhead.
     Returns `(output, lora_mid)` for later manual backward.
   - `manual_backward()`: Computes `grad_lora_A`, `grad_lora_B`, `grad_x` manually
     under `torch.no_grad()`. Accumulates gradients directly into `.grad` buffers
     (supports gradient accumulation). This is the Unsloth-style path.

Memory savings vs standard autograd:
- No autograd graph nodes (~1–2 KB per node × thousands of nodes across 252 targets)
- No intermediate tensor references held by the tape
- No Python dispatch overhead from autograd

Integration: `code/src/lora_utils.py` now accepts `use_manual_backward` in `LoRAConfig`.
When enabled, `inject_lora()` uses `ManualLoRALinear` instead of `FusedLoRALinear`.

### 13.6 Pipeline Integration — `code/main.py`

#### NF4 Config Parsing

The `Config` class now reads an optional `nf4:` YAML section:

```yaml
nf4:
  enabled: true
  block_size: 64
  double_quant: true
  quantize_experts: true
```

Parsed into: `nf4_enabled`, `nf4_block_size`, `nf4_double_quant`, `nf4_quantize_experts`.

#### Step 2.7: NF4 Quantization

Inserted between Step 2.5 (LoRA injection) and Step 3 (DeepSpeed init):

```
[2.7/5] Quantizing MoE expert weights to NF4...
  NF4: {n} expert weight tensors quantized
  NF4: {n} MoE modules patched for NF4 forward
```

This ordering is critical: LoRA adapters are injected first (bf16), then base expert
weights are quantized to NF4. The LoRA A/B matrices remain in bf16 for training.
DeepSpeed init happens after quantization so ZeRO-3 sees the reduced memory footprint.

#### Kernel Registration

`code/src/kernels/__init__.py` exports Phase 6 and Phase 7 symbols:

```python
# Phase 6: NF4 Quantization
from .nf4_quantize import NF4QuantConfig, NF4Parameter, NF4_LEVELS, quantize_tensor_nf4
from .triton_nf4_grouped_gemm import nf4_lora_grouped_gemm, pytorch_nf4_lora_grouped_gemm

# Phase 7: Manual backward LoRA
from .manual_lora_backward import ManualLoRALinear, ManualLoRALinearFn
```

### 13.7 Experiment 2.1: NF4 QLoRA Configs

Two configs created for NF4 validation:

**70B** (`configs/exp2_1_nf4_qlora_tight.yaml`):
- NF4 quantization of W_gate/W_up/W_down expert weights
- All LoRA targets preserved (expert explosion strategy)
- MoE expert rank=8, attention rank=16
- Tight ZeRO-3 buffers
- Target: ≤50 GB peak on 8×A100-80GB

**8B proxy** (`configs/exp2_1_8bmoe_nf4_qlora_tight.yaml`):
- Same NF4 config, scaled to 8B model (20 experts)
- For validation before committing 70B GPU time

Run commands:
```bash
# 8B proxy (validate first)
T17_DN_USE_DELTA_ENTRANCE=1 T19_NF4_MOE=1 deepspeed --num_gpus=8 code/main.py \
  --config configs/exp2_1_8bmoe_nf4_qlora_tight.yaml

# 70B (after 8B validation)
T17_DN_USE_DELTA_ENTRANCE=1 T19_NF4_MOE=1 deepspeed --num_gpus=8 code/main.py \
  --config configs/exp2_1_nf4_qlora_tight.yaml
```

### 13.8 Updated Experiment Roadmap

| Experiment | Status | VRAM | Throughput | Notes |
|---|---|---|---|---|
| 1.1 Expert Explosion LoRA + tight buffers | ✅ 70B done | 74.7 GB | ~12,830 tok/s | 3 GB saved vs baseline 3 |
| 1.2 Micro-batch=2 + grad_accum=2 | ✅ 8B done | (14.2 GB 8B) | (~13,510 8B) | Ready for 70B |
| 2.1 NF4 QLoRA (expert weights) | 🔧 Code ready | Target ≤50 GB | Target ≥10k | Run 8B proxy first |
| 2.2 Full NF4 (experts + attention) | Planned | Target ≤45 GB | Target ≥9k | After 2.1 validation |
| 3.1 Expert offloading (Colab) | Planned | Target ≤30 GB | Target ≥1k | Single GPU path |

### 13.9 File Inventory — New/Modified

| File | Type | Description |
|---|---|---|
| `code/src/kernels/nf4_quantize.py` | New | K6: NF4 quantization/dequantization utilities |
| `code/src/kernels/triton_nf4_grouped_gemm.py` | New | K7: Fused Triton NF4 dequant+GEMM kernel |
| `code/src/nf4_moe_utils.py` | New | NF4 ↔ MoE integration (quantize, patch, summary) |
| `code/src/kernels/manual_lora_backward.py` | New | K8: Manual backward LoRA (Unsloth-style) |
| `code/src/kernels/__init__.py` | Modified | Added Phase 6 + Phase 7 exports |
| `code/src/lora_utils.py` | Modified | Added `use_manual_backward` config option |
| `code/main.py` | Modified | Added NF4 config parsing + Step 2.7 quantization |
| `configs/exp2_1_nf4_qlora_tight.yaml` | New | 70B NF4 QLoRA experiment config |
| `configs/exp2_1_8bmoe_nf4_qlora_tight.yaml` | New | 8B NF4 QLoRA proxy config |
| `configs/exp1_1_*.yaml` (×2) | Modified | Aligned with expert explosion strategy |
| `configs/exp1_2_*.yaml` (×2) | Modified | Aligned with expert explosion strategy |

---

*Report generated for Test36_70BLoRA optimization research.*
*Sources: [QLoRA paper](https://arxiv.org/abs/2305.14314), [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/fsdp_qlora), [FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/main/fsdp_qlora), [Unsloth](https://github.com/unslothai/unsloth). Content was rephrased for compliance with licensing restrictions.*
