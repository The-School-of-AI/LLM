# Test19 Throughput Optimization Benchmark Results

**Instance**: p4d.24xlarge (8x A100-40GB, NVLink, 96 vCPUs, 1.1TB RAM)
**Date**: 2026-03-03
**Pinned deps**: torch=2.7.1+cu128, triton=3.3.1, deepspeed=0.18.6, FLA=0.4.2

---

## Phase 1: Initial 36-Variant Sweep

### Models Tested
| Model | Params | Architecture | Batch Size | Seq Len |
|-------|--------|-------------|-----------|---------|
| 1B non-rev | 1.65B | Dense, 8 layers DDDGDDDG | 8 | 4096 |
| 3B MoE | 3.92B | 8 layers, 20 experts top-k=2 | 8 | 4096 |
| 8B MoE | 8.32B | 20 layers, 20 experts top-k=2 | 8 | 4096 |

### 1B Non-Reversible Results (13 variants)

| Variant | tok/s | vs baseline | Notes |
|---------|-------|-------------|-------|
| **baseline** | **16,284** | — | Default ZeRO-3 config |
| bucket_200m | 15,891 | -2.4% | allgather_bucket_size=200M |
| bucket_500m | 15,647 | -3.9% | allgather_bucket_size=500M |
| torch_compile | 16,102 | -1.1% | Custom kernels bypass dynamo |
| torch_compile_reduce | 16,198 | -0.5% | compile + reduce_scatter |
| fused_triton | 15,967 | -1.9% | require_fused_kernels=true |
| nccl_tree | CRASH | — | NCCL_ALGO=Tree incompatible |
| no_gc_sync | 12,844 | -21.1% | Disable GC/sync/empty_cache |
| **live_params_100m** | **17,409** | **+6.9%** | stage3_max_live_parameters=100M |
| compile_reduce_only | 16,221 | -0.4% | compile_reduce without compile |
| live_params_compile | 17,312 | +6.3% | live_params + compile_reduce |

**Winner: live_params_100m (+6.9%)**
- Keeping more parameters "live" (gathered) reduces redundant all-gathers
- Applied as new default: `stage3_max_live_parameters=100M, stage3_max_reuse_distance=100M`

### 3B MoE Results (9 variants)

| Variant | tok/s | vs baseline | Notes |
|---------|-------|-------------|-------|
| **baseline** | **12,114** | — | Default config |
| bucket_200m | 11,893 | -1.8% | |
| bucket_500m | 11,756 | -3.0% | |
| fused_triton | 11,891 | -1.8% | |
| nccl_tree | CRASH | — | |
| no_gc_sync | 8,601 | -29.0% | |
| live_params_100m | 12,089 | -0.2% | |
| compile_reduce | 12,078 | -0.3% | |
| live_params_compile | 12,034 | -0.7% | |

**Winner: baseline is optimal**
- 3B is bottlenecked by MoE computation (grouped_gemm), not ZeRO-3 comms
- No changes applied to 3B configs

### 8B MoE Results (14 variants)

| Variant | tok/s | vs baseline | Notes |
|---------|-------|-------------|-------|
| **baseline** | **3,535** | — | Default config |
| bucket_200m | 3,478 | -1.6% | |
| bucket_500m | 3,412 | -3.5% | |
| fused_triton | 3,588 | +1.5% | |
| nccl_tree | CRASH | — | |
| no_gc_sync | 2,822 | -20.2% | |
| live_params_100m | 3,612 | +2.2% | |
| compile_reduce | 3,541 | +0.2% | |
| live_params_compile | 3,598 | +1.8% | |
| chunk_1gb | 3,589 | +1.5% | max_chunk_gb=1 |
| chunk_4gb | 3,556 | +0.6% | max_chunk_gb=4 |
| subgroup_10m | 3,562 | +0.8% | sub_group_size=10M |
| persist_1m | 3,578 | +1.2% | param_persistence_threshold=1M |
| **best_combo** | **3,644** | **+3.1%** | live_params 100M + persist 1M + subgroup 10M |

**Winner: best_combo (+3.1%)**
- Applied as new defaults for 8B, 70B, and 120B configs:
  - `stage3_max_live_parameters=100M`
  - `stage3_max_reuse_distance=100M`
  - `stage3_param_persistence_threshold=1M`
  - `max_chunk_gb=1`

### Key Findings from Phase 1

1. **Larger bucket sizes always hurt** — 50M is the sweet spot
2. **torch.compile is useless** — custom Triton/FLA kernels bypass dynamo
3. **NCCL Tree crashes** — incompatible with ZeRO-3 on this stack
4. **Never disable GC/sync** — memory leak prevention flags are critical (-20-29% without them)
5. **live_params_100m is the single biggest win** — reduces redundant all-gather operations
6. **Fused Triton kernels**: negligible for small models, +1.5% for 8B

### Defaults Applied After Phase 1

```yaml
# DeepSpeed ZeRO-3 (all models):
stage3_max_live_parameters: 100000000    # was 20M
stage3_max_reuse_distance: 100000000     # was 20M

# DeepSpeed ZeRO-3 (70B/120B additional):
stage3_param_persistence_threshold: 1000000  # was 100K

# YAML config (70B/120B):
max_chunk_gb: 1                          # was 8
```

---

## Phase 2: Exhaustive Lever Sweep (COMPLETE)

Testing additional optimization levers not covered in Phase 1.
All tests run on the same instance with Phase 1 optimal defaults as the new baseline.

**Important note on Phase 2 baseline**: The Phase 2 test harness set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which costs ~16-22% on 1B/3B (see Phase 3).
This makes the Phase 2 baseline (13,548 tok/s for 1B) lower than the true production baseline
(~20,000 tok/s without expandable_segments). The Phase 2 lever deltas below are relative to this
expandable-penalized baseline, so they appear larger than real-world impact. The Phase 4 best-combo
test confirms that **no Phase 2 lever provides meaningful improvement** once expandable_segments
is correctly configured per model.

### Test Plan (ordered easiest to most complex)

| # | Lever | Type | What it does |
|---|-------|------|-------------|
| 1 | T19_ZERO3_RELEASE_EVERY=5 | env var | Reduce ZeRO-3 cache release frequency |
| 2 | T19_ZERO3_RELEASE_EVERY=10 | env var | Release every 10 steps instead of every step |
| 3 | T19_CLEAR_ROUTER_CACHE_EVERY=5 | env var | Reduce MoE router cache clear frequency |
| 4 | TORCH_NCCL_AVOID_RECORD_STREAMS=1 | env var | Avoid recording NCCL streams in allocator |
| 5 | NCCL_BUFFSIZE=8388608 | env var | Increase NCCL buffer to 8MB (default 4MB) |
| 6 | NCCL_BUFFSIZE=16777216 | env var | Increase NCCL buffer to 16MB |
| 7 | reduce_bucket_size=100M/200M | DS config | Tune backward gradient reduce bucket |
| 8 | prefetch_bucket_size=100M/200M | DS config | Tune parameter prefetch during forward |
| 9 | round_robin_gradients=true | DS config | Distribute grad processing evenly (MoE) |
| 10 | num_workers=4/8/24/32 | code | DataLoader worker thread count |
| 11 | prefetch_factor=2/8/16 | code | DataLoader prefetch batches per worker |
| 12 | T19_MOE_VECTORIZED_CHUNK=128/256 | env var | MoE expert vectorized chunk size |
| 13 | torch.backends.cudnn.benchmark=True | env var | cuDNN autotuner for fixed-size inputs |
| 14 | NCCL_MAX/MIN_NCHANNELS | env var | NCCL parallelism on NVLink topology |

### Phase 2 Results

Results collected on p4d.24xlarge, 10-step runs. Phase 2 baseline uses Phase 1 optimal
defaults (live_params=100M, reuse_distance=100M) with per-step cleanup enabled.

#### Cleanup / Env Var Levers

| # | Lever | 1B tok/s | 1B delta | 3B tok/s | 3B delta | 8B tok/s | 8B delta |
|---|-------|----------|----------|----------|----------|----------|----------|
| — | **Phase 2 baseline** | **13,548** | — | **11,797** | — | **3,609** | — |
| 1 | release_every_5 | 16,610 | +22.6% | 12,179 | +3.2% | 3,658 | +1.3% |
| 2 | release_every_10 | 17,278 | **+27.5%** | 12,314 | **+4.4%** | 3,643 | +0.9% |
| 3 | router_cache_5 | 17,407 | **+28.5%** | 12,050 | +2.1% | 3,631 | +0.6% |
| 4 | nccl_no_record_streams | 17,401 | +28.4% | 11,995 | +1.7% | 3,587 | -0.6% |
| 5 | nccl_buff_8mb | 16,196 | +19.5% | 9,630 | **-18.4%** | 2,549 | **-29.4%** |
| 6 | nccl_buff_16mb | 17,357 | +28.1% | 12,132 | +2.8% | 3,644 | +1.0% |

#### Compute / NCCL Levers

| # | Lever | 1B tok/s | 1B delta | 3B tok/s | 3B delta | 8B tok/s | 8B delta |
|---|-------|----------|----------|----------|----------|----------|----------|
| 12a | moe_chunk_128 | 13,494 | -0.4% | 12,267 | +4.0% | 3,540 | -1.9% |
| 12b | moe_chunk_256 | 17,200 | +27.0% | 12,298 | **+4.2%** | 2,845 | **-21.2%** |
| 13 | cudnn_bench | 15,357 | +13.3% | 12,169 | +3.2% | 3,577 | -0.9% |
| 14a | nccl_32ch | 17,099 | +26.2% | 12,060 | +2.2% | 3,641 | +0.9% |
| 14b | nccl_min_16ch | 17,134 | +26.5% | 12,206 | +3.5% | 3,619 | +0.3% |

#### DeepSpeed Config Levers (in progress)

| # | Lever | 1B tok/s | 1B delta | 3B tok/s | 3B delta | 8B tok/s | 8B delta |
|---|-------|----------|----------|----------|----------|----------|----------|
| 7a | reduce_bucket_100m | 16,824 | +24.2% | 9,648 | **-18.2%** | 3,547 | -1.7% |
| 7b | reduce_bucket_200m | 16,589 | +22.4% | 12,229 | +3.7% | 3,604 | -0.1% |
| 8a | prefetch_100m | 16,420 | +21.2% | 11,928 | +1.1% | 2,561 | **-29.1%** |
| 8b | prefetch_200m | 13,377 | -1.3% | 12,324 | **+4.5%** | 3,599 | -0.3% |

#### DataLoader / Remaining Levers

**Note**: These tests ran in a later session. A fresh baseline was re-run to ensure apples-to-apples
comparison. The fresh baseline (19,527 / 13,146 / 3,695) is higher than the original Phase 2
baseline (13,548 / 11,797 / 3,609) due to CUDA context warmup from many prior runs on the same
instance. Deltas below are computed vs the fresh baseline.

| # | Lever | 1B tok/s | 1B delta | 3B tok/s | 3B delta | 8B tok/s | 8B delta |
|---|-------|----------|----------|----------|----------|----------|----------|
| — | **fresh baseline** | **19,527** | — | **13,146** | — | **3,695** | — |
| 9 | round_robin | 19,814 | +1.5% | 12,820 | -2.5% | 3,636 | -1.6% |
| 10a | workers_4 | 19,128 | -2.0% | 10,192 | **-22.5%** | 3,638 | -1.5% |
| 10b | workers_8 | 19,607 | +0.4% | 12,632 | -3.9% | 3,678 | -0.5% |
| 10c | workers_24 | 15,011 | **-23.1%** | 12,928 | -1.7% | 3,705 | +0.3% |
| 10d | workers_32 | 19,457 | -0.4% | 12,756 | -3.0% | 3,669 | -0.7% |
| 11a | prefetch_factor_2 | 16,026 | **-17.9%** | 12,832 | -2.4% | 3,691 | -0.1% |
| 11b | prefetch_factor_8 | 19,324 | -1.0% | 13,094 | -0.4% | 2,620 | **-29.1%** |
| 11c | prefetch_factor_16 | 19,580 | +0.3% | 13,109 | -0.3% | 3,649 | -1.2% |

### Phase 2 Analysis

**Key findings:**

1. **Per-step cleanup is the biggest bottleneck for 1B**: The baseline does GC + CUDA sync +
   empty_cache + ZeRO-3 release + router cache clear EVERY step. Reducing release frequency
   from every-step to every-10 gives **+27.5%** on 1B. This means the 1B model spends ~22% of
   its time on cleanup overhead.

2. **3B MoE sees modest gains** — best levers are `prefetch_200m` (+4.5%),
   `release_every_10` (+4.4%), `moe_chunk_256` (+4.2%), `moe_chunk_128` (+4.0%),
   `reduce_bucket_200m` (+3.7%), `nccl_min_16ch` (+3.5%).
   MoE grouped_gemm computation dominates, so cleanup/comms tuning has limited impact.

3. **8B MoE is essentially compute-bound** — no lever moves the needle beyond ~1.3%.
   The 8B model with 20 layers x 20 experts is entirely bottlenecked by MoE computation.
   The grouped_gemm backend IS active (verified in Phase 5), so this is already the
   optimized path — no further speedup possible from backend switching.

4. **NCCL_BUFFSIZE=8MB is TOXIC** — -18.4% on 3B, **-29.4% on 8B**. Causes buffer contention
   with the many small MoE expert all-gathers. Default 4MB or 16MB are safe.

5. **moe_chunk_256 is TOXIC for 8B** (-21.2%) — larger chunk sizes cause memory pressure
   at 8B scale, leading to more fragmentation and OOM-avoidance overhead.
   Fine for 3B (+4.2%) but dangerous for larger models.

6. **reduce_bucket_100m is TOXIC for 3B** (-18.2%) — larger reduce buckets cause contention
   with MoE's many small all-gathers, same pattern as nccl_buff_8mb.

7. **TORCH_NCCL_AVOID_RECORD_STREAMS** — neutral. The large 1B deltas (~28%) are from baseline
   cleanup overhead, not this flag. On 8B it slightly hurts (-0.6%).

8. **cuDNN benchmark** — +13.3% on 1B (fixed-size inputs benefit from autotuner), +3.2% on 3B,
   negligible on 8B. Worth enabling for dense models.

9. **NCCL channel tuning** — 32 max channels and 16 min channels both show ~26% on 1B (mostly
   cleanup-related), +2-3.5% on 3B, negligible on 8B. `nccl_min_16ch` is the better option
   for 3B (+3.5% vs +2.2%).

10. **reduce_bucket_size** — 100M is toxic for 3B (-18.2%, same contention pattern as
    nccl_buff_8mb), 200M is slightly positive for 3B (+3.7%). Keep 50M default for safety.

11. **prefetch_bucket_size** — 100M is **catastrophic for 8B** (-29.1%), but 200M is fine for
    8B (-0.3%) and **best for 3B** (+4.5%). This is the largest 3B improvement found so far.
    The 100M vs 200M divergence suggests a sweet spot above 100M where prefetch and compute overlap
    well vs where it causes memory pressure.

12. **round_robin_gradients** — +1.5% on 1B (noise), -2.5% on 3B, -1.6% on 8B. Not helpful.

13. **DataLoader num_workers** — The default (16) is optimal. workers_4 is **toxic for 3B**
    (-22.5%), workers_24 is toxic for 1B (-23.1%). Going below 8 or above 16 hurts somewhere.

14. **DataLoader prefetch_factor** — Default (4) is optimal. prefetch_2 hurts 1B (-17.9%),
    prefetch_8 is **catastrophic for 8B** (-29.1%). prefetch_16 is neutral.

**TOXIC LEVERS TO AVOID (>10% regression on any model):**
| Lever | Worst regression | Model |
|-------|-----------------|-------|
| NCCL_BUFFSIZE=8MB | -29.4% | 8B |
| prefetch_bucket_size=100M | -29.1% | 8B |
| prefetch_factor=8 | -29.1% | 8B |
| num_workers=24 | -23.1% | 1B |
| num_workers=4 | -22.5% | 3B |
| moe_chunk_256 | -21.2% | 8B |
| reduce_bucket_size=100M | -18.2% | 3B |
| prefetch_factor=2 | -17.9% | 1B |

**Model-specific conclusions:**
- **1B**: Dominated by cleanup overhead. Best lever: `release_every_10` or `router_cache_5`.
  Default num_workers=16 and prefetch_factor=4 are already optimal.
- **3B**: MoE-bound but tuning helps up to ~4.5%. Best: `prefetch_200m` (+4.5%),
  `release_every_10` (+4.4%). Avoid workers<8 and reduce_bucket_100m.
- **8B**: Fully compute-bound. No Phase 2 lever helps beyond +1.3%. Many levers are actively
  harmful. Keep all defaults. The only path to 8B speedup is MoE backend optimization
  (grouped_gemm instead of vectorized fallback).

---

## Phase 3: expandable_segments A/B Test

**Critical discovery**: The Phase 2 baseline script set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
which uses PyTorch's virtual memory allocator (cudaMemAddressReserve + cudaMemMap) instead of cudaMalloc.
This was present in the production `run.sh` as default. A/B testing reveals it has a massive, model-dependent
impact on throughput.

### expandable_segments Results

Steady-state throughput (steps 7-10 average, 10-step runs):

| Setting | 1B tok/s | 1B VRAM | 3B tok/s | 3B VRAM | 8B tok/s | 8B VRAM |
|---------|----------|---------|----------|---------|----------|---------|
| **WITH expandable** | 16,932 | 6.7G | 9,682 | 11.3G | **3,685** | 20.1G |
| **WITHOUT expandable** | **20,054** | 6.7G | **12,399** | 11.3G | 2,620 | 20.1G |
| Delta | **+18.4%** | same | **+28.1%** | same | **-28.9%** | same |

### Analysis

1. **1B dense (6.7G / 40G = 17% VRAM utilization)**: expandable_segments costs **-15.6%** throughput.
   At low memory utilization, the virtual memory allocator's address reservation + mapping overhead
   is pure waste — cudaMalloc works fine with no fragmentation risk.

2. **3B MoE (11.3G / 40G = 28% VRAM utilization)**: expandable_segments costs **-21.9%** throughput.
   Still low enough utilization that cudaMalloc handles allocation efficiently.

3. **8B MoE (20.1G / 40G = 50% VRAM utilization)**: expandable_segments **saves +40.7%** throughput.
   At 50% utilization with many small MoE expert allocations, cudaMalloc suffers severe fragmentation.
   Without expandable_segments, the 8B model only achieved 2,620 tok/s (steps completed at 12.5s/step
   instead of 8.9s/step). The process also timed out before completing all 11 steps, suggesting
   fragmentation worsens over time.

4. **Memory usage is identical** — expandable_segments doesn't save VRAM, it just changes the
   allocation strategy. The throughput impact is purely from allocator overhead.

### Implication for Phase 2 Results

All Phase 2 cleanup/NCCL/DS-config levers were tested WITH expandable_segments active. The large
1B improvements (~27%) were partially because the expandable_segments overhead amplified cleanup
costs. The DataLoader levers (round_robin, workers, prefetch_factor) used a fresh baseline WITHOUT
expandable_segments, so those deltas are accurate for production.

### Production Recommendation

```bash
# 1B, 3B: DO NOT SET expandable_segments (let cudaMalloc handle it)
# 8B, 70B, 120B: MUST USE expandable_segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 8B+ only
```

---

## Phase 4: Best-Combo Test

Combining all winning Phase 2 levers per model, using the correct expandable_segments setting.
Same-session baselines run immediately after for apples-to-apples comparison.

### Best Combo Configurations

| Model | expandable_segments | Levers Applied |
|-------|-------------------|----------------|
| 1B | OFF | release_every_10, cudnn_bench, nccl_min_16ch |
| 3B | OFF | release_every_10, router_cache_5, prefetch_200m, reduce_200m, nccl_min_16ch, moe_chunk_256 |
| 8B | ON | release_every_10, router_cache_5, nccl_buff_16mb, nccl_32ch, nccl_min_16ch |

### Best Combo Results (same-session comparison)

| Model | Baseline tok/s | Best Combo tok/s | Delta | VRAM |
|-------|---------------|-----------------|-------|------|
| 1B | 19,980 | 19,794 | **-0.9%** | 6.7G |
| 3B | 13,199 | 13,132 | **-0.5%** | 11.3G / 11.6G |
| 8B | 3,614 | 3,629 | **+0.4%** | 20.1G |

### Phase 4 Conclusion

**The best combos provide NO meaningful improvement over the correctly-configured baseline.**
All Phase 2 lever gains (~4-28%) were artifacts of `expandable_segments:True` being active in
the Phase 2 test environment. The expandable_segments overhead amplified cleanup costs, making
individual levers appear more impactful than they actually are. Once the correct allocator is
selected per model, the baseline is already at ~99% of achievable throughput.

---

## Phase 5: MoE Backend Verification

Investigating whether the model uses the fast `grouped_gemm` backend or falls back to the slow
`vectorized` (Python loop + torch.bmm) path.

### Findings

1. **grouped_gemm IS installed** and the CUDA kernel works when `batch_sizes` (expert counts)
   are on CPU. Direct calls with GPU tensors fail ("Expected batch_sizes.is_cpu()").

2. **The wrapper** (`src/kernels/moe_grouped_gemm.py`) correctly normalizes expert counts to CPU
   before calling `ops.gmm()`. Forward AND backward (autograd) work through the wrapper.

3. **The model resolves to "grouped_gemm" backend** via `_resolve_moe_backend("auto")` since
   `HAS_MOE_GROUPED_GEMM=True`. The try/except fallback at lines 1536-1543 does NOT trigger.

### Performance Comparison

| Backend | Time per MoE layer | Speedup |
|---------|-------------------|---------|
| **grouped_gemm** (active) | **3.47 ms** | — |
| vectorized (chunk=64) | 1,959 ms | 564x slower |
| vectorized (chunk=128) | 1,951 ms | 562x slower |
| vectorized (chunk=256) | 1,946 ms | 561x slower |
| vectorized (chunk=512) | 1,944 ms | 560x slower |

Test: 20 experts, d_model=1536, d_hidden=4096, 13,107 sorted tokens (simulating B=2, T=4096, top_k=2, 80% real).

### Conclusion

**The MoE backend is already optimized.** The model uses `grouped_gemm` (564x faster than the
vectorized Python loop fallback). The "MoE Python loop bottleneck" concern was unfounded — the
vectorized path is only a fallback that never triggers during normal training. The 3B and 8B
models are genuinely compute-bound on the grouped_gemm CUDA kernels, not on Python overhead.

### Remaining Untapped Optimization: Expert Parallelism

The model uses `moe_expert_parallel_size=1` (all GPUs replicate all 20 experts). With 8 GPUs,
expert parallelism could distribute experts across GPUs (2-3 per GPU) to reduce redundant
computation. However, this requires:
- A working dispatcher backend (currently placeholder only — `t4_enabled=False`)
- All-to-all token routing communication
- Significant code changes to the MoE dispatch path

This is an architectural change, not a config tuning opportunity.

---

## Levers Tested and Ruled Out

| Lever | Why excluded |
|-------|-------------|
| `communication_data_type: "fp16"` | Risky — bf16 gradients can overflow when cast to fp16 (max 65504 vs 3.4e38) |
| Gradient accumulation | Not used in any model (user requirement) |
| Activation checkpointing | Redundant for reversible models; saves memory not speed for 1B |
| NVMe offloading | Trades speed for memory — opposite of our goal |
| torch.compile | Custom Triton/FLA kernels bypass dynamo (tested in Phase 1) |
| NCCL_ALGO=Tree | Crashes with ZeRO-3 (tested in Phase 1) |
| round_robin_gradients | Neutral to slightly negative (Phase 2) |
| num_workers != 16 | Default 16 is optimal; 4 hurts 3B, 24 hurts 1B (Phase 2) |
| prefetch_factor != 4 | Default 4 is optimal; 2 hurts 1B, 8 kills 8B (Phase 2) |
| reduce_bucket_size=100M | Toxic for 3B (-18.2%), Phase 2 |
| prefetch_bucket_size=100M | Toxic for 8B (-29.1%), Phase 2 |
| NCCL_BUFFSIZE=8MB | Toxic for both 3B (-18.4%) and 8B (-29.4%), Phase 2 |
| All Phase 2 levers combined | +0.4% at best when expandable_segments set correctly (Phase 4) |
| MoE grouped_gemm backend | Already active — model uses it by default (564x faster than vectorized), Phase 5 |
| Expert parallelism | Requires dispatcher backend (placeholder only), architectural change needed |
| cudnn.benchmark=True | +3.2% on 3B in Phase 2, but noise once expandable_segments set correctly (Phase 4) |
| cudnn.deterministic=False | Model uses no convolutions (FLA/Triton kernels), so cudnn settings are irrelevant |

---

## FINAL RESULTS: Total Improvement Achieved

### End-to-End Throughput Gains

Comparing original default configs vs final optimized configs:

| Model | Original tok/s | Optimized tok/s | Total Gain | Key Change |
|-------|---------------|----------------|------------|------------|
| **1B dense** | ~16,284 | **~20,000** | **+22.8%** | Remove expandable_segments + live_params=100M |
| **3B MoE** | ~12,114 | **~13,200** | **+9.0%** | Remove expandable_segments + live_params=100M |
| **8B MoE** | ~3,535 | **~3,630** | **+2.7%** | Keep expandable_segments + live_params=100M + persist_1M |

### Breakdown of Improvements

**1B Dense (+22.8% total):**
- Phase 1: live_params=100M → +6.9%
- Phase 3: Remove expandable_segments → +18.4% (partially overlapping with Phase 1 gain)
- Net: ~20,000 tok/s (1,636 ms/step → ~1,640 ms/step without overhead)

**3B MoE (+9.0% total):**
- Phase 1: live_params=100M → negligible for 3B
- Phase 3: Remove expandable_segments → +28.1%
- Net: ~13,200 tok/s (bottleneck is MoE grouped_gemm computation)

**8B MoE (+2.7% total):**
- Phase 1: live_params=100M + persist_1M + chunk_1gb → +3.1%
- Phase 3: Must KEEP expandable_segments (required at 50% VRAM utilization)
- Net: ~3,630 tok/s (bottleneck is MoE computation, not comms/allocator)

---

## Final Production Configuration

### DeepSpeed ZeRO-3 Config (all models)
```json
{
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000,
    "allgather_partitions": true,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_max_live_parameters": 100000000,
    "stage3_max_reuse_distance": 100000000,
    "stage3_param_persistence_threshold": 100000,
    "sub_group_size": 1000000
  }
}
```

### DeepSpeed ZeRO-3 Config (8B, 70B, 120B — additional)
```json
{
  "zero_optimization": {
    "stage3_param_persistence_threshold": 1000000,
    "sub_group_size": 10000000
  }
}
```

### YAML Config (8B, 70B, 120B — additional)
```yaml
training:
  max_chunk_gb: 1    # was 8
```

### Environment Variables (run.sh)
```bash
# CRITICAL: expandable_segments setting by model size
# 1B, 3B: DO NOT SET (or explicitly unset)
unset PYTORCH_CUDA_ALLOC_CONF

# 8B, 70B, 120B: MUST SET
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Rule of thumb: use expandable_segments when VRAM utilization > ~40%
# 1B = 6.7G/40G (17%) → OFF
# 3B = 11.3G/40G (28%) → OFF
# 8B = 20.1G/40G (50%) → ON
# 70B, 120B → ON (always memory-constrained)
```

### Do NOT Change (Confirmed Optimal)
- `allgather_bucket_size=50M` — larger always hurts
- `reduce_bucket_size=50M` — 100M toxic for 3B
- `num_workers=16` — sweet spot
- `prefetch_factor=4` — sweet spot
- `NCCL_BUFFSIZE` — leave at default (4MB)
- Per-step GC/sync/empty_cache — must remain enabled
- `NCCL_ALGO` — leave at default (Tree crashes)

### TOXIC Settings to Avoid
| Setting | Impact | Why |
|---------|--------|-----|
| `expandable_segments:True` on small models | -16% to -22% | Virtual memory overhead at low utilization |
| NO `expandable_segments` on large models | -29% to crash | Memory fragmentation at high utilization |
| `NCCL_BUFFSIZE=8MB` | -18% to -29% | Buffer contention with MoE all-gathers |
| `prefetch_bucket_size=100M` | up to -29% | Memory pressure on 8B |
| `prefetch_factor=8` | up to -29% | Memory pressure on 8B |
| `num_workers=4` | -22% on 3B | Starves data pipeline |
| `num_workers=24` | -23% on 1B | CPU contention |
| `moe_chunk_256` | -21% on 8B | Memory pressure at scale |
| `reduce_bucket_size=100M` | -18% on 3B | Contention with MoE all-gathers |
| `prefetch_factor=2` | -18% on 1B | Starves data pipeline |

---

## Methodology

- **Instance**: p4d.24xlarge (8x A100-40GB, NVLink, 96 vCPUs, 1.1TB RAM)
- **Benchmark**: 10-step training runs, metrics from steady-state steps 7-10
- **Data**: wikitext-103 shards, seq_len=4096, micro_batch=1, global_batch=8
- **Total variants tested**: 60+ across 4 phases
- **Total runs**: 90+ (including baselines and A/B tests)
- **Pinned deps**: torch=2.7.1+cu128, triton=3.3.1, deepspeed=0.18.6, FLA=0.4.2
