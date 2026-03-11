# Testing Plan: 8B First, Then 70B

Test DeltaNet optimizations and the MoE sync fix on a small model (8B) with profiling, then validate throughput on 70B.

---

## Prerequisites

- **8B config:** There is no `test_8b_moe_lora_*.yaml` in `configs/` by default. Either:
  - Copy the 70B config and set `model_name: 8bmoe` and e.g. `metrics_jsonl_path: ../results/8b_run/metrics.jsonl` (and optionally `profile_steps: [1, 2, 3, 4, 5]` for profiling), or
  - Use **3B** for the “small model” phase: `configs/test17_3b_moe_perf_4096_bs32_10steps.yaml` (set `profile_steps: [1,2,3,4,5]` in the YAML if you want kernel timing).
- **8B config (created):** `configs/test_8b_moe_lora_4096_bs32_10steps.yaml` — metrics and profile under `results/8b_run/`.
- **70B config:** `configs/test_70b_moe_lora_4096_bs32_10steps.yaml`. For throughput testing, set `profile_steps: []` to avoid OOM.
- **Init:** First run will save init; use `FORCE_REWRITE_INIT=0` for later runs to reuse it.

---

## Phase 1: 8B (or 3B) — Profiling + sanity

Goal: confirm no throughput collapse from syncs, and that DeltaNet kernel times improve when opts are enabled.

| Step | What | Config | Env | Record |
|------|------|--------|-----|--------|
| 1.1 | **Baseline** (no DeltaNet opts, with MoE fix) | 8B/3B config | (none) | Throughput (tok/s), `profile_report.txt` → `deltanet.entry.proj`, `deltanet.entry.conv`, `deltanet.entry.norm_rope` ms |
| 1.2 | **Exp1** Fused QKVG only | Same | `T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=0` | Throughput; `deltanet.entry.proj` should drop (~10–15%) |
| 1.3 | **Exp2** Fused entrance only | Same | `T17_DN_USE_FUSED_QKVG=0 T17_DN_USE_DELTA_ENTRANCE=1` | Throughput; profile should show `deltanet.entry.fused` instead of conv+norm_rope |
| 1.4 | **Exp3** Both | Same | `T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=1` | Throughput; combined DeltaNet entry time lower than baseline |
| 1.5 | **Exp4** (optional) FLA recompute | Same | Add `T17_DN_FLA_RECOMPUTE_BACKWARD=1` to Exp3 | VRAM (nvidia-smi), throughput (may be slightly lower) |

**Commands (8B).**

```bash
cd /path/to/Test20_70B_Zero3_LeakFixed
export FORCE_REWRITE_INIT=0
export CFG=configs/test_8b_moe_lora_4096_bs32_10steps.yaml

# 1.1 Baseline
./run.sh
# Save: results/8b_run/profile_report.txt, results/8b_run/metrics.jsonl (throughput)

# 1.2 Exp1
T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=0 ./run.sh

# 1.3 Exp2
T17_DN_USE_FUSED_QKVG=0 T17_DN_USE_DELTA_ENTRANCE=1 ./run.sh

# 1.4 Exp3
T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=1 ./run.sh
# Or in config: model.deltanet_use_fused_entry: true
```

**Success (Phase 1):** Throughput stays in the same ballpark as baseline (no ~10× drop); Exp1/2/3 show expected kernel-time changes in the profile. If baseline was already broken by syncs, the MoE fix alone should restore throughput.

---

## Phase 2: 70B — Throughput only

Goal: confirm production throughput and VRAM with DeltaNet opts; no profiling (or 1 step only) to avoid OOM.

| Step | What | Config | Env | Record |
|------|------|--------|-----|--------|
| 2.1 | **Baseline** | 70B config, `profile_steps: []` | (none) | tok/s, VRAM per GPU |
| 2.2 | **Exp1** Fused QKVG | Same | `T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=0` | tok/s, VRAM |
| 2.3 | **Exp2** Fused entrance | Same | `T17_DN_USE_FUSED_QKVG=0 T17_DN_USE_DELTA_ENTRANCE=1` | tok/s, VRAM |
| 2.4 | **Exp3** Both | Same | `T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=1` or `model.deltanet_use_fused_entry: true` | tok/s, VRAM |
| 2.5 | **Exp4** (optional) FLA recompute | Same | Add `T17_DN_FLA_RECOMPUTE_BACKWARD=1` | tok/s, VRAM (expect slight drop in VRAM) |

**Commands (70B).**

```bash
export CFG=configs/test_70b_moe_lora_4096_bs32_10steps.yaml
export FORCE_REWRITE_INIT=0
# In the YAML: profile_steps: []

# 2.1 Baseline
./run.sh

# 2.2–2.4
T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=0 ./run.sh
T17_DN_USE_FUSED_QKVG=0 T17_DN_USE_DELTA_ENTRANCE=1 ./run.sh
T17_DN_USE_FUSED_QKVG=1 T17_DN_USE_DELTA_ENTRANCE=1 ./run.sh
```

**Success (Phase 2):** Throughput ~11k+ tok/s (no collapse); Exp1/2/3 at or above baseline; VRAM stable or lower.

---

## If you only have 3B config (no 8B)

Use 3B for Phase 1:

```bash
# Create a copy with profiling enabled
cp configs/test17_3b_moe_perf_4096_bs32_10steps.yaml configs/test17_3b_moe_perf_4096_bs32_profile.yaml
# Edit: profile_steps: [1, 2, 3, 4, 5] and e.g. metrics_jsonl_path: ../results/3b_run/metrics.jsonl

export CFG=configs/test17_3b_moe_perf_4096_bs32_profile.yaml
export FORCE_REWRITE_INIT=0
# Then run 1.1–1.4 with the same env vars (3B also has fused QKVG and fused entrance).
```

---

## Order summary

1. **8B (or 3B) first:** Baseline → Exp1 → Exp2 → Exp3 (optionally Exp4). Use profiling to confirm kernel timings and no sync-induced collapse.
2. **70B next:** Same order, throughput + VRAM only; `profile_steps: []`.

This way you catch sync/performance issues on the smaller model before running 70B.
