# Test19 ZeRO-3 Patch Notes

This folder is derived from `Test17_3B_Zero3` and applies targeted memory-stability fixes for 3B MoE + ZeRO-3.

## Core changes

1. Ported model/runtime fixes from Test18:
- `code/src/models/recurrence_model_3b_moe.py`
- `code/src/models/reversible_ops_midpoint.py`

2. Reduced ZeRO-3 transient memory spikes in training loop:
- `code/src/train.py`
- Fused CE now gathers `lm_head` under ZeRO-3 without cloning full `[V, H]` weights each step.

3. Tuned ZeRO-3 config for lower peak memory:
- `deepspeed/zero-3-moe-bf16.json`
- lower bucket sizes, disabled overlap comm, added stage3 prefetch/persistence thresholds.

4. Added fallback offload config:
- `deepspeed/zero-3-moe-bf16-offload.json`
- CPU optimizer offload for stability-first validation.

5. Runtime launcher updates:
- `run.sh` now defaults `CFG` to this folder's config and sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless already provided.
- Added runtime hardening defaults for memory-retention triage:
  - `TORCHDYNAMO_DISABLE=1`
  - `T19_STEP_CUDA_SYNC=1`
  - `T19_STEP_GC_COLLECT=1`
  - `T19_STEP_EMPTY_CACHE=1`
  - `T19_ZERO3_RELEASE_EVERY=1`
  - `T19_ZERO3_FORCE_CLEAR_CONTAINERS=0` (opt-in aggressive fallback)
  - `T19_CLEAR_ROUTER_CACHE_EVERY=1`
  - `T19_TRACK_CUDA_MEMORY=1`
  - Reversible checkpoint reentrant toggle: `T19_REV_CKPT_USE_REENTRANT={0|1}`

6. Added lower-context diagnostic config:
- `configs/test17_3b_moe_offload_1024.yaml`
- Purpose: check whether per-step growth slope persists even at seq=1024.

## Suggested AWS validation order

1. Try default config first:
- `configs/test17_3b_moe.yaml` with `../deepspeed/zero-3-moe-bf16.json`

2. If OOM persists at optimizer step, switch YAML `deepspeed.config_path` to:
- `../deepspeed/zero-3-moe-bf16-offload.json`

3. Keep:
- `batch_size=1`
- `sequence_length=4096`
- `gradient_accumulation_steps=1`
- run at least 20 steps to validate no ramping/step-5 failure.

## Fast triage matrix (40GB A100)

1. Baseline hardening (2048):
- `CFG=configs/test17_3b_moe_offload_2048.yaml ./run.sh`

2. Lower context slope check (1024):
- `CFG=configs/test17_3b_moe_offload_1024.yaml ./run.sh`

3. Reentrant checkpoint variant:
- `T19_REV_CKPT_USE_REENTRANT=1 CFG=configs/test17_3b_moe_offload_1024.yaml ./run.sh`

4. Aggressive ZeRO cache-container clear (debug only):
- `T19_ZERO3_FORCE_CLEAR_CONTAINERS=1 CFG=configs/test17_3b_moe_offload_1024.yaml ./run.sh`
