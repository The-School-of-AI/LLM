# Test_11_add_fused_moe_to_3B_1000steps

## Objective
Take Test 10's 3B-class MoE profile and enable fused MoE dispatch (grouped GEMM), then run 1000 steps.

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_11_add_fused_moe_to_3B_1000steps"
./run.sh
```

Optional:
```bash
NUM_GPUS=4 ./run.sh
```

Force regenerate init model:
```bash
FORCE_REWRITE_INIT=1 ./run.sh
```

## Fixed controls
- model_variant: `reversible`
- embedding_type: `kronecker`
- max_train_steps: `1000`
- log_interval: `1` (every step)
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`
- global batch size: `32`

## 3B-class MoE profile
- hidden_size: `4096`
- layers: `8` (`DDDGDDDG`)
- DeltaNet/GSA: `6/2`
- real experts: `20`
- null experts: `20`
- total slots: `40`
- top_k: `2`
- routed intermediate: `1024`
- shared intermediate: `2048`

## Fused MoE addition in Test 11
- `moe_backend: grouped_gemm`
- `require_fused_moe_kernel: true`
- init assertions require grouped GEMM kernel availability and active backend = `grouped_gemm`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`
- Notes: `notes/reference_run.md`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with the reversible model stack files needed for this run.
