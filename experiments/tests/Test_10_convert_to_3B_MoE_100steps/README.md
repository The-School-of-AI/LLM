# Test_10_convert_to_3B_MoE_100steps

## Objective
Convert the winner reversible 1B stack into the agreed 3B-class MoE profile and run a 100-step gated smoke.

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_10_convert_to_3B_MoE_100steps"
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
- max_train_steps: `100`
- log_interval: `1` (every step)
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`
- global batch size: `32`

## 3B-class MoE profile in this test
- hidden_size: `4096`
- layers: `8` (`DDDGDDDG`)
- DeltaNet/GSA: `6/2`
- real experts: `20`
- null experts: `20` (derived from `rho=0.5`)
- total slots: `40`
- top_k: `2`
- routed intermediate: `1024`
- shared intermediate: `2048`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`
- Notes: `notes/reference_run.md`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with only the reversible model stack files needed for this run.

Init script hard checks:
- reversible stack is active
- `DDDGDDDG` layer pattern is active
- Triton GSA + fused DeltaNet kernels are present
- additional fused kernels path (fused CE + fused SwiGLU) is present
- MoE shape is exactly `20 real / 20 null / top_k=2 / shared=2048 / routed=1024`
- Kronecker path is active (`token_embed=None`, `pf_to_model` present)
