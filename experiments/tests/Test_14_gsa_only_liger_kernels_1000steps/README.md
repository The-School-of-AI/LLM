# Test_14_gsa_only_liger_kernels_1000steps

## Objective
GSA-only reversible model (no DeltaNet): all 8 layers use GSA only. Liger kernels are used for RoPE, SwiGLU MLP, and fused linear+cross-entropy (CE). Training uses the fused CE path when `use_fused_ce: true`. Kronecker embeddings; 1000 steps.

## Run
```bash
cd "experiments/tests/Test_14_gsa_only_liger_kernels_1000steps"
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
- use_fused_ce: `true`
- log_interval: `1`
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with the reversible model only. No DeltaNet; Liger RoPE, Liger SwiGLU MLP, and Liger fused CE are used; fused CE is used in the training loop.
