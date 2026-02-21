# Test_16_lucid_precond_gsa

## Objective
GSA layers enhanced with LUCID preconditioning (arXiv:2602.10410). LUCID decorrelates keys in RKHS by solving a triangular system `P·Y = V` where `P = exp(K_RN·K_RN⊤/√d − √d)`, producing preconditioned values `Y` that are fed to the existing sparse attention kernel. DeltaNet layers remain unchanged. Liger kernels used for RoPE, SwiGLU MLP, and fused linear+CE. Kronecker embeddings; 1000 steps.

## Run
```bash
cd "experiments/tests/Test_16_lucid_precond_gsa"
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
- use_lucid_precond: `true`
- log_interval: `1`
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`

## What's new vs Test_14
- Added `lucid_preconditioner.py` — PyTorch reference + Triton kernel for block-wise triangular solve
- GSA layers apply LUCID preconditioning to values before sparse attention
- New config flag: `use_lucid_precond: true`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with the reversible model and LUCID preconditioning kernel.
