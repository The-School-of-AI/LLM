# Test_3_sandard_embedding_1000steps

## Objective
Winner-only baseline run (`diff_rec`) for 1000 steps with standard embeddings.

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_3_sandard_embedding_1000steps"
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
- model_variant: `diff_rec`
- embedding_type: `standard`
- max_train_steps: `1000`
- log_interval: `1` (every step)
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`
- global batch size: `32`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`
- Notes: `notes/reference_run.md`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with winner model only.
