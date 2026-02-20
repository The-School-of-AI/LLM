# Test_2_20-step_save_init_model

## Objective
20-step winner smoke run using `diff_rec`, with an explicit saved init model artifact.

## Winner fixed
- `model_variant: diff_rec` (already selected by team)

## What this folder guarantees
- Single-model test (no variant switching)
- Saves init model before training: `results/init/model_init.pt`
- Stores init metadata/hash: `results/init/model_init_meta.json`
- Runs 20 training steps with detailed loss logging

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_2_20-step_save_init_model"
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
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq len: `512`
- global batch size: `32`
- precision: `bf16`
- embedding: `standard`
- steps: `20`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`
- Notes: `notes/batch_replay_proof.md`

## Self-sufficient contents
- `code/main.py`
- `code/src/**`
- `code/requirements.txt`
- `configs/test2_diff_rec_20steps.yaml`
- `deepspeed/zero-2-dense-bf16-test2-20steps.json`
- `scripts/save_init_model.py`
