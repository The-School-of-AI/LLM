# Test 1 - Short Comparative Run (Recurrence Variant Selection)

## Goal
Pick the better recurrence implementation using an apples-to-apples short run:
- `lead_wo_rev` (`recurrence_model_1b_wo_rev.py`)
- `diff_rec` (`different_recurrence_model_1b_wo_rev.py`)

## Why this exists
This test is the gate before all downstream ablations. We lock one winner here and use that same winner for later tests.

## Fixed controls (must remain identical across both runs)
- Dataset slice/order: `wikitext` + `wikitext-103-raw-v1`
- Sequence length: `512`
- Global batch size: `32` via DeepSpeed config
- Optimizer, LR schedule, grad clipping: identical (`deepspeed/zero-2-dense-bf16-test1-100steps.json`)
- Precision: `bf16`
- Seed: `42`
- Embedding type: `standard` (not Kronecker)
- Reversibility: off for both variants

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test 1"
./run.sh
```

Optional GPU override:
```bash
NUM_GPUS=4 ./run.sh
```

## Expected
- Both runs complete 100 steps without NaN/inf.
- Comparable curves under same setup.
- Select winner by primary criterion: lower NTP loss (`loss`), then throughput stability.

## Outputs
- Lead run log: `results/lead_wo_rev/train.log`
- Diff-rec run log: `results/diff_rec/train.log`
- Lead metrics: `results/lead_wo_rev/metrics.jsonl`
- Diff-rec metrics: `results/diff_rec/metrics.jsonl`
- Comparison notes: `notes/comparison.md`

## Self-sufficiency
This folder contains its own runnable code snapshot in `code/`:
- `code/main.py`
- `code/src/**`
- `code/requirements.txt`

No dependency on parent-folder code files during execution.
