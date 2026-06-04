# Training Cookbook

This cookbook walks through the full LightningLM training path: prepare data,
use or build the tokenizer, train 2B, grow to 5B, grow to 9B, and launch 120B
TurboQuantPretraining.

## 1. Environment

Install dependencies with one of the setup scripts:

```bash
bash scripts/setup_stable.sh
```

For AWS B200/B300-class training machines:

```bash
bash scripts/setup_aws_b300.sh
```

Run a health check:

```bash
python3 scripts/doctor.py
```

## 2. Tokenizer

The training configs point to `tokenizer/`. The directory already contains the
tokenizer JSON, config files, special tokens, and token permutation arrays.

Check the tokenizer:

```bash
python3 tokenizer/byte_analysis/analyze.py --tokenizer-dir tokenizer
```

To train or adapt a tokenizer, follow [tokenizer_pipeline.md](tokenizer_pipeline.md).

## 3. Data Shards

For a quick local check, create synthetic curriculum shards:

```bash
python3 scripts/create_curriculum_test_shards.py \
  --output-dir data/curriculum_test_shards \
  --manifest-dir manifests \
  --shards-per-pool 2
```

For real data, process raw text/parquet inputs:

```bash
python3 scripts/data/process.py \
  --input-dir /path/to/raw_inputs \
  --output-dir data/d1_shards \
  --tokenizer-dir tokenizer \
  --band-map configs/curriculum_v2.yaml \
  --verify-after
```

Verify generated shards:

```bash
python3 scripts/data/verify.py \
  --shard-dir data/d1_shards \
  --tokenizer-dir tokenizer
```

For larger curriculum runs, write or mount shards at
`data/training_shards_8k` or update the selected config to your shard root.

## 4. Train 2B

Config: `configs/train_2b.yaml`

```bash
NUM_GPUS=8 bash scripts/run_2b_stage.sh
```

Checkpoint output defaults to `results/run/checkpoints`. Update the config if
you want a different output path.

## 5. Grow 2B To 5B

Create the 5B MoE initialization checkpoint from the 2B dense checkpoint:

```bash
python3 -m lightninglm.growth.dense_to_moe \
  --src results/2b/checkpoint.pt \
  --dst results/5b/init_from_2b.pt \
  --strategy partition
```

Point `configs/train_5b.yaml` at that initialization checkpoint, then run:

```bash
NUM_GPUS=8 bash scripts/run_5b_stage.sh
```

## 6. Grow 5B To 9B

Create the 9B initialization checkpoint from the 5B checkpoint. This is the
depth-growth step:

```bash
python3 -m lightninglm.growth.depth_map \
  --src results/5b/checkpoint.pt \
  --dst results/9b/init_from_5b.pt \
  --mapping lightninglm_5b_to_9b
```

Point `configs/train_9b.yaml` at that initialization checkpoint, then run:

```bash
NUM_GPUS=8 bash scripts/run_9b_stage.sh
```

## 7. Train 120B TQP

Config: `configs/train_120b_tqp.yaml`

Build the 120B initialization checkpoint from the 9B checkpoint:

```bash
python3 scripts/build_120b_init.py \
  --src results/9b/checkpoint.pt \
  --dst results/120b_tqp/init/120b_init_proper_v2.pt \
  --config configs/train_120b_tqp.yaml \
  --ratio 0.5 \
  --router_sigma 0.05 \
  --seed 1337
```

Then launch TQP training:

```bash
NUM_GPUS=8 bash scripts/run_120b_tqp.sh
```

The TQP implementation lives in `lightninglm/tqp/`.
Runtime hot-configuration knobs are described in
[runtime_hotconfig.md](runtime_hotconfig.md).

## 8. Save Tensor Hashes

Hash checkpoints after each stage:

```bash
python3 scripts/hash_tensors.py \
  --checkpoint results/2b/checkpoint.pt \
  --out results/2b/tensor_hashes.json
```

Repeat for the 5B, 9B, and 120B checkpoints.

## 9. Useful Checks

```bash
python3 scripts/doctor.py
python3 -m compileall -q lightninglm scripts tokenizer/paper_artifacts tokenizer/byte_analysis
```
