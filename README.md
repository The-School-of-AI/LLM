# LightningLM Training

LightningLM Training is a complete pretraining pipeline for building
LightningLM-style language models from a 2B dense seed stage through 5B, 9B, and
120B TurboQuantPretraining (TQP).

The repository includes the model code, tokenizer assets, data/shard pipeline,
curriculum manifests, growth utilities, DeepSpeed launch configs, checkpointing,
OPUS data-selection support, TQP modules, and reproducibility utilities.

## What You Can Do

- Use the included 131K tokenizer artifacts.
- Build your own tokenizer from text/parquet samples.
- Clean and tokenize raw datasets into `tokens.bin` shard directories.
- Generate and validate curriculum manifests.
- Train the 2B seed stage.
- Grow a checkpoint from 2B to 5B, then 5B to 9B.
- Launch the 120B TQP stage.
- Hash checkpoints tensor-by-tensor for transfer and reproducibility checks.

## Repository Layout

```text
lightninglm/              model, training, data loading, OPUS, TQP, kernels
configs/                  stage and curriculum configs
deepspeed/                DeepSpeed ZeRO configs
scripts/                  setup, launch, shard, growth, and validation commands
scripts/data/             dataset cleaning, tokenization, sharding, verification
scripts/tokenizer/        tokenizer build and analysis tools
tokenizer/                included tokenizer artifacts
manifests/                curriculum shard manifests
data/                     local shard/checkpoint mount points
docs/                     user guides
```

## Requirements

For setup and data preparation:

- Python 3.11+
- `pip`
- `pyarrow`, `datasets`, `tokenizers`, `transformers`, `numpy`, `pandas`

For training:

- CUDA-capable multi-GPU machine
- PyTorch and DeepSpeed matching your CUDA stack
- Large local NVMe or a fast mounted data volume
- AWS B200/B300-class hardware for the 120B TQP stage

The setup helpers are:

```bash
bash scripts/setup_stable.sh
bash scripts/setup_aws_b300.sh
```

## Quickstart

Run the repository health check:

```bash
python3 scripts/doctor.py
```

Create a tiny synthetic curriculum dataset:

```bash
python3 scripts/create_curriculum_test_shards.py \
  --output-dir data/curriculum_test_shards \
  --manifest-dir manifests \
  --shards-per-pool 2
```

Use the included tokenizer:

```bash
python3 tokenizer/byte_analysis/analyze.py --tokenizer-dir tokenizer
```

Train the 2B stage:

```bash
NUM_GPUS=8 bash scripts/run_2b_stage.sh
```

Grow and train the next stages:

```bash
python3 -m lightninglm.growth.dense_to_moe \
  --src results/2b/checkpoint.pt \
  --dst results/5b/init_from_2b.pt \
  --strategy partition

NUM_GPUS=8 bash scripts/run_5b_stage.sh

python3 -m lightninglm.growth.depth_map \
  --src results/5b/checkpoint.pt \
  --dst results/9b/init_from_5b.pt \
  --mapping lightninglm_5b_to_9b

NUM_GPUS=8 bash scripts/run_9b_stage.sh
```

Launch the 120B TQP stage:

```bash
python3 scripts/build_120b_init.py \
  --src results/9b/checkpoint.pt \
  --dst results/120b_tqp/init/120b_init_proper_v2.pt \
  --config configs/train_120b_tqp.yaml \
  --ratio 0.5 \
  --router_sigma 0.05 \
  --seed 1337

NUM_GPUS=8 bash scripts/run_120b_tqp.sh
```

## Data

For real training data, process raw datasets into shard directories:

```bash
python3 scripts/data/process.py \
  --input-dir /path/to/raw_inputs \
  --output-dir data/d1_shards \
  --tokenizer-dir tokenizer \
  --band-map configs/curriculum_v2.yaml \
  --verify-after
```

Then verify the shards:

```bash
python3 scripts/data/verify.py \
  --shard-dir data/d1_shards \
  --tokenizer-dir tokenizer
```

See [docs/data_pipeline.md](docs/data_pipeline.md) for the full data workflow.

## Tokenizer

The default configs use the included tokenizer in `tokenizer/`. To rebuild or
adapt a tokenizer:

```bash
python3 scripts/tokenizer/build_tokenizer.py \
  --data-dir /path/to/tokenizer_samples \
  --output-dir tokenizer_out \
  --work-dir tokenizer_work
```

See [docs/tokenizer_pipeline.md](docs/tokenizer_pipeline.md).

## Training Cookbook

The full stage-by-stage workflow lives in [docs/cookbook.md](docs/cookbook.md).
For runtime hot-configuration knobs used by the MoE stages, see
[docs/runtime_hotconfig.md](docs/runtime_hotconfig.md).

## Checkpoint Hashing

Create tensor hashes for checkpoints before upload, transfer, or comparison:

```bash
python3 scripts/hash_tensors.py \
  --checkpoint results/2b/checkpoint.pt \
  --out results/2b/tensor_hashes.json
```

The hash manifest records tensor name, dtype, shape, and SHA-256 digest.
