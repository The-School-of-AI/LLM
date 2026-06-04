# Data Pipeline

The data pipeline converts raw text/parquet inputs into LightningLM shard
directories containing token blocks and metadata.

## Directory Convention

Use the `data/` directory for local development:

```text
data/d1_shards/              2B/5B/9B default shard root
data/curriculum_test_shards/ synthetic smoke-test shards
data/training_shards_8k/     large curriculum shard root for TQP stages
```

You can also mount data elsewhere and update the selected config.

## Process Raw Inputs

```bash
python3 scripts/data/process.py \
  --input-dir /path/to/raw_inputs \
  --output-dir data/d1_shards \
  --tokenizer-dir tokenizer \
  --band-map configs/curriculum_v2.yaml \
  --verify-after
```

Use `--sources` to process a comma-separated subset of source names.

## Verify Shards

```bash
python3 scripts/data/verify.py \
  --shard-dir data/d1_shards \
  --tokenizer-dir tokenizer
```

## Generate Synthetic Shards

For a quick loader check:

```bash
python3 scripts/create_curriculum_test_shards.py \
  --output-dir data/curriculum_test_shards \
  --manifest-dir manifests \
  --shards-per-pool 2
```

## Curriculum Manifests

The `manifests/` directory contains the curriculum lists consumed by
`lightninglm.data.curriculum_dataloader_v2`. When you build a new dataset,
generate a manifest set for your shard root:

```bash
python3 scripts/data/generate_manifest_v2.py
```

Keep the manifest directory path in sync with the selected training config.

## Cleaning Smoke Test

```bash
python3 scripts/data/test_cleaning_smoke.py --quick --no-examples
```

This checks the cleaning rules on a small sample and catches common input
format issues before a large processing run.
