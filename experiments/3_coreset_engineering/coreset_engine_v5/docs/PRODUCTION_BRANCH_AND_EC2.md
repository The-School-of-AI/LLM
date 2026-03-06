# Production Branch and EC2 Setup

This doc gives exact commands to create a production branch (with only pipeline-relevant files), push to GitHub, and pull/run on EC2.

## What’s included vs excluded

**Included (needed for pipeline run):**

- `coreset_builder.py`, `shard.sh`, `requirements.txt`
- `config/` (e.g. `pipeline.yaml`, `curriculum_t3_aligned.yaml`)
- `src/` (selection, io, curriculum, core, diversity, dedup, error_handling)
- `tools/` (e.g. `merge_selected_indices.py`, `validate_coreset_outputs.py`, `generate_books_sample.py`, and other run/validation tools)
- `scripts/` (e.g. `commands.sh`, `run_local_books_test.sh`, `setup_nvme.sh`, `validate_infra.sh`, monitoring)
- `emr/`, `glue/` (if you use EMR/Glue in production)
- `production_run_steps.md`, `README.md`, `VALIDATION_README.md`
- `docs/` (reports and guides)
- `tests/`, `conftest.py` (optional but useful)

**Excluded via `.gitignore` (not needed for run):**

- `t2_starcoder.py`, `starcoder_preview.csv`, `starcoder.md` (initial review)
- `T3StatsFromT2.txt`, `validation_output.txt`, `VALIDATION_COMPLETE.md`
- Root-level sample parquet: `part-*.parquet`, `selected_indices*.parquet`
- `output/`, `data/`, `*.log` (already in `.gitignore`)

---

## 1. Create branch and push (on your machine)

Run from the **LLM repo root** (parent of `experiments/`).

```bash
cd /Users/shwethd/Desktop/Coreset/LLM

# Ensure remote points to School of AI repo (if not already)
git remote -v
# If 'origin' is not The-School-of-AI/LLM, add it:
# git remote add origin https://github.com/The-School-of-AI/LLM.git

# Create and switch to a new branch (e.g. for production)
git checkout -b p3/feat/coreset-engine-v5-production

# Stage only under experiments/3_coreset_engineering/coreset_engine_v5
# (review-only files are ignored by .gitignore there)
git add experiments/3_coreset_engineering/coreset_engine_v5/

# Optional: stage any dependency/config at repo root used by the pipeline (e.g. pyproject.toml, uv.lock under experiments)
git add experiments/3_coreset_engineering/pyproject.toml experiments/3_coreset_engineering/uv.lock 2>/dev/null || true
git add pyproject.toml uv.lock 2>/dev/null || true

# Commit and push
git status   # sanity check: no t2_starcoder.py, starcoder_preview.csv, T3StatsFromT2.txt, root parquets
git commit -m "Coreset engine v5: production-ready branch (pipeline code, config, tools, scripts)"
git push -u origin p3/feat/coreset-engine-v5-production
```

---

## 2. On EC2: clone or pull and run

### Option A: Fresh clone

```bash
git clone https://github.com/The-School-of-AI/LLM.git
cd LLM
git checkout p3/feat/coreset-engine-v5-production
cd experiments/3_coreset_engineering/coreset_engine_v5
```

### Option B: Already have LLM repo

```bash
cd /path/to/LLM
git fetch origin
git checkout p3/feat/coreset-engine-v5-production
git pull origin p3/feat/coreset-engine-v5-production
cd experiments/3_coreset_engineering/coreset_engine_v5
```

### Install deps and run (from `coreset_engine_v5`)

Use the same env as your local books test (uv, PYTHON_BIN for `shard.sh`):

```bash
# From experiments/3_coreset_engineering or repo root, depending where pyproject.toml/uv.lock live
cd /path/to/LLM/experiments/3_coreset_engineering
uv sync --all-packages

# Run pipeline (e.g. via commands.sh as in production_run_steps.md)
cd coreset_engine_v5
# Copy commands.sh to /mnt/nvme if using NVMe; set S3_BUCKET, NUM_SHARDS, STAGES, ENABLE_NVME, etc.
# Then from /mnt/nvme: ./commands.sh
```

Or run the engine directly (see `production_run_steps.md` for full env vars):

```bash
export PYTHON_BIN=$(which python)
./shard.sh
```

---

## Summary

| Step | Where | Action |
|------|--------|--------|
| 1 | Local | Update `.gitignore` in `coreset_engine_v5` (already done); create branch, `git add experiments/3_coreset_engineering/coreset_engine_v5/`, commit, push |
| 2 | EC2 | Clone or pull branch `p3/feat/coreset-engine-v5-production`, `uv sync`, run via `commands.sh` or `shard.sh` per `production_run_steps.md` |

Branch name is a suggestion; you can use e.g. `p3/feat/coreset-v5-prod` if you prefer.
