# Coreset Selection Engine for 70B LLM Pre-training

**Version**: 1.0.0  
**Status**: Production Ready  
**Team**: Coreset Selection Architecture

## Overview

The Coreset Selection Engine is a production-grade pipeline that compresses 2 trillion tokens to ~400 billion tokens for efficient 70B parameter LLM pre-training. The engine uses curriculum-aware stratified sampling, deduplication, and diversity optimization to create high-quality training datasets across multiple stages.

**Key Features**:

- ✅ **Deterministic & Reproducible**: Fully seeded, version-controlled pipeline
- ✅ **Curriculum-Compliant**: Strict adherence to frozen curriculum specifications
- ✅ **Scalable (2T-ready)**: Streaming/batched selection with checkpoint/resume and optional sharding
- ✅ **Protective**: Preserves rare, capability-critical content (B4/B5, code, agentic, Indic)
- ✅ **Non-overlap Across Stages (Streaming)**: Disk-backed used-chunk membership store
- ✅ **Auditable**: Detailed manifests, rolling-window stats, and validation/checklist reports per stage

## Quick Start

### Prerequisites

```bash
Python 3.10+
CUDA 11.8+ (optional, for GPU acceleration)
```

### Installation

```bash
# Clone repository
git clone <repo-url>
cd coreset_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Recommended: streaming builder (default, 2T-safe)
# Requires an input dataset path (file or directory).
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --input-path data/datasets/sample_chunks.jsonl \
  --input-format jsonl \
  --checkpoint-dir output/checkpoints \
  --checkpoint-every-n-batches 3

# Run with custom stages
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --input-path data/datasets/sample_chunks.jsonl \
  --input-format jsonl \
  --stages 1B 3B 8B 70B

# Debug/smoke on small datasets: scale targets down while exercising real constraints
python coreset_builder.py \
  --config config/pipeline_large_only.yaml \
  --curriculum data/datasets/curriculum_min_for_large_test.yaml \
  --input-path data/datasets/large_sample_chunks.jsonl \
  --input-format jsonl \
  --batch-size 5000 \
  --checkpoint-dir output/checkpoints_smoke \
  --total-input-tokens-estimate 15001600 \
  --stage-target-scale 0.00005 \
  --stages 1B 3B 8B 70B

# Legacy mode (in-memory; not 2T-safe)
python coreset_builder.py \
  --legacy \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml

# Run ablation study (no near-dedup)
python coreset_builder.py \
  --config config/ablation_no_neardup.yaml \
  --curriculum config/curriculum.yaml \
  --input-path data/datasets/sample_chunks.jsonl \
  --input-format jsonl \
  --ablation-variant no_neardup
```

Note: default `--batch-size` is `80000`; reduce it on low-memory machines.

### How to Pick `--total-input-tokens-estimate` and `--stage-target-scale`

- `--stage-target-scale` multiplies the **global** curriculum stage targets (e.g., `70B.total_tokens`) so you can run end-to-end on small samples.
  - Streaming selection uses an **effective per-worker stage budget** computed as:
    $$T_{\mathrm{shard}} = \left\lfloor\frac{T_{\mathrm{global}} \times s}{N_{\mathrm{shards}}}\right\rfloor$$
  - The total targeted across all shards is therefore approximately:
    $$\sum T_{\mathrm{shard}} \approx T_{\mathrm{global}} \times s$$
  - where $T_{\mathrm{global}}$ = global stage target, $s$ = `--stage-target-scale`, and $N_{\mathrm{shards}}$ = `--num-shards`.
  - Examples:
    - Global target = 2,000,000,000, `--num-shards 2`, `--stage-target-scale 1.0` → each shard targets ~1,000,000,000 (total ~2,000,000,000).
    - Global target = 2,000,000,000, `--num-shards 2`, `--stage-target-scale 0.5` → each shard targets ~500,000,000 (total ~1,000,000,000).
  - Practical rule: if you want to reach the full global target across shards, use `--stage-target-scale 1.0` (sharding already divides the work).
- `--total-input-tokens-estimate` enables *proportional per-batch budgeting* in streaming mode.
  - The batched engine computes each batch’s selection budget roughly as:
    $$B_{\mathrm{target}} = T_{\mathrm{stage}} \times \frac{B_{\mathrm{raw}}}{T_{\mathrm{input}}}$$
  - where $B_{\mathrm{target}}$ = batch target, $T_{\mathrm{stage}}$ = stage target, $B_{\mathrm{raw}}$ = raw tokens in the current batch, and $T_{\mathrm{input}}$ = `--total-input-tokens-estimate`.
  - If you omit it, the engine defaults to “select everything in batch” (not representative for 2T-scale behavior).

**Practical way to compute it** (exact for typical JSONL/Parquet inputs):

```bash
# JSONL (file or directory)
python tools/estimate_total_tokens.py --input-path data/datasets/large_sample_chunks.jsonl --input-format jsonl

# Parquet (file or directory)
python tools/estimate_total_tokens.py --input-path data/datasets --input-format parquet
```

Use the printed `total_tokens` value as `--total-input-tokens-estimate`.

### Sharding (Multi-Worker) and Non-Overlap (Streaming)

Streaming runs support multi-worker sharding with:

- `--num-shards N`: total number of workers
- `--shard-id K`: this worker’s index, `0..N-1`

#### Stage budgets in sharded runs

- In streaming mode, each worker applies the **per-shard** stage budget (`target_tokens_shard`) computed from the global stage target, `--stage-target-scale`, and `--num-shards` (see formula above).
- Manifests record both values for clarity:
  - `target_tokens_global`: pre-scaling, pre-shard-split stage target
  - `target_tokens_shard`: effective per-worker target budget used by selection
- When `--num-shards > 1`, each shard only sees a *subset* of the data. Do not compare a single shard’s `actual_tokens` to a `--num-shards 1` run; compare the **merged** output across all shards (and ensure you actually ran all shard workers `0..N-1`).

#### How sharding is applied

- **Many input files (directory input): file-level sharding**
  - When `--input-path` is a directory containing multiple `*.jsonl` (or `*.parquet`) files, files are deterministically assigned to shards.
  - Each worker reads a disjoint subset of files.

- **Single JSONL file: row-level sharding by chunk identifier**
  - If `--input-path` resolves to exactly one JSONL file and `--num-shards > 1`, each worker reads the file but only keeps rows whose identifier hashes to its shard.
  - The identifier is taken from `chunk_id` if present, otherwise `uid`, `guid`, or `id`.
  - If `chunk_id` exists but is empty (e.g., `""`), it is treated as missing and the fallback (`uid/guid/id`) is used for sharding.
  - This guarantees each chunk belongs to exactly one worker (deterministically) assuming the identifier is stable.

#### Checkpointing: use a unique directory per shard

Checkpoint filenames are keyed by stage and batch number. If multiple shards share the same `--checkpoint-dir`, they will overwrite each other’s checkpoints.

Checkpoint cadence is configurable in streaming mode via:

- `--checkpoint-every-n-batches N`
  - `N=3` (default): checkpoint every 3 successful batches
  - `N=1`: checkpoint every successful batch
  - `N>1`: checkpoint every N successful batches, and always write a final checkpoint at stage end

Examples:

```bash
# default behavior (every batch)
python coreset_builder.py ... --checkpoint-dir output/checkpoints --checkpoint-every-n-batches 3

# reduced checkpoint write frequency
python coreset_builder.py ... --checkpoint-dir output/checkpoints --checkpoint-every-n-batches 10
```

Important: if you change `--num-shards`, `--shard-id`, or `--stage-target-scale`, you must use a fresh `--checkpoint-dir` (or delete old checkpoints). Checkpoints are only safely resumable when these run parameters are unchanged.

Recommended pattern:

- `--checkpoint-dir output/checkpoints_1B/shard000`
- `--checkpoint-dir output/checkpoints_1B/shard001`
- …

#### Non-overlap across stages (streaming mode)

Streaming enforces cross-stage non-overlap via a disk-backed used-chunk membership store under `output/coresets/.used_chunks/`.

- Before selection, candidates are filtered against the store.
- After each batch, selected ids are added to the store.

Operational rules:

- Keep the same `--num-shards` and shard-id mapping across all stages.
- Do not delete `output/coresets/.used_chunks/` between stages if you want strict non-overlap.
- Do not run multiple processes with the same `--shard-id` writing to the same output path concurrently.
- Prefer running stages sequentially (finish 1B across all shards, then 3B, etc.) so later stages don’t race ahead of the used-chunk updates.

#### Example: 8-worker run for a single JSONL input (stage 1B)

Run these on 8 separate machines/processes, changing only `--shard-id` and `--checkpoint-dir`:

```bash
# shard 0
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path /path/to/data.jsonl --input-format jsonl --stages 1B \
  --num-shards 8 --shard-id 0 --checkpoint-dir output/checkpoints_1B/shard000 \
  --batch-size 80000 --total-input-tokens-estimate <TOTAL_TOKENS>

# shard 1
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path /path/to/data.jsonl --input-format jsonl --stages 1B \
  --num-shards 8 --shard-id 1 --checkpoint-dir output/checkpoints_1B/shard001 \
  --batch-size 80000 --total-input-tokens-estimate <TOTAL_TOKENS>

# ... repeat for shard-id 2..7 with distinct checkpoint-dir
```

Note: for single-file inputs, all shards will read the same file. For best throughput at very large scale, consider pre-splitting the JSONL into multiple files so file-level sharding can distribute I/O.

### Sharded Execution with `shard.sh`

The `shard.sh` script provides a convenient way to launch and manage multiple parallel shards on a single machine using background processes.

#### 1. Fresh Start (Standard Run)

Use this if it is your first time running the job or if you want to wipe previous results and start fresh. It will delete the `output/checkpoints` and `output/coresets` folders before starting.

```bash
bash shard.sh \
  --num-shards 8 \
  --stages "1B 3B 8B 70B" \
  --input-path "data/combined/bands/" \
  --input-format parquet \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --checkpoint-base output/checkpoints \
  --batch-size 80000 \
  --band-inference none \
  --band-score-source auto \
  --total-tokens 4523096944 \
  --checkpoint-every-n-batches 3 \
  --used-cache-max-entries 0 \
  --used-cache-stats-every 0 \
  --batch-prefetch-mode auto
```

#### 2. Resume (Continue Interrupted Job)

Use this if your job was interrupted (e.g., manual kill or crash). It will **keep** existing progress and skip any data that has already been processed by each shard.

```bash
bash shard.sh \
  --num-shards 8 \
  --stages "1B 3B 8B 70B" \
  --input-path "data/combined/bands/" \
  --input-format parquet \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --checkpoint-base output/checkpoints \
  --batch-size 80000 \
  --band-inference none \
  --band-score-source auto \
  --total-tokens 4523096944 \
  --checkpoint-every-n-batches 3 \
  --used-cache-max-entries 0 \
  --used-cache-stats-every 0 \
  --batch-prefetch-mode auto \
  --resume
```

#### 3. Complete Template (All Parameters + Defaults)

Use this as a baseline command. Keep flags you need, and remove/override the rest.

```bash
bash shard.sh \
  --num-shards 4 \
  --stages "1B 3B 8B 70B" \
  --input-path "data/datasets/large_sample_chunks.parquet" \
  --input-format parquet \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --checkpoint-base output/checkpoints \
  --band-inference none \
  --band-score-source auto \
  --batch-size 80000 \
  --checkpoint-every-n-batches 3 \
  --used-cache-max-entries 0 \
  --used-cache-stats-every 0 \
  --batch-prefetch-mode auto \
  --batch-prefetch-queue-size 1 \
  --batch-prefetch-auto-min-batch-size 50000 \
  --batch-prefetch-auto-max-shard-cpu-ratio 1.0 \
  --batch-prefetch-auto-min-wait-ms 2.0 \
  --batch-prefetch-auto-warmup-batches 5 \
  --total-tokens 4523096944
# Add --resume to continue from checkpoints.
```

#### 4. Minimal Command (Only Required Flags)

Use this shortest form when you want `shard.sh` defaults for everything except the required input path.

```bash
bash shard.sh \
  --input-path "data/combined/bands/"
```

Equivalent minimal run with an explicit token estimate (recommended for proportional per-batch budgeting):

```bash
bash shard.sh \
  --input-path "data/combined/bands/" \
  --total-tokens 4523096944
```

#### `shard.sh` Parameters

`shard.sh` accepts the following parameters:

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-path` | Yes | — | Input file or directory path (local FS or `s3://...`) |
| `--num-shards` | No | `4` | Number of parallel shard workers to launch |
| `--stages` | No | `"1B 3B 8B 70B"` | Space-separated stage list |
| `--input-format` | No | `parquet` | Input format passed to `coreset_builder.py` (`parquet` or `jsonl`) |
| `--config` | No | `config/pipeline.yaml` | Pipeline config path |
| `--curriculum` | No | `config/curriculum.yaml` | Curriculum config path |
| `--checkpoint-base` | No | `output/checkpoints` | Base directory where per-shard checkpoint folders are created |
| `--total-tokens` | No | empty | Passed as `--total-input-tokens-estimate` for proportional per-batch budgeting |
| `--batch-size` | No | `80000` | Rows/chunks processed per batch |
| `--checkpoint-every-n-batches` | No | `3` | Checkpoint cadence (`3` default; `1` = every batch; `N>1` = every N batches, plus stage-end checkpoint) |
| `--used-cache-max-entries` | No | `0` | Optional in-memory LRU size for used-chunk lookups (`0` disables cache) |
| `--used-cache-stats-every` | No | `0` | Emit periodic used-cache hit-rate logs every N batches (`0` disables periodic logs) |
| `--batch-prefetch-mode` | No | `auto` | Batch prefetch mode for `_iter_batches`: `off`, `on`, `auto` |
| `--batch-prefetch-queue-size` | No | `1` | Prefetch queue depth (number of prefetched batches buffered) |
| `--batch-prefetch-auto-min-batch-size` | No | `50000` | In `auto`, disable prefetch for smaller batches |
| `--batch-prefetch-auto-max-shard-cpu-ratio` | No | `1.0` | In `auto`, disable prefetch when `num_shards / cpu_count` is high |
| `--batch-prefetch-auto-min-wait-ms` | No | `2.0` | Warmup wait threshold used for prefetch usefulness logging |
| `--batch-prefetch-auto-warmup-batches` | No | `5` | Warmup batch count before usefulness check |
| `--band-inference` | No | `none` | Band inference mode: `none`, `infer_if_missing`, `infer_if_ineligible`, `force` |
| `--band-score-source` | No | `auto` | Score source for band inference (`auto`, `band_score`, `difficulty_score`, `band_p_max`, `band_p_argmax`, `band_p_B0..band_p_B5`) |
| `--resume` | No | `false` | Resume from existing checkpoints and skip output cleanup |

High-impact tuning knobs:

- `--batch-size`: controls memory pressure vs throughput (bigger is faster until memory becomes tight).
- `--checkpoint-every-n-batches`: reduces checkpoint write amplification while keeping resumability.
- `--used-cache-max-entries`: can reduce SQLite read pressure on repeated membership checks.
- `--batch-prefetch-mode`: overlaps I/O and compute (`auto` chooses based on shard/CPU and batch-size heuristics).

For most local/manual runs via `shard.sh`, set both explicitly:

- `--batch-size` based on memory headroom (default `80000`; reduce if RAM is tight).
- `--checkpoint-every-n-batches` to reduce checkpoint churn (e.g., `5` or `10`).

Prefetch mode guidance:

- `--batch-prefetch-mode off`: fully disable prefetch (simplest debug mode)
- `--batch-prefetch-mode on`: always prefetch next batch with a single producer + FIFO queue
- `--batch-prefetch-mode auto`: enable/disable based on runtime parameters (batch size, shard-to-CPU ratio), and log warmup usefulness

Advanced prefetch tuning flags (optional; usually keep defaults):

| Flag | Default | What it controls | When to change |
| --- | --- | --- | --- |
| `--batch-prefetch-queue-size` | `1` | Number of prefetched batches buffered ahead | Increase to `2` only if you still observe loader stalls and have extra RAM |
| `--batch-prefetch-auto-min-batch-size` | `50000` | In `auto`, disables prefetch for smaller batches | Lower only if you want prefetch on smaller test/debug batches |
| `--batch-prefetch-auto-max-shard-cpu-ratio` | `1.0` | In `auto`, disables prefetch when `num_shards / cpu_count` is too high | Lower for CPU-constrained hosts; raise if you want prefetch to stay on more aggressively |
| `--batch-prefetch-auto-min-wait-ms` | `2.0` | Warmup threshold for “is prefetch helping?” logging | Mostly for observability tuning; rarely needed |
| `--batch-prefetch-auto-warmup-batches` | `5` | Number of batches before warmup usefulness check | Increase for noisy workloads, decrease for faster feedback |

Do users need to pass these manually?

- **No**, not for normal runs.
- Typical usage is just `--batch-prefetch-mode auto` (or `off` / `on`).
- The advanced flags are expert tuning knobs; defaults are designed to work safely without extra input.
- `shard.sh` and `commands.sh` now also expose the advanced prefetch knobs; you can still leave them at defaults in most runs.

Quick tuning presets:

- **CPU-constrained host (many shards, limited vCPU headroom)**
  - Keep `--batch-prefetch-mode auto`
  - Use `--batch-prefetch-auto-max-shard-cpu-ratio 0.75`
  - Keep queue conservative: `--batch-prefetch-queue-size 1`
  - Optional env equivalents: `BATCH_PREFETCH_MODE=auto`, `BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=0.75`, `BATCH_PREFETCH_QUEUE_SIZE=1`

- **High-RAM host (good memory headroom, occasional loader stalls)**
  - Keep `--batch-prefetch-mode auto`
  - Increase queue depth: `--batch-prefetch-queue-size 2`
  - Make auto mode more permissive: `--batch-prefetch-auto-max-shard-cpu-ratio 1.25`
  - Optional env equivalents: `BATCH_PREFETCH_MODE=auto`, `BATCH_PREFETCH_QUEUE_SIZE=2`, `BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=1.25`

Switch back to defaults (`queue-size=1`, `auto-max-shard-cpu-ratio=1.0`) once prefetch logs show consistently low producer/consumer waits and no recurring loader stalls.

#### Sizing `--used-cache-max-entries` and `--used-cache-stats-every`

Use shard-aware sizing (not global chunk count):

- Start `--used-cache-max-entries` at about **0.5% to 2% of per-shard unique chunks**.
- Compute per-shard chunks as:
  $$C_{\mathrm{shard}} \approx \frac{C_{\mathrm{total}}}{N_{\mathrm{shards}}}$$
- Then size cache as:
  $$E_{\mathrm{cache}} \approx (0.005\ \text{to}\ 0.02) \times C_{\mathrm{shard}}$$

Example for **1B total chunks**:

- `num-shards=8` → `chunks_per_shard ≈ 125M`
- recommended cache range ≈ **625k to 2.5M** entries
- `--used-cache-max-entries 1000000` is a good practical starting point.

Memory rule-of-thumb for cache budget:

$$\text{RAM} \approx \text{entries} \times (150\ \text{to}\ 350\ \text{bytes})$$

So `1,000,000` entries is typically on the order of **~150MB to ~350MB** (can vary by id length/object overhead).

For `--used-cache-stats-every`, use log cadence based on total batches:

- target roughly every **1% to 5%** of batches
- practical formula:
  $$S_{\mathrm{every}} \approx \max\left(20,\ \left\lfloor\frac{B_{\mathrm{total}}}{50}\right\rfloor\ \text{to}\ \left\lfloor\frac{B_{\mathrm{total}}}{20}\right\rfloor\right)$$
- `--used-cache-stats-every 100` is reasonable for long multi-batch runs.

#### Used-cache hit rate: calculation and why it matters

When used-cache is enabled (`--used-cache-max-entries > 0`), the pipeline logs cache stats in stdout and `coreset_selection.log`:

- periodic (if `--used-cache-stats-every > 0`): `used-cache: size=... hit_rate=... hits=... misses=...`
- stage-final summary: `used-cache final: size=... hit_rate=... hits=... misses=...`

Hit-rate formula:

$$R_{\mathrm{hit}}(\%) = 100 \times \frac{H}{H + M}$$

Where:

- `hits`: membership checks answered directly from in-memory cache
- `misses`: checks not found in cache (these require SQLite lookup, then cache backfill)

Interpretation guide:

- `<40%`: cache is likely too small (or IDs are highly non-repeating)
- `40%–70%`: decent benefit
- `>70%`: strong benefit, many lookups are avoiding disk/SQLite reads

Why this helps:

- higher hit rate means fewer SQLite membership reads in the hot path
- that reduces I/O and lock pressure and generally lowers per-batch latency
- with stable memory, increasing cache size is useful only until hit rate plateaus

### Optional: Band Inference for Curriculum Eligibility

Some datasets provide a `band` label that makes large portions of the data curriculum-ineligible for early stages (e.g., many rows labeled `B0` but with domains only allowed in higher bands). In streaming mode, this can lead to unexpectedly low selection volume and skewed language/domain composition.

To address this, the streaming builder supports optional band inference using a configurable score source.

Band inference has two knobs:

- `--band-inference`: when to infer (none / infer_if_missing / infer_if_ineligible / force)
- `--band-score-source`: which field to use as the signal for inference

- `--band-inference none` (default): do not modify input bands.
- `--band-inference infer_if_ineligible`: only re-band rows when the current `(band, domain)` is not eligible under the curriculum.
- `--band-inference infer_if_missing`: only infer when `band` is missing/invalid.
- `--band-inference force`: always infer a band when a score exists.

#### Band score source (`--band-score-source`)

- `auto` (default): `band_score → difficulty_score → band_p_max`
- `band_score`: use `band_score` only
- `difficulty_score`: use `difficulty_score` only
- `band_p_max`: use `max(band_p_B0..band_p_B5)` as a continuous score (0..1)
- `band_p_argmax`: infer the *discrete* band as `argmax(band_p_B0..band_p_B5)` when band inference triggers
- `band_p_Bx`: pin to a single probability column (e.g., `band_p_B5`)

Notes:

- `band_p_argmax` is the right choice when `band_p_B0..B5` represent a classifier distribution over bands.
- `band_p_max` treats the maximum probability as a continuous “difficulty-like” score; this can behave differently than argmax.

Recommended setting for datasets with suspicious band/domain mismatches:

```bash
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path data/cdset --input-format jsonl \
  --stages 1B 3B 8B 70B \
  --band-inference infer_if_ineligible \
  --band-score-source auto
```

If your input provides `band_p_B0..band_p_B5`, you can infer directly from the argmax:

```bash
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path data/cdset --input-format parquet \
  --stages 1B 3B 8B 70B \
  --band-inference infer_if_ineligible \
  --band-score-source band_p_argmax
```

For sharded runs via `shard.sh`, you can pass the same option:

```bash
bash shard.sh --input-path "data/cdset" --band-inference infer_if_ineligible --band-score-source band_p_argmax
```

### Deployment with `commands.sh`

`commands.sh` is a wrapper script that automates the full setup and execution of the coreset pipeline. It handles system dependencies, repository cloning, virtual environment setup, and launches `shard.sh` with the correct parameters.

#### Execution Modes

| Mode | Command | Use Case |
| --- | --- | --- |
| **Manual EC2** | `./commands.sh` | Full setup on a fresh EC2 instance (clones repo, installs deps, runs pipeline in background via `nohup`) |
| **Dry Run** | `./commands.sh --dry-run` | Validates setup steps without launching the pipeline — useful for debugging |
| **CI (self-hosted)** | `./commands.sh --foreground --skip-repo-setup` | For GitHub Actions self-hosted runners where `actions/checkout` already cloned the repo |
| **CI (SSH)** | `./commands.sh --foreground` | For GitHub Actions SSH deployment — clones repo on the remote EC2 instance |

#### Flags

| Flag | Effect |
| --- | --- |
| `--foreground` | Runs `shard.sh` in the foreground (no `nohup`). Required for CI so the job waits for completion. |
| `--skip-repo-setup` | Skips `git clone` and uses the current directory as the repo root. Includes safety checks: verifies `.git` exists, remote URL matches, and critical files are present. |
| `--dry-run` | Prints what each step would do without executing. No pipeline is launched. |

#### Environment Variables

All parameters are configured via environment variables (with defaults):

| Variable | Default | Description |
| --- | --- | --- |
| `S3_BUCKET` | *(required)* | S3 bucket name for input data |
| `S3_INPUT_PATH` | `s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/source=ncert/` | S3 prefix containing input JSONL/Parquet files |
| `NUM_SHARDS` | `8` | Number of parallel shards |
| `STAGES` | `1B` | Space-separated stage list (e.g., `"1B 3B 8B 70B"`) |
| `TOTAL_TOKENS` | `4523096944` | Total token count of the input dataset (must match `S3_INPUT_PATH`) |
| `BATCH_SIZE` | `80000` | Passed to `shard.sh --batch-size`; controls rows/chunks processed per batch |
| `CHECKPOINT_EVERY_N_BATCHES` | `3` | Streaming checkpoint cadence (`3` default; `1` = every batch; `N>1` = every N batches) |
| `USED_CACHE_MAX_ENTRIES` | `0` | Passed to `shard.sh --used-cache-max-entries` (`0` disables in-memory used-cache) |
| `USED_CACHE_STATS_EVERY` | `0` | Passed to `shard.sh --used-cache-stats-every` (`0` disables periodic hit-rate logging) |
| `BATCH_PREFETCH_MODE` | `auto` | Passed to `shard.sh --batch-prefetch-mode` (`off`, `on`, `auto`) |
| `BATCH_PREFETCH_QUEUE_SIZE` | `1` | Passed to `shard.sh --batch-prefetch-queue-size` |
| `BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE` | `50000` | Passed to `shard.sh --batch-prefetch-auto-min-batch-size` |
| `BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO` | `1.0` | Passed to `shard.sh --batch-prefetch-auto-max-shard-cpu-ratio` |
| `BATCH_PREFETCH_AUTO_MIN_WAIT_MS` | `2.0` | Passed to `shard.sh --batch-prefetch-auto-min-wait-ms` |
| `BATCH_PREFETCH_AUTO_WARMUP_BATCHES` | `5` | Passed to `shard.sh --batch-prefetch-auto-warmup-batches` |
| `RESUME` | `false` | Set to `true` to resume from last checkpoint (skips output cleanup) |
| `BRANCH_NAME` | `p3/feat/stage-wise-coreset-selection_v2` | Git branch to clone (only used when repo setup is not skipped) |

> **Important**: `TOTAL_TOKENS` must match the actual token count of the data at `S3_INPUT_PATH`. If you change the input path to point at a different source or multiple sources, recompute this value. See [How to Pick `--total-input-tokens-estimate`](#how-to-pick---total-input-tokens-estimate-and---stage-target-scale) for details.

#### Usage Examples

#### Run against ncert on EC2 (from a local machine or CI)

```bash
S3_BUCKET=t2-datacurriculum-353 \
NUM_SHARDS=8 \
STAGES="1B" \
TOTAL_TOKENS=4523096944 \
BATCH_SIZE=80000 \
CHECKPOINT_EVERY_N_BATCHES=10 \
USED_CACHE_MAX_ENTRIES=1000000 \
USED_CACHE_STATS_EVERY=100 \
BATCH_PREFETCH_MODE=auto \
BATCH_PREFETCH_QUEUE_SIZE=1 \
BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE=50000 \
BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=1.0 \
BATCH_PREFETCH_AUTO_MIN_WAIT_MS=2.0 \
BATCH_PREFETCH_AUTO_WARMUP_BATCHES=5 \
S3_INPUT_PATH="s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/source=ncert/" \
RESUME=false \
bash experiments/3_coreset_engineering/coreset_engine_v5/commands.sh --foreground --skip-repo-setup
```

#### Run all stages with resume

```bash
S3_BUCKET=t2-datacurriculum-353 \
NUM_SHARDS=16 \
STAGES="1B 3B 8B 70B" \
TOTAL_TOKENS=4523096944 \
BATCH_SIZE=80000 \
CHECKPOINT_EVERY_N_BATCHES=5 \
USED_CACHE_MAX_ENTRIES=1000000 \
USED_CACHE_STATS_EVERY=100 \
BATCH_PREFETCH_MODE=auto \
BATCH_PREFETCH_QUEUE_SIZE=1 \
BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE=50000 \
BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=1.0 \
BATCH_PREFETCH_AUTO_MIN_WAIT_MS=2.0 \
BATCH_PREFETCH_AUTO_WARMUP_BATCHES=5 \
S3_INPUT_PATH="s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/source=ncert/" \
RESUME=true \
bash experiments/3_coreset_engineering/coreset_engine_v5/commands.sh --foreground --skip-repo-setup
```

#### Dry run to verify configuration

```bash
S3_BUCKET=t2-datacurriculum-353 \
bash experiments/3_coreset_engineering/coreset_engine_v5/commands.sh --dry-run
```

#### Manual deployment on a fresh EC2 instance (runs in background)

```bash
export S3_BUCKET=t2-datacurriculum-353
export NUM_SHARDS=8
export STAGES="1B 3B 8B 70B"
export TOTAL_TOKENS=4523096944
export BATCH_SIZE=80000
export CHECKPOINT_EVERY_N_BATCHES=10
export USED_CACHE_MAX_ENTRIES=1000000
export USED_CACHE_STATS_EVERY=100
export BATCH_PREFETCH_MODE=auto
export BATCH_PREFETCH_QUEUE_SIZE=1
export BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE=50000
export BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=1.0
export BATCH_PREFETCH_AUTO_MIN_WAIT_MS=2.0
export BATCH_PREFETCH_AUTO_WARMUP_BATCHES=5
./commands.sh
# Pipeline runs in background via nohup. Check output/logs for progress.
```

#### What `commands.sh` Does (Step by Step)

1. **System Setup** — Installs Python 3.12, git, pip (Linux only; skips on macOS)
2. **Install `uv`** — Installs the `uv` package manager if not present
3. **Repository Setup** — Clones the repo and checks out the branch (skipped with `--skip-repo-setup`)
4. **Virtual Environment** — Creates `.venv` (if needed) and runs `uv sync` to install dependencies
5. **Launch Pipeline** — Runs `shard.sh` with all parameters, either in foreground or background

### S3 Inputs (Streaming)

Streaming runs can read input data directly from S3 via `s3://...` paths.

#### Credentials / environment

The pipeline relies on the standard AWS credential resolution chain (environment variables, shared config/credentials files, or instance/role credentials).

Common environment variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` (only for temporary credentials)
- `AWS_DEFAULT_REGION` (or `AWS_REGION`)
- `AWS_PROFILE` (optional; if using `~/.aws/credentials`)

#### JSONL on S3

- Single object:

```bash
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path s3://my-bucket/datasets/chunks.jsonl --input-format jsonl \
  --checkpoint-dir output/checkpoints_s3 \
  --batch-size 80000 --total-input-tokens-estimate <TOTAL_TOKENS>
```

- Prefix containing many `*.jsonl` objects (recommended for throughput + file-level sharding):

```bash
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path s3://my-bucket/datasets/jsonl/ --input-format jsonl \
  --num-shards 8 --shard-id 0 --checkpoint-dir output/checkpoints_1B/shard000 \
  --batch-size 80000 --total-input-tokens-estimate <TOTAL_TOKENS>
```

Note: JSONL S3 streaming requires `boto3`.

#### Parquet on S3

Parquet streaming uses `pyarrow.dataset` and can read from:

- a single `s3://.../*.parquet` object path, or
- an `s3://.../prefix/` containing many parquet objects (recommended).

```bash
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml \
  --input-path s3://my-bucket/datasets/parquet/ --input-format parquet \
  --num-shards 8 --shard-id 0 --checkpoint-dir output/checkpoints_1B/shard000 \
  --batch-size 80000 --total-input-tokens-estimate <TOTAL_TOKENS>
```

For very large datasets (hundreds of GB to TB+), prefer many S3 objects under a prefix rather than a single huge file so workers can shard by file and avoid redundant reads.

### Streaming Output Merge (Recommended)

Streaming runs emit per-batch Parquet part files. Merge them into a single file per stage:

```bash
python tools/merge_selected_indices.py --coreset-root output/coresets --stages 1B 3B 8B 70B
```

Optional: export the merged Parquet to JSONL (line-delimited) for downstream systems that prefer JSON:

```bash
python tools/merge_selected_indices.py \
  --coreset-root output/coresets \
  --stages 1B 3B 8B 70B \
  --export-jsonl --overwrite-jsonl
```

### Summarize Outputs (Duplicates + Distributions)

To check for duplicate chunks (by `chunk_id`) and get percentage breakdowns by `band`, `language`, and `domain`:

```bash
# If you exported JSONL
python tools/summarize_selected_indices.py \
  --input-path output/coresets/1B/selected_indices.jsonl \
  --input-format jsonl

# If you have merged Parquet (recommended default output)
python tools/summarize_selected_indices.py \
  --input-path output/coresets/1B/selected_indices.parquet \
  --input-format parquet
```

Notes:

- For very large outputs, duplicate detection defaults to `--duplicate-mode sqlite` (disk-backed) to avoid high RAM usage.
- If your identifier column is not `chunk_id`, override `--id-fields` (first non-empty wins), e.g. `--id-fields chunk_id,uid,guid,id`.

### Validate Outputs

```bash
python tools/validate_coreset_outputs.py \
  --output-dir output/coresets \
  --curriculum config/curriculum.yaml \
  --stages 1B 3B 8B 70B \
  --format both
```

### Generate Verification Artifacts (One-Stop)

Generates a single Markdown report with:

- Per-stage validator summary
- Selected id counts
- Cross-stage overlap check (non-overlap)

```bash
python tools/generate_verification_artifacts.py \
  --curriculum config/curriculum.yaml \
  --output-dir output/coresets \
  --stages 1B 3B 8B 70B \
  --report-path output/manifests/verification_artifacts.md
```

### Expected Output

```text
coreset_engine/
├── output/
│   ├── coresets/
│   │   ├── 1B/
│   │   │   ├── selected_indices_part_shard000_batch000000.parquet
│   │   │   ├── selected_indices.parquet              # after merge_selected_indices.py
│   │   │   ├── manifest_shard000.json
│   │   │   ├── manifest.json                          # written for single-shard runs
│   │   ├── 3B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── 8B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── 70B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── .used_chunks/
│   │   │   ├── used_chunks_shard000.sqlite          # streaming non-overlap store
│   │   │   ├── used_chunks_shard001.sqlite          # (present when num-shards > 1)
│   │   │   ├── ...
│   └── manifests/
│       └── ablation_report.md
│   └── validation_reports/
│       ├── 1B_checklist.txt
│       ├── 70B_report.md
└── coreset_selection.log
```

## Architecture

### Pipeline Stages

```text
1. Data Loading & Registration
   ↓
2. Deduplication (Exact + Near)
   ↓
3. Curriculum Validation
   ↓
4. Diversity Scoring (Vectorized)
   ↓
5. Stratified Selection
   ↓
6. Protected Slice Enforcement
   ↓
7. Validation & Audit
   ↓
8. Output Generation
```

### Core Components

| Component | Location | Purpose |
| --- | --- | --- |
| Pipeline Config (source of truth) | `config/pipeline.yaml` | All runtime configuration knobs (stages, dedup, diversity, selection, IO) |
| Config Schema/Loader | `src/core/config.py` | Parses/validates YAML into typed config objects |
| Type System | `src/core/types.py` | Type-safe data structures |
| Curriculum Loader | `src/curriculum/loader.py` | Load & validate frozen curriculum |
| Exact Deduplicator | `src/dedup/deduplicator.py` | XXHash-based exact dedup |
| Near Deduplicator | `src/dedup/deduplicator.py` | SimHash/MinHash fuzzy matching |
| Diversity Scorer | `src/diversity/scorer.py` | Token rarity + coverage scoring |
| Selection Engine | `src/selection/engine.py` | Main orchestrator |
| Batched/Streaming Selection Engine | `src/selection/engine_batched.py` | 2T-safe batch selection + rolling-window stats |
| I/O Utilities | `src/io/loaders.py` | Load & save with S3/FS support |
| Used-Chunks Store | `src/io/used_chunks_store.py` | Disk-backed cross-stage non-overlap for streaming |
| Output Validator | `tools/validate_coreset_outputs.py` | Checklist/report validation (bands/domains/lang/rolling-window/targets) |
| Part Merger | `tools/merge_selected_indices.py` | Merge streaming Parquet part files into per-stage parquet |

## Configuration

The pipeline’s configuration details should be read from `config/pipeline.yaml` (that file is the source of truth).
The Python module `src/core/config.py` implements the schema/loader/validation for that YAML.

### Main Config (`config/pipeline.yaml`)

Key sections:

- **dedup**: Exact and near-duplicate detection settings
- **diversity**: Token rarity boosting and coverage weighting
- **selection**: Strategy (stratified, density-aware), protected slices
- **curriculum**: Frozen curriculum path and deterministic guarantees
- **stages**: Per-stage configurations (1B, 3B, 8B, 70B, SFT, ALIGNMENT)

Example customization:

```yaml
# Disable near-dedup for faster processing
dedup:
  enable_near_dedup: false

# Increase diversity boosting
diversity:
  rare_token_boost: 2.0    # From 1.5
  tail_token_boost: 3.0    # From 2.0
```

### Curriculum Config (`config/curriculum.yaml`)

**FROZEN** curriculum defining:

- Band definitions (B0-B5 with allowed domains)
- Stage-wise band ratios (1B: 45% B0 / 30% B1 / ..., etc.)
- Language constraints (92% English, 8% Hindi)
- Perplexity filters per band
- Rolling window constraints

⚠️ **Do not modify** once frozen. Changes require curriculum team approval.

### Curriculum Schema Compatibility (Recent)

The engine supports the updated curriculum schema for language policy gating:

- `language_and_context.language_policy.secondary_languages` may specify an `earliest_stage`.
- Streaming selection will only allow those secondary languages at/after `earliest_stage`.
- `explicitly_excluded` languages are always filtered.

This matches the early language filtering used by the batched selection engine.

## Streaming / 2T-Scale Notes (Recent)

### Cross-stage Non-overlap (Streaming)

Streaming runs enforce disjoint coresets across stages using a disk-backed membership store:

- Store location: `output/coresets/.used_chunks/used_chunks_shard###.sqlite`
- Behavior: each stage filters out previously-selected chunk_ids before selection

### Rolling-window Anti-spike Constraints (End-to-end)

If the curriculum config enables rolling-window constraints, the streaming pipeline:

- enforces them during selection
- writes `rolling_window_stats` into the stage manifest
- validator requires these stats when rolling-window is configured

### Availability-aware Manifests + Validation (Recent)

When running with strict non-overlap on a limited dataset, later-stage targets or band ratios can become infeasible.
Streaming manifests may include `availability_stats` describing the *eligible unused pool* observed (post non-overlap + stage gating).

The validator uses this to label certain target/band failures as **availability-limited** (informational) when the manifest proves the constraint is not achievable with remaining data.

## Usage Example's

### Example 1: Basic Selection for 70B Stage

```python
from src.core.config import PipelineConfig
from src.curriculum.loader import CurriculumLoader
from src.selection.engine import SelectionEngine
from src.io.loaders import ChunkLoader, CoresetWriter

# Load configuration
config = PipelineConfig.load_from_file("config/pipeline.yaml")
curriculum = CurriculumLoader("config/curriculum.yaml")
curriculum.load()

# Load chunks
loader = ChunkLoader(base_path="/data/datasets")
all_chunks = loader.load_all_chunks()

# Initialize engine
engine = SelectionEngine(config, curriculum)
engine.register_chunks([(cid, meta, None) for cid, meta in all_chunks.items()])

# Run selection
selected_chunks, stats = engine.select_for_stage(
    all_chunks=all_chunks,
    stage_name="70B",
)

# Save outputs
writer = CoresetWriter("/output/coresets")
writer.save_selected_indices("70B", selected_chunks, metadata_dict)
```

### Example 2: Ablation Study

```python
from src.core.config import PipelineConfig

# Load baseline config
config = PipelineConfig.load_from_file("config/pipeline.yaml")

# Ablation 1: No near-dedup
config.dedup.enable_near_dedup = False
config.save_to_file("config/ablation_no_neardup.yaml", format="yaml")

# Ablation 2: No diversity boosting
config.diversity.rare_token_boost = 1.0
config.diversity.tail_token_boost = 1.0
config.save_to_file("config/ablation_no_diversity.yaml", format="yaml")

# Run with ablated configs
# python coreset_builder.py --config config/ablation_no_neardup.yaml ...
```

### Example 3: Custom Protected Slices

```python
from src.core.types import ProtectedSliceRule

protected_slices = [
    ProtectedSliceRule("B5", 0.98, "Critical for emergent abilities"),
    ProtectedSliceRule("B4", 0.95, "Advanced reasoning"),
    ProtectedSliceRule("code", 0.93, "Programming capability"),
    ProtectedSliceRule("agentic", 0.92, "Agent grounding"),
    ProtectedSliceRule("indic", 0.80, "Multilingual support"),
]

selected_chunks, stats = engine.select_for_stage(
    all_chunks=all_chunks,
    stage_name="70B",
    protected_slices=protected_slices,  # Override defaults
)
```

## Key Metrics & Monitoring

### Expected Compression Results

| Stage | Input Tokens | Output Tokens | Ratio | Chunks |
| --- | --- | --- | --- | --- |
| 1B | 400B | 20B | 20x | ~5M |
| 3B | 800B | 40B | 20x | ~10M |
| 8B | 2T | 100B | 20x | ~25M |
| 70B | 2T | 240B | 8.3x | ~60M |

### Coverage Validation

After selection, check:

```python
from src.core.types import BandDistribution

band_dist = stats['band_distribution']
print(f"B0: {band_dist.B0:.2%}")  # Should be ~5% for 70B
print(f"B4: {band_dist.B4:.2%}")  # Should be ~25% for 70B
print(f"B5: {band_dist.B5:.2%}")  # Should be ~15% for 70B
```

### Protected Slice Preservation

```python
manifest = CoresetManifest(...)
preserved = manifest.protected_slices_preserved

assert preserved.B5_preservation_ratio >= 0.95, "B5 not preserved!"
assert preserved.code_preservation_ratio >= 0.90, "Code not preserved!"
```

## Troubleshooting

### Issue: "Curriculum not frozen"

**Error**:

```text
Curriculum validation failed: Curriculum is not frozen
```

**Solution**:

- Ensure curriculum status is "FROZEN" in `config/curriculum.yaml`
- Contact curriculum team if status is "DRAFT"

### Issue: "Rolling window violation"

**Error**:

```text
HARD_REJECT: Rolling window constraint violated
```

**Solution**:

- Reduce `diversity.rare_token_boost` or `tail_token_boost`
- Add `smooth_selection_via_rolling_window()` post-processing
- Increase `rolling_window.window_tokens` to allow more variance

### Issue: "Protected slices under-preserved"

**Error**:

```text
B5 preservation ratio: 0.85 < 0.95 (minimum required)
```

**Solution**:

1. Increase `selection.protected_preservation_override["B5"]` to be reachable
2. Run selection without other constraints (debug mode)
3. Check if enough B5 chunks exist in source data

### Issue: Memory exhaustion on large datasets

**Solution**:

```yaml
# Reduce parallel loaders
io:
  num_parallel_loaders: 8  # Down from 32

# Prefer the streaming builder (default) with checkpointing
# and reduce batch size if needed.
```

If you see `--input-path is required unless --legacy is set`, you are running the streaming builder without an input dataset path.

## Integration Points

### Upstream: Accepting Input from Teams

```json
{
  "team_1": "Provide clean dataset + metadata (parquet/jsonl)",
  "team_2": "Provide FROZEN curriculum.yaml",
  "team_3": "Provide chunk indices + metadata",
  "team_4": "Provide difficulty band assignments",
  "team_5": "Provide dedup signatures + quality scores"
}
```

### Downstream: Providing Output to Teams

```json
{
  "training_team": {
    "format": "Parquet index files per stage",
    "guarantee": "Non-overlapping chunks",
    "reproducibility": "Deterministic given seed + curriculum"
  },
  "benchmarking_team": {
    "format": "Ablation report + coverage audit",
    "metrics": ["compression_ratio", "band_coverage", "convergence_speed"]
  },
  "synthetic_team": {
    "format": "Available band/domain quotas",
    "max_injection": "5-10% per stage"
  }
}
```

## Performance Benchmarks

### Runtime (Measured on 64-node GPU cluster)

| Stage | Input Tokens | Dedup | Scoring | Selection | Total |
| --- | --- | --- | --- | --- | --- |
| 1B | 400B | 15m | 10m | 5m | 30m |
| 3B | 800B | 20m | 15m | 8m | 43m |
| 8B | 2T | 45m | 35m | 15m | 95m |
| 70B | 2T | 45m | 35m | 15m | 95m |

**Total pipeline runtime**: ~4 hours (all stages in parallel: ~2 hours)

### Memory Usage

- Metadata cache: ~50GB (for 2T tokens)
- Dedup structures: ~30GB (exact + near-dedup hashes)
- Scoring vectors: ~20GB (diversity scores)
- **Total**: ~100GB (fits in typical cluster node)

## Reproducibility & Versioning

### Reproducibility Guarantee

Every coreset output includes:

```json
{
  "deterministic": true,
  "seed": 42,
  "config_hash": "sha256(...)",
  "curriculum_hash": "sha256(...)",
  "algorithm_version": "1.0.0",
  "created_at": "2026-02-03T10:30:00Z"
}
```

To reproduce exactly:

```bash
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml
```

Same outputs will be produced (bit-for-bit identical indices, same seed).

### Versioning Strategy

- **Pipeline Version**: `1.0.0` (algorithm changes bump minor/major)
- **Config Version**: Git commit hash (`abc123def...`)
- **Curriculum Version**: Git commit hash (frozen checkpoints)
- **Data Version**: Dataset timestamp + version ID

## Documentation

- [Quickstart (2T-scale)](docs/QUICKSTART.md)
- [Output Format Guide](docs/OUTPUT_FORMAT_GUIDE.md)
- [Curriculum Adherence Notes](docs/CODE_CHANGES_CURRICULUM_ADHERENCE.md)
- [Pipeline Fix Summary](docs/PIPELINE_FIX_SUMMARY.md)
- [Performance Fix Summary](docs/PERFORMANCE_FIX_SUMMARY.md)

## Support & Issues

**Bug Reports**: Create issue with:

- Config file (sanitized)
- Curriculum file
- Error logs
- Hardware specs (CPU/GPU, RAM)

**Feature Requests**:

- Propose as issue with use case
- Include performance requirements
- Discuss in team meeting before implementation

---

**Last Updated**: 2026-02-22  
**Maintainer**: Coreset Selection Team  
**License**: Internal Use Only
