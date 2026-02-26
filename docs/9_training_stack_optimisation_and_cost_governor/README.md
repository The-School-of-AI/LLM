# P9 — Training Stack Optimisation & Cost Governor

> Workstream 9 owns the end-to-end training infrastructure: DeepSpeed-based training pipelines, high-performance data loading, real-time observability, compute/cost estimation, and automatic halt systems for the ERA V4 multi-stage LLM training plan (1B → 3B → 8B → 70B MoE).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Pipelines Overview](#pipelines-overview)
3. [How to Run Each Pipeline Independently](#how-to-run-each-pipeline-independently)
   - [DeepSpeed Training Pipeline](#1-deepspeed-training-pipeline)
   - [Data Loader (SPDL)](#2-data-loader-spdl)
   - [FLOPS & Cost Calculator](#3-flops--cost-calculator)
   - [Monitoring Dashboard](#4-monitoring-dashboard)
4. [Configuration Reference](#configuration-reference)
5. [Artifacts & Outputs](#artifacts--outputs)
6. [Output Storage Locations](#output-storage-locations)
7. [Additional Research Details](#additional-research-details)
8. [Git Workflow](#git-workflow)

---

## Repository Layout

All P9 code lives under `experiments/9_training_stack_optimisation_and_cost_governor/`:

```
experiments/9_training_stack_optimisation_and_cost_governor/
├── training/
│   └── deepspeed_template/         # Core training pipeline (DeepSpeed ZeRO 2/3)
│       ├── main.py                 # Entry point
│       ├── config.yaml             # Active training config
│       ├── config.example.yaml     # Documented example config
│       ├── deepspeed/              # ZeRO JSON configs (zero-1, zero-2-*, zero-3)
│       ├── src/                    # Training source code
│       │   ├── train.py            # Training, eval, generation loops
│       │   ├── checkpoint.py       # Checkpoint mgmt + S3 integration
│       │   ├── data.py             # HuggingFace data loading + tokenization
│       │   ├── prefetch_loader.py  # Async GPU prefetch data loader
│       │   ├── shard_tracker.py    # Shard consumption tracking
│       │   ├── models/             # Model architectures (1B/3B/8B/70B recurrence)
│       │   ├── kernels/            # Triton fused kernels + PyTorch fallbacks
│       │   ├── growth/             # Weight growth/transfer utilities
│       │   └── tokenizer/          # Tokenizer assets
│       ├── aws/                    # S3 configuration utilities
│       ├── scripts/                # S3 verification and test scripts
│       ├── test/                   # CPU + GPU test suites
│       └── docs/                   # Data pipeline strategy, benchmarks
├── data_loader/                    # SPDL high-perf data loading pipeline
│   ├── spdl_dataloader.py          # Main SPDL implementation (.bin/.idx)
│   ├── dataloader.py               # Unified CLI for the data loader
│   ├── common.py                   # Shared utilities
│   ├── configuration_P4.yaml       # Config for Tesla P4 GPUs
│   ├── configuration_P5.yaml       # Config for A100/P5 GPUs
│   ├── run_spdl_production.sh      # Production launch script
│   ├── run_test.sh                 # Automated test runner
│   └── test_result.md              # Latest test metrics
├── FLOPS-Calculation/              # Compute & cost governor
│   ├── compute.py                  # FLOPs/cost estimator
│   ├── dashboard.py                # Reversibility analysis dashboard
│   ├── config.json                 # Default/starter config
│   ├── configs/                    # Preset configurations
│   │   ├── 1b_presets/             # 8 variants: base, GSA, MLA, MHC, MTP, YaRN, full
│   │   ├── moe_team8/             # Stage 1–4 + combined all-stages config
│   │   └── reference_all_knobs.json
│   └── reversibility_analysis.md   # Memory impact study
├── monitoring/                     # Real-time training observability dashboard
│   ├── dashboard_server.py         # Flask backend (ClickHouse-backed)
│   ├── dashboard/                  # Frontend (HTML/CSS/JS)
│   ├── start.sh                    # Local launch script
│   ├── monitoring.service          # systemd unit file (EC2 prod)
│   ├── monitoring_nginx.conf       # Nginx reverse-proxy config
│   └── gunicorn.conf.py            # Gunicorn production config
├── halt_system/                    # Automatic training halt (planned)
├── deliverables/                   # Halt checkpoints & incident logs
│   ├── halt_system/checkpoints/
│   └── halt_system/incidents/
├── configs/                        # (reserved for per-stage configs)
├── scripts/                        # Automation scripts
├── docs/                           # Extra documentation
└── P9_Git_Workflow.md              # Team branching & PR conventions
```

Companion directories:
- `docs/9_training_stack_optimisation_and_cost_governor/` — this README (authoritative docs).
- `tests/9_training_stack_optimisation_and_cost_governor/` — additional top-level tests.

---

## Pipelines Overview

| # | Pipeline | Purpose | Key Tech |
|---|----------|---------|----------|
| 1 | **DeepSpeed Training** | Pre-training loop (dense & MoE models) with ZeRO 2/3, reversible training, S3 checkpointing | DeepSpeed, PyTorch, Triton kernels |
| 2 | **SPDL Data Loader** | High-throughput streaming of pre-tokenized `.bin/.idx` shards from S3 → NVMe → GPU | Meta SPDL, PyArrow, mmap |
| 3 | **FLOPS & Cost Calculator** | Estimate training time, FLOPs, GPU memory, and cloud cost for any model config | Python (standalone) |
| 4 | **Monitoring Dashboard** | Real-time charts, event logs, run comparison, and summary KPIs | Flask, ClickHouse, Chart.js |

---

## How to Run Each Pipeline Independently

### 1. DeepSpeed Training Pipeline

**Location:** `experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/`

#### Prerequisites
- NVIDIA GPU(s) with CUDA 11.8+
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

#### Setup

```bash
cd experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template

# Install dependencies
pip install uv          # if not already installed
uv sync                 # installs everything from pyproject.toml + uv.lock

# Copy & edit configuration
cp config.example.yaml config.yaml
# Edit config.yaml (dataset, model, DeepSpeed config, checkpointing, S3, etc.)
```

#### Run

```bash
# Multi-GPU (recommended)
deepspeed --num_gpus=4 main.py

# Single GPU (debug/test)
python main.py

# Custom config file
deepspeed main.py --config config/my_experiment.yaml
```

#### Specialised Configs

| Config File | Use Case |
|-------------|----------|
| `deepspeed/zero-2-dense.json` | Dense model, ZeRO Stage 2 (default) |
| `deepspeed/zero-2-dense-reversible.json` | Dense model, reversible training |
| `deepspeed/zero-2-dense-4k-throughput.json` | 4K context throughput run |
| `deepspeed/zero-2-dense-8k-throughput.json` | 8K context throughput run |
| `deepspeed/zero-2-dense-oom-fixed.json` | OOM-safe settings |
| `deepspeed/zero-2-moe.json` | MoE model, ZeRO Stage 2 |
| `deepspeed/zero-2-moe-reversibile.json` | MoE model, reversible training |
| `deepspeed/zero-3.json` | Full ZeRO Stage 3 (largest models) |

#### Running Tests

```bash
# CPU-only (no GPU required)
pytest test/test_training_cpu.py -v

# GPU tests (requires CUDA)
pytest test/test_training_gpu.py -v

# All tests
pytest test/ -v
```

---

### 2. Data Loader (SPDL)

**Location:** `experiments/9_training_stack_optimisation_and_cost_governor/data_loader/`

#### Prerequisites
- Python 3.10+ (tested on 3.11)
- Pre-tokenized `.bin/.idx` shard files

#### Setup

```bash
cd experiments/9_training_stack_optimisation_and_cost_governor/data_loader

# Option A: Automated setup
source setup_venv.sh

# Option B: Manual
pip install uv
uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip sync requirements.uv.txt
```

#### Run

```bash
# Standalone test
python dataloader.py --token-folder /path/to/shards --batches 10 --log-level INFO

# Production run with config
bash run_spdl_production.sh <CONFIG_FILE> <TOKEN_FOLDER>
# e.g.  bash run_spdl_production.sh configuration_P5.yaml /mnt/nvme/tokens

# Test suite
bash run_test.sh
```

#### Hardware-Specific Configs

| Config | Target Hardware | Batch Size | Seq Len |
|--------|----------------|-----------|---------|
| `configuration_P4.yaml` | Tesla P4 (8 GB) | 512 | 2048 |
| `configuration_P5.yaml` | A100 / P5 (40+ GB) | 2048 | 4096 |

#### Integration with DeepSpeed Training

The SPDL loader can be used inside the DeepSpeed training pipeline by setting in `config.yaml`:

```yaml
data:
  use_dataloader: true
  shard_dir: "/mnt/nvme/tokens"
```

---

### 3. FLOPS & Cost Calculator

**Location:** `experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/`

#### Prerequisites
- Python 3.10+ (no GPU required)

#### Run

```bash
cd experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation

# Default config
python3 compute.py

# With a preset
python3 compute.py --config configs/moe_team8/moe_team8_all_stages.json

# 1B preset
python3 compute.py --config configs/1b_presets/1b_deepseek_gsa.json
```

#### Available Presets

**1B Presets** (`configs/1b_presets/`):

| File | Variant |
|------|---------|
| `1b_base.json` | Baseline GQA + YaRN |
| `1b_deepseek_gsa.json` | DeepSeek GSA (recommended sparse attention) |
| `1b_gsa.json` | Original GSA variant |
| `1b_deepseek_mla.json` | DeepSeek MLA (KV compression) |
| `1b_mhc.json` | Manifold hyper-connections |
| `1b_mtp.json` | Multi-token prediction |
| `1b_yarn.json` | Extended 32K context via YaRN |
| `1b_full.json` | All features combined |

**Team-8 MoE Configs** (`configs/moe_team8/`):

| File | Stage |
|------|-------|
| `stage1_1b_dense.json` | Stage 1 — 1B dense |
| `stage2_3b_moe.json` | Stage 2 — 3B MoE |
| `stage3_8b_moe.json` | Stage 3 — 8B MoE |
| `stage4_70b_moe.json` | Stage 4 — 70B MoE |
| `moe_team8_all_stages.json` | All four stages combined |

#### Reversibility Analysis Dashboard

```bash
python3 dashboard.py
# or
python3 generate_reversibility_report.py
```

Output: `reversibility_analysis.md` — compares standard vs. reversible training memory at different micro-batch sizes for the 70B model on 8× H200 GPUs.

---

### 4. Monitoring Dashboard

**Location:** `experiments/9_training_stack_optimisation_and_cost_governor/monitoring/`

#### Prerequisites
- Python 3.10+
- ClickHouse instance with training metrics

#### Local Setup

```bash
cd experiments/9_training_stack_optimisation_and_cost_governor/monitoring

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set credentials
cp .env.example .env
# Edit .env: CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DB

# Start
./start.sh
# Open http://localhost:5050
```

#### EC2 Production Deployment

1. Launch **Ubuntu 24.04 LTS ARM64** on **t4g.small**
2. Install dependencies, clone repo, set up `.env`
3. Copy `monitoring.service` → `/etc/systemd/system/`
4. Copy `monitoring_nginx.conf` → `/etc/nginx/sites-available/`
5. Enable and start the service:
   ```bash
   sudo systemctl enable monitoring
   sudo systemctl start monitoring
   sudo systemctl enable nginx && sudo systemctl start nginx
   ```
6. Access at `http://<ec2-public-ip>` (port 80)

Full step-by-step instructions are in `monitoring/Readme.md`.

#### Key API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | List all training runs |
| `GET /api/runs/<run_id>/metrics` | Metrics grouped by category |
| `GET /api/runs/<run_id>/data?metrics=a,b` | Time-series data |
| `GET /api/runs/<run_id>/events` | Training event log |
| `GET /api/health` | ClickHouse status check |

---

## Configuration Reference

### Training Config (`config.yaml` / `config.example.yaml`)

| Section | Key Settings |
|---------|-------------|
| `data` | `dataset_name`, `max_length`, `use_dataloader`, `shard_dir`, `prefetch_to_gpu`, `prefetch_depth` |
| `training` | `num_epochs`, `max_train_steps`, `log_interval`, `seed`, `enable_system_metrics` |
| `deepspeed` | `config_path` (points to a ZeRO JSON) |
| `model` | `tokenizer_name`, `embedding_type` |
| `checkpoint` | `output_dir`, `checkpoint_interval`, `resume_from_checkpoint`, `keep_last_n_checkpoints` |
| `s3` | `enabled`, `bucket`, `prefix`, `region`, `cleanup_after_upload` |
| `generation` | `test_generation`, `generation_prompt` |

### DeepSpeed ZeRO Configs (`deepspeed/*.json`)

11 config variants covering ZeRO Stage 1/2/3, dense/MoE, reversible, OOM-fixed, and throughput profiles. See the table under [Specialised Configs](#specialised-configs) above.

### FLOPS Calculator Configs

Located in `FLOPS-Calculation/configs/`. Supports both flat and nested JSON formats. Key sections: `hardware` (GPUs, pricing, MFU, ZeRO stage, parallelism), `architecture` (hidden size, layers, attention type, MoE), `training` (activation checkpointing, precision). Full reference in `FLOPS-Calculation/README.md`.

### Data Loader Configs

| File | Target |
|------|--------|
| `configuration_P4.yaml` | Tesla P4 (batch 512, seq 2048) |
| `configuration_P5.yaml` | A100/P5 (batch 2048, seq 4096) |

---

## Artifacts & Outputs

### Training Pipeline Artifacts

| Artifact | Description | Format |
|----------|-------------|--------|
| **Checkpoints** | Model weights, optimizer states, training state | DeepSpeed checkpoint dirs (`epoch*_step*`) |
| **S3 Checkpoints** | Same, uploaded asynchronously to S3 | `s3://<bucket>/<prefix>/step_*/` |
| **Training Logs** | Per-step loss, perplexity, learning rate, timings | Console stdout + optional system metrics |
| **Generated Text** | Sample model outputs after training | Console stdout |
| **Shard Manifest** | Tracks consumed data shards for resumability | `consumed_shards.json` |

### FLOPS Calculator Artifacts

| Artifact | Description |
|----------|-------------|
| `last_run.txt` | Baseline FLOPs/cost/memory estimate |
| `last_run_growth.txt` | Growth/expansion estimate |
| `reversibility_analysis.md` | Standard vs. reversible memory comparison |

### Monitoring Artifacts

| Artifact | Description |
|----------|-------------|
| Dashboard UI | Real-time charts, run comparisons, event logs |
| API JSON | RESTful endpoints returning run/metric/event data |

### Data Loader Artifacts

| Artifact | Description |
|----------|-------------|
| `test_result.md` | Benchmark metrics (throughput, batch processing time) |
| Console logs | Per-batch processing stats, hardware info |

---

## Output Storage Locations

| Output | Default Location | Configurable Via |
|--------|-----------------|------------------|
| Training checkpoints (local) | `./checkpoints/` | `checkpoint.output_dir` in `config.yaml` |
| Training checkpoints (S3) | `s3://<bucket>/<prefix>/` | `s3.bucket`, `s3.prefix` in `config.yaml` |
| FLOPS estimate snapshots | `FLOPS-Calculation/last_run*.txt` | N/A (overwritten each run) |
| Reversibility report | `FLOPS-Calculation/reversibility_analysis.md` | N/A |
| Data loader test results | `data_loader/test_result.md` | N/A |
| Monitoring dashboard | `http://localhost:5050` or `http://<ec2-ip>` | `.env` file for ClickHouse connection |
| Training logs | Console (stdout/stderr) | Redirect via shell or logging config |
| NVMe data cache (AWS) | `/mnt/nvme/` | RAID setup scripts in `data_loader/Readme.md` |

---

## Additional Research Details

### Model Architectures (src/models/)

P9 implements custom recurrence-based models at four scales:

| Model | File | Params | Architecture |
|-------|------|--------|-------------|
| 1B Dense | `recurrence_model_1b.py` | ~1B | Hybrid DeltaNet + GSA, MHC, MTP |
| 3B MoE | `recurrence_model_3b.py` | ~3B | MoE + recurrence + GSA |
| 8B MoE | `recurrence_model_8b.py` | ~8B | MoE + recurrence + GSA |
| 70B MoE | `recurrence_model_70b.py` | ~70B | Full MoE with all features |

### Fused Kernels (src/kernels/)

Custom Triton kernels with automatic PyTorch fallback:

| Kernel | Purpose | Speedup |
|--------|---------|---------|
| `triton_sparse_attn.py` | GSA sparse attention — O(T·k) vs O(T²) | Significant for long contexts |
| `triton_indexer.py` | Gated lightning indexer (shared-K design) | Reduces memory from ~6 GB to ~134 MB at T=4096 |
| `triton_sinkhorn.py` | Fused Sinkhorn-Knopp for mHC routing | 40 kernel launches → 1 |
| `triton_rmsnorm.py` | Fused RMSNorm + residual | 50% less memory bandwidth |
| `fla_deltanet.py` | flash-linear-attention DeltaNet wrapper | Fused recurrence kernel |

All kernels are pure functions, safe for reversible training (midpoint integrator).

### Reversible Training

Based on [Gal et al., arXiv:2512.02056v2](https://arxiv.org/abs/2512.02056). Key results on 70B MoE with 8× H200:

| Precision | Standard Max Batch | Reversible Max Batch | Improvement |
|-----------|-------------------|---------------------|-------------|
| BF16 | 8 | 512 | **64×** |
| FP8 | 16 | 1024 | **64×** |

Activation memory drops from O(layers) to O(1) with ~33% FLOPs overhead.

### Data Pipeline Strategy

Two-phase S3 → NVMe → GPU pipeline for p5en.48xlarge (8× H200):
1. **Phase 1 (blocking):** Pre-stage initial shards from S3 to NVMe RAID-0 (30 TB) during model init
2. **Phase 2 (background):** Background thread pool keeps downloading ahead of training

See `training/deepspeed_template/docs/Data_load_strategy.md` for the full design.

### SPDL vs Standard DataLoader Benchmark

Tested on 2 parallel processes, 8 shards, block_size=4096, batch_size=8:

| Metric | Standard PyTorch | SPDL | Improvement |
|--------|-----------------|------|-------------|
| Aggregate throughput | 9.8M tok/s | 10.0M tok/s | 1.02× |
| DataLoader wait ratio | 16.8% | 13.7% | 18% lower |

Full results in `training/deepspeed_template/docs/benchmark_results.txt`.

---

## Git Workflow

See `experiments/9_training_stack_optimisation_and_cost_governor/P9_Git_Workflow.md` for:
- Branch naming conventions (`p09/feat/<name>`)
- Integration branch pattern for multi-person features
- Daily rebase routine (morning + evening)
- Pre-commit hooks (Black, isort, Ruff, secrets scanning)
- PR flow (sub-branch → integration → main)
