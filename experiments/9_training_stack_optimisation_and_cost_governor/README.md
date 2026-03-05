# P9 — Training Stack Optimisation & Cost Governor

End-to-end training infrastructure for the ERA V4 multi-stage LLM plan (1B → 3B → 8B → 70B MoE).

> **Full documentation:** [`docs/9_training_stack_optimisation_and_cost_governor/README.md`](../../docs/9_training_stack_optimisation_and_cost_governor/README.md)

---

## Quick Links

| Component | Path | What It Does |
|-----------|------|-------------|
| **DeepSpeed Training** | [`training/deepspeed_template/`](training/deepspeed_template/) | Core pre-training loop (dense & MoE, ZeRO 2/3, reversible training, S3 checkpointing) |
| **Data Loader (SPDL)** | [`data_loader/`](data_loader/) | High-throughput streaming of `.bin/.idx` token shards (S3 → NVMe → GPU) |
| **FLOPS & Cost Calculator** | [`FLOPS-Calculation/`](FLOPS-Calculation/) | Estimate training time, FLOPs, GPU memory, and cloud cost |
| **Monitoring Dashboard** | [`monitoring/`](monitoring/) | Real-time training observability (Flask + ClickHouse + Chart.js) |
| **Halt System** | [`halt_system/`](halt_system/) | Automatic training halt triggers (planned) |
| **Deliverables** | [`deliverables/`](deliverables/) | Halt checkpoints & incident records |
| **Git Workflow** | [`P9_Git_Workflow.md`](P9_Git_Workflow.md) | Branching conventions, PR flow, daily rebase routine |

---

## Quick Start

### 1. DeepSpeed Training

```bash
cd training/deepspeed_template
pip install uv && uv sync
cp config.example.yaml config.yaml   # edit as needed
deepspeed --num_gpus=4 main.py       # multi-GPU
```

### 2. Data Loader (SPDL)

```bash
cd data_loader
source setup_venv.sh
bash run_spdl_production.sh configuration_P5.yaml /path/to/tokens
```

### 3. FLOPS & Cost Estimation

```bash
cd FLOPS-Calculation
python3 compute.py --config configs/moe_team8/moe_team8_all_stages.json
```

### 4. Monitoring Dashboard

```bash
cd monitoring
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ClickHouse credentials
./start.sh             # http://localhost:5050
```

---

## Key Configs

| Config | Location | Purpose |
|--------|----------|---------|
| `config.yaml` | `training/deepspeed_template/` | Training hyperparams, data, checkpointing, S3 |
| `deepspeed/zero-*.json` | `training/deepspeed_template/deepspeed/` | ZeRO stage 1/2/3, dense/MoE/reversible variants |
| `configuration_P4.yaml` | `data_loader/` | SPDL settings for Tesla P4 |
| `configuration_P5.yaml` | `data_loader/` | SPDL settings for A100/P5 |
| `configs/1b_presets/*.json` | `FLOPS-Calculation/configs/` | 1B model FLOPs configs (8 variants) |
| `configs/moe_team8/*.json` | `FLOPS-Calculation/configs/` | Multi-stage MoE FLOPs configs |
| `.env` | `monitoring/` | ClickHouse connection credentials |

## Key Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Checkpoints (local) | `training/deepspeed_template/checkpoints/` | DeepSpeed model + optimizer state |
| Checkpoints (S3) | `s3://<bucket>/<prefix>/step_*/` | Background-uploaded snapshots |
| Shard manifest | `consumed_shards.json` | Tracks consumed data shards for resume |
| FLOPS estimates | `FLOPS-Calculation/last_run*.txt` | Training time & cost estimates |
| Reversibility report | `FLOPS-Calculation/reversibility_analysis.md` | Memory comparison (standard vs reversible) |
| Benchmark results | `training/deepspeed_template/docs/benchmark_results.txt` | SPDL vs standard DataLoader |
| Data loader metrics | `data_loader/test_result.md` | Throughput & latency benchmarks |

---

## Tests

```bash
# Training (CPU only — no GPU required)
cd training/deepspeed_template && pytest test/test_training_cpu.py -v

# Training (GPU required)
cd training/deepspeed_template && pytest test/ -v

# Data loader
cd data_loader && bash run_test.sh
```

---

_See the [Git Workflow](P9_Git_Workflow.md) document for branching conventions, PR flow, and daily rebase routines._