# 03_evaluation — Evaluation Scripts

Post-training evaluation: quantization validation, benchmark evaluation, IDFT smoke test, and pre-training checkpoint selection.

## Files

| File | Purpose |
|------|---------|
| `validate_quantization.py` | Validates end-to-end quantization support (Issue #333). Run before training to confirm hardware setup. |
| `evaluate_smoke_test.py` | Runs lm-evaluation-harness on a checkpoint. Used in the IDFT smoke test. |
| `phi_diagnostic.py` | Computes phi distribution on base model outputs. Go/no-go gate for IDFT. |
| `run_idft_smoke_test.py` | Full IDFT A/B smoke test orchestrator (phases 0-4). |
| `select_pretrain_checkpoint.py` | Ranks pre-training checkpoints by perplexity + benchmarks to select the best SFT base. |

## Imports

Some scripts in this directory import helpers from `../02_sft_training/` and add that path at runtime. `select_pretrain_checkpoint.py` is self-contained. No manual `PYTHONPATH` changes should be needed for the commands below.

## Usage

```bash
cd 03_evaluation

# Validate quantization before training
python validate_quantization.py --quick
python validate_quantization.py --config ../02_sft_training/default_config.yaml

# Run full IDFT smoke test
python run_idft_smoke_test.py

# Evaluate a specific checkpoint
python evaluate_smoke_test.py \
    --checkpoint_dir /path/to/checkpoint \
    --label sft \
    --output_json results_sft.json \
    --use_peft \
    --base_model Qwen/Qwen2.5-7B

# Run phi diagnostic on base model (IDFT go/no-go)
python phi_diagnostic.py --model_name Qwen/Qwen2.5-7B --max_batches 100
```

---

## Checkpoint Selection (`select_pretrain_checkpoint.py`)

Used to pick the best pre-training checkpoint to start SFT from. Run this once pre-training has saved candidate checkpoints, before kicking off any SFT job.

### What it measures

| Signal | Method | Role |
|--------|--------|------|
| **Validation perplexity** | Forward pass on held-out tokens (WikiText-103 or custom JSONL) | Primary — lower is better |
| **HellaSwag** | 10-shot, 10K examples, ~30 min at 70B | Secondary — higher is better |
| **ARC-Challenge** | 25-shot, 1.2K examples, ~5 min at 70B | Secondary — higher is better |
| **WinoGrande** | 5-shot, 1.3K examples, ~5 min at 70B | Secondary — higher is better |
| **LAMBADA** | 0-shot next-word prediction, 5K examples, ~15 min at 70B | Secondary — higher is better |
| **Training stability** | Scans loss log CSV for spikes in the 500 steps before each checkpoint | Filter — spikes push to bottom |

Checkpoints are ranked by a composite of perplexity rank + benchmark rank. Unstable checkpoints (loss spikes) are always sorted to the bottom regardless of score.

**Why not MMLU-Pro or GSM8K?** Both require instruction following — raw PT models score near-random on them (~10-15% and ~1-2% respectively, as confirmed by our p17 eval on Gemma-3-1B PT). They're useless for ranking PT checkpoints. Use them post-SFT via `evaluate_smoke_test.py` instead.

### Recommended environment

Run this on the **GPU training instance**, not your local machine.

- **GPU**: At minimum 1x A100 80GB or equivalent. For our 70B MoE, you need enough VRAM to load the model in bfloat16 with `device_map="auto"` — this means multi-GPU is fine.
- **CPU-only fallback**: Works but will be very slow for 70B (hours per checkpoint). Only use CPU for small proxy models (e.g. 7B) during testing.
- **Storage**: Checkpoints should be local (NVMe) or on a fast NFS mount. Avoid evaluating directly from S3 — download first with `aws s3 sync`.

```
# Recommended instance type: same training node, or a dedicated eval node
# GPU memory required: ~140GB for 70B bfloat16 (2x A100 80GB or 4x A40)
# Disk: ~140GB per checkpoint (70B bfloat16) + ~1GB for tokenizer
```

### Installation

```bash
pip install transformers accelerate torch datasets

# Only needed if running benchmarks (--skip_benchmarks to skip):
pip install lm-eval
```

The tokenizer is already present at `../FINAL_TOKENIZER/` — no download needed.
If you omit `--val_data`, the script falls back to WikiText-103 through `datasets`, so that package must be installed and the machine needs the dataset cached locally or network access to fetch it.
Custom tokenizers must define `eos_token_id`.

### Usage

**Basic — perplexity only (fastest, ~20 min per 70B checkpoint on 2xA100):**
```powershell
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_80000 C:\ckpts\step_90000 C:\ckpts\step_100000 `
    --skip_benchmarks `
    --output_json .\checkpoint_ranking.json
```

**Full — perplexity + benchmarks (~2-3h per checkpoint):**
```powershell
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_80000 C:\ckpts\step_90000 C:\ckpts\step_100000 `
    --output_json .\checkpoint_ranking.json
```
If `lm-eval` is not installed, the script logs a warning and continues with perplexity-only ranking. Treat missing benchmark columns in the output table/JSON as a failed prerequisite, not as a valid full run.

**With training stability check (recommended if you have the loss log):**
```powershell
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_80000 C:\ckpts\step_90000 C:\ckpts\step_100000 `
    --loss_log C:\logs\train_loss.csv `
    --output_json .\checkpoint_ranking.json
```
The loss log must be a CSV with columns exactly `step` and `loss`. If the step cannot be inferred from the checkpoint directory name, the script marks stability as `UNKNOWN` instead of penalizing the checkpoint.

**Custom validation data instead of WikiText-103:**
```powershell
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_100000 `
    --val_data C:\data\val.jsonl `
    --output_json .\checkpoint_ranking.json
```
The JSONL must have a `"text"` field per line. Ideally use a held-out sample from your own pre-training distribution (B1/B2 mix), not just WikiText — this gives a more accurate perplexity for your specific model.

**Custom tokenizer (override default):**
```powershell
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_100000 `
    --tokenizer_path C:\path\to\tokenizer `
    --output_json .\checkpoint_ranking.json
```

### ZeRO / DeepSpeed checkpoints

The script expects **HuggingFace format** checkpoints (a directory with `config.json` + weight files). If pre-training saved ZeRO sharded checkpoints, convert them first:

```powershell
# Run from the checkpoint directory
python zero_to_fp32.py . /ckpts/step_100000_hf

# Then point the script at the converted directory
python .\select_pretrain_checkpoint.py `
    --checkpoints C:\ckpts\step_100000_hf `
    --output_json .\checkpoint_ranking.json
```

`zero_to_fp32.py` ships with DeepSpeed and should already be on the training node.

### Example output

```
========================================================================================================================
PRE-TRAINING CHECKPOINT RANKING
========================================================================================================================
Rank   Checkpoint                           Perplexity   BenchAvg  HellaSwag   ARC-C    Wino   LAMBADA   Stable
1      step_100000                              8.4321      67.3%      72.1%   48.2%   64.8%     63.1%      YES
2      step_090000                              9.1204      64.8%      69.4%   46.1%   62.5%     61.2%      YES
3      step_080000                              9.8847           -          -       -       -         -    SPIKE
========================================================================================================================

RECOMMENDED: /ckpts/step_100000
  Perplexity: 8.4321 | Benchmark avg: 67.3
```

Total benchmark time (all 4): ~55 min per checkpoint at 70B on 2xA100. Use `--skip_benchmarks` if you just need a fast perplexity ranking.

Results are also saved as JSON to `--output_json` for logging / sharing with the team.
