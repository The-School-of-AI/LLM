# IndicIFEval (evaluation-only harness)

[![ArXiv](https://img.shields.io/badge/arXiv-2602.22125-b31b1b.svg)](https://arxiv.org/abs/2602.22125)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/ai4bharat/IndicIFEval)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repo is a **minimal setup** to run the IndicIFEval benchmark using the EleutherAI Language Model Evaluation Harness (`lm-eval`) (tested on Windows + Ubuntu).

What this repo includes:

- IndicIFEval task configs for `lm-eval`: `lm-evaluation-harness/custom_configs/indicifeval-trans/` and `lm-evaluation-harness/custom_configs/indicifeval-ground/`
- Python runners to launch evaluations with Hugging Face models: `scripts/run_indicifeval_hf*.py`

## Project structure

High-level layout:

```text
.
├─ scripts/                         # Python CLIs to run evals + generate reports
├─ lm-evaluation-harness/
│  └─ custom_configs/               # lm-eval task YAMLs + metric logic (utils.py)
│     ├─ indicifeval-trans/
│     └─ indicifeval-ground/
├─ results/                         # Output directories (results.json, logs, cache, status)
├─ paper_baselines/                 # Paper baseline JSONs used in comparisons
├─ tmp/                             # One-off analysis helpers (not required for runs)
├─ pyproject.toml / uv.lock         # uv-based dependency management
└─ requirements-eval.txt            # pip fallback dependency list
```

Where to look when changing behavior:

- Task prompting / splits / generation kwargs: `lm-evaluation-harness/custom_configs/**/indicifeval_*.yaml`
- Metric computation + instruction checks: `lm-evaluation-harness/custom_configs/**/utils.py`
- Runner behavior (resume/detach/retry/logging): `scripts/*.py`

## Features supported by the runners

The Python runners under `scripts/` support the following:

- **Deterministic seeds (best-effort):** `--seed` is forwarded to `lm_eval` and also sets common determinism-related environment variables.
- **Pause/resume (interrupt + resume):** uses `lm-eval`'s request cache (sqlite) so you can stop a run and resume without redoing completed generations.
- **Detached (survives VS Code restarts):** `--detached` starts a background worker process and writes `pid.txt` and `status.json` to the run directory.
- **Fault tolerance / retry:** `--max_attempts` and `--retry_delay_sec` allow automatic retries on transient failures; the sqlite cache keeps work from being lost.

Scripts responsible:

- `scripts/run_indicifeval_hf.py`: top-level CLI (creates the run directory, writes `run_meta.json`, supports `--detached` and `--resume`).
- `scripts/run_indicifeval_hf_worker.py`: runs `lm_eval`, writes `lm_eval.log` and `status.json`, and implements retry behavior.

## Scripts (entrypoints)

All CLIs live in `scripts/`:

- `run_indicifeval_hf.py`: main entrypoint for full runs (supports `--out_dir`, `--resume`, `--detached`, retries).
- `run_indicifeval_hf_worker.py`: invoked by the runner; executes `lm_eval` and streams output into `lm_eval.log`.
- `run_indicifeval_hf_sanity.py`: convenience wrapper for a small CPU run (`--limit`) to validate end-to-end setup.
- `generate_hi_full_report_cli.py`: CLI wrapper to generate a Hindi report from completed run directories.
- `generate_hi_full_report.py`: implementation that reads two `results.json` files (Trans + Ground) and writes a Markdown report.
- `common.py`: shared helpers (status JSON writing, deterministic env vars, safe filenames).

Notes:

- If you are using `--detached`, the **runner** starts the **worker** in a separate process and writes `pid.txt`.
- If you want to change evaluation flags passed to `lm_eval`, edit `scripts/run_indicifeval_hf_worker.py`.


## Setup (Windows)

Prereqs:

- Windows 11
- Python 3.12+

Create a venv and install dependencies:

```powershell
# Option A (recommended): uv
# Install uv first: https://docs.astral.sh/uv/
uv venv --python 3.12
uv sync --extra hf

# Option B: plain venv + pip
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe -m pip install torch transformers accelerate
```


Notes:

- For **GPU** runs, install a CUDA-enabled PyTorch wheel appropriate for your system (follow https://pytorch.org/get-started/locally/).

## Setup (Ubuntu)

Prereqs:

- Ubuntu 22.04/24.04
- Python 3.10+ (Ubuntu 24.04 ships with Python 3.12)

Install Python + venv tooling:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Create a venv and install dependencies:

```bash
# Option A (recommended): uv
# Install uv first: https://docs.astral.sh/uv/
uv venv
uv sync --extra hf

# Option B: plain venv + pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python -m pip install torch transformers accelerate
```

Notes:

- For **GPU** runs, install the correct CUDA-enabled PyTorch wheel for your system (follow https://pytorch.org/get-started/locally/).
- The provided runners are Python (`scripts/*.py`) and work on Ubuntu as-is.

## Run the benchmark (IndicIFEval)

## Overview (what is IndicIFEval?)

IndicIFEval is an instruction-following evaluation benchmark from AI4Bharat.
This harness runs the benchmark through EleutherAI `lm-eval` using the Hugging Face backend.

This repo exposes two families of tasks:

- **Trans** (`indicifeval_trans_<lang>`): translated prompts.
- **Ground** (`indicifeval_ground_<lang>`): grounded prompts.

Each example contains a prompt plus a list of instructions. The evaluation checks whether the model response follows each instruction.

All tasks are defined under `lm-evaluation-harness/custom_configs/`.

Common task ids:

- Translated subset: `indicifeval_trans_<lang>` (e.g. `indicifeval_trans_hi`)
- Grounded subset: `indicifeval_ground_<lang>` (e.g. `indicifeval_ground_hi`)

Chat-formatted variants (useful for chat-tuned models like Qwen):

- `indicifeval_trans_<lang>_chat` (e.g. `indicifeval_trans_hi_chat`)
- `indicifeval_ground_<lang>_chat` (e.g. `indicifeval_ground_hi_chat`)

These wrap `doc["prompt"]` into Qwen's `<|im_start|>user ... <|im_start|>assistant` format and stop at `<|im_end|>`. For some models this can materially change instruction-following accuracy compared to raw plain-text prompting.

## Metrics (what is reported)

Each dataset example contains multiple instructions (an `instruction_id_list`). The task code evaluates instruction-following in two ways:

- **Strict**: evaluate the model response “as-is”.
- **Loose**: evaluate an *upper bound* by trying small normalizations of the response (remove leading/trailing line, remove `*` characters), and marking an instruction as followed if it passes under any of those variants.

The four metrics in `results.json` are:

- `prompt_level_strict_acc`: for each prompt, 1 if the response follows **all** instructions strictly, else 0; final score is mean over prompts.
- `prompt_level_loose_acc`: same as above, but using loose checking.
- `inst_level_strict_acc`: for each prompt, a list of booleans (one per instruction); aggregation flattens all booleans across the dataset and returns their mean.
- `inst_level_loose_acc`: same as above, but using loose checking.

Equivalent definitions (pseudocode):

```text
prompt_level_acc = mean_over_prompts( all(instruction_followed[p][i] for i in instructions[p]) )
inst_level_acc   = mean_over_all_instructions( instruction_followed[p][i] )
```

Implementation lives in:

- `lm-evaluation-harness/custom_configs/indicifeval-trans/utils.py`
- `lm-evaluation-harness/custom_configs/indicifeval-ground/utils.py`

### Primary metric (matching the paper tables)

The IndicIFEval paper commonly reports **prompt-level loose accuracy**. In this harness, that corresponds to:

- `prompt_level_loose_acc`

When comparing with paper numbers, ensure you are comparing the same subset (Trans vs Ground), language split, and prompting format (plain vs `_chat`).

## Evaluation configuration (what we run)

There are two layers of configuration:

1) **Task YAMLs** (under `lm-evaluation-harness/custom_configs/`)

- `dataset_path`: `ai4bharat/IndicIFEval`
- `dataset_name`: `indicifeval-trans` or `indicifeval-ground`
- `test_split`: language code (e.g. `hi`, `en`)
- `output_type`: `generate_until`
- `generation_kwargs`: greedy decode (`do_sample: false`, `temperature: 0.0`) with `max_gen_toks` (1280 for full tasks; sanity tasks use 128)
- Chat tasks override `doc_to_text` and set `until: ["<|im_end|>"]`

2) **Runner/worker flags** (what `scripts/run_indicifeval_hf_worker.py` passes to `lm_eval`)

- Backend: `--model hf`
- Model args: `--model_args pretrained=<HF_ID>,trust_remote_code=True`
- Prompts: loaded from the YAML via `--include_path lm-evaluation-harness/custom_configs`
- No few-shot: `--num_fewshot 0`
- Logging: `--log_samples` (writes a per-task JSONL with prompts/outputs)
- Caching (pause/resume): `--use_cache <out_dir>/lm_eval_cache.sqlite` and `--cache_requests true`
- Reproducibility: `--seed <N>` forwarded to `lm_eval` and runner sets common determinism env vars
- `--limit <K>` is optional and intended only for smoke tests

## Result documentation (runbook-style)

Where this harness records the key facts:

- **Model + tasks + run args/config**: `run_meta.json` in the run output directory
- **Environment info** (PyTorch/CUDA/Python versions): embedded by `lm-eval` inside `results.json` under `pretty_env_info`
- **Exact metrics**: `results.json` under `results.<task_name>`
- **Logs / stdout+stderr**: `lm_eval.log`
- **Success/failure + exit code**: `status.json`

Recommended checklist for any “full pass” result you publish:

- Model id (and ideally HF revision/commit if pinned)
- Task(s) (e.g. `indicifeval_trans_hi`, `indicifeval_ground_hi`, and whether `_chat` was used)
- Run command (copy/paste) and output directory
- Split / few-shot (`num_fewshot: 0`) / decoding (`do_sample: false`, `temperature: 0.0`)
- Primary metric: `prompt_level_loose_acc` (plus the other three metrics for context)
- Any comparability caveats (prompt format, subset filtering, harness/version differences)

### Comparison & caveats (why scores differ)

Small evaluation differences can move scores by several points:

- **Prompting format:** for chat-tuned models, the `_chat` tasks wrap prompts into a chat template and change stop tokens.
- **Subset differences:** if you compare against a paper result, confirm the paper used the same subset/split and filtering.
- **Harness/version differences:** `lm-eval` version, HF generation defaults, and tokenizer behavior can all affect results.

### Validation checklist (recommended)

Before treating a score as “final” (or comparing against paper baselines):

- Confirm `status.json` is `succeeded` and `exit_code` is 0
- Verify `results.json` contains the expected task key(s) and metrics under `results.<task_name>`
- Confirm you ran the intended task variant (plain vs `_chat`) and split (e.g. `hi`)
- Spot-check the logged samples JSONL (written by `--log_samples`) to ensure prompts and outputs look as expected
- Record any known comparability caveats (prompt format, subset filtering, harness/version) alongside the number

### Quick sanity run (CPU)

```powershell
uv run python .\scripts\run_indicifeval_hf_sanity.py --model Qwen/Qwen3-0.6B --task indicifeval_trans_en_sanity --limit 2
```

### Full runs (GPU)

Hindi, Trans:

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_trans_hi --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_trans_full --detached
```

Hindi, Ground:

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_ground_hi --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_ground_full --detached
```

Hindi, chat-formatted (often closer to the model's intended interface):

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_trans_hi_chat --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_trans_chat_full --detached
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_ground_hi_chat --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_ground_chat_full --detached
```

### Outputs

Each run directory contains:

- `results.json` (final metrics)
- `lm_eval.log` (combined stdout/stderr)
- `lm_eval_cache.sqlite*` (sqlite cache; enables resume)
- `status.json`, `run_meta.json`, `pid.txt` (runner bookkeeping)

### Pause/resume and fault tolerance

The runners support **stop + resume** via `lm-eval`'s sqlite request cache.

Configuration requirements:

- Always set an explicit `--out_dir` for long runs. If you let the runner auto-pick a timestamp directory, you won't be able to reliably resume into the same run folder.

How to resume:

1. Start a run with a fixed `--out_dir` (optionally `--detached`):

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_trans_hi --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_trans_full --detached
```

2. If the run is interrupted (crash, reboot, manual stop), rerun the exact same command but add `--resume`:

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_trans_hi --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_trans_full --resume --detached
```

Fault tolerance knobs:

- `--max_attempts N`: number of times to attempt `lm_eval`.
- `--retry_delay_sec S`: delay between attempts.

Example (retry up to 3 times):

```powershell
uv run python .\scripts\run_indicifeval_hf.py --model Qwen/Qwen3-0.6B --tasks indicifeval_trans_hi --device cuda:0 --batch_size 1 --seed 42 --out_dir results/hf/Qwen_Qwen3-0.6B/hi_trans_full --detached --max_attempts 3 --retry_delay_sec 60
```

## Full-pass Hindi run (Qwen/Qwen3-0.6B)

The end-to-end full Hindi results and paper comparison are captured in:

- [indicifeval_hi_qwen3-0.6b_full_vs_paper.md](indicifeval_hi_qwen3-0.6b_full_vs_paper.md)

## Citation

If you use IndicIFEval in your work, please cite us:

```bibtex
@article{jayakumar2026indicifeval,
      title={IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages}, 
      author={Thanmay Jayakumar and Mohammed Safi Ur Rahman Khan and Raj Dabre and Ratish Puduppully and Anoop Kunchukuttan},
      year={2026},
      eprint={2602.22125},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.22125}, 
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Links

- [GitHub Repository 💻](https://github.com/AI4Bharat/IndicIFEval)
- [Paper 📄](https://arxiv.org/abs/2602.22125)
- [Hugging Face Dataset 🤗](https://huggingface.co/datasets/ai4bharat/IndicIFEval)


