# Team 16 — Early Warning Evaluation Suite (Experiment 1)

Lightweight, quantized, distributed evaluation infrastructure for monitoring
LLM checkpoint quality during training. Designed to run on consumer hardware
(MacBooks with 16 GB+ RAM, RTX 3090-class GPUs) and detect regressions early.

---

## Table of Contents

1. [How MMLU Works — What Happens Before Your Model Exists](#1-how-mmlu-works--what-happens-before-your-model-exists)
2. [Run the Pipeline — the Only File You Need to Edit](#2-run-the-pipeline--the-only-file-you-need-to-edit)
3. [Step-by-Step: First-Time Setup](#3-step-by-step-first-time-setup)
4. [Step-by-Step: Adding Your Training Checkpoint](#4-step-by-step-adding-your-training-checkpoint)
5. [MMLU Category Coverage Including Indian Languages](#5-mmlu-category-coverage-including-indian-languages)
6. [Evaluation Suite Details](#6-evaluation-suite-details)
7. [Anomaly Detection](#7-anomaly-detection)
8. [Quantization Backends](#8-quantization-backends)
9. [Distributed Workflow](#9-distributed-workflow)
10. [Output Files](#10-output-files)
11. [Full Configuration Reference](#11-full-configuration-reference)
12. [Project Structure](#12-project-structure)
13. [Phase Checklist](#13-phase-checklist)

---

## 1. How MMLU Works — What Happens Before Your Model Exists

**Q: We don't have `mmlu_subset.json` yet. How does evaluation work?**

The MMLU subset is **downloaded from HuggingFace, not bundled with this repo**.
Here is the exact sequence on the very first run:

```
python run_pipeline.py
        │
        ▼
build_mmlu_subset.py ──internet──► HuggingFace (cais/mmlu + sarvamai/mmlu-indic)
        │
        │  Downloads questions, samples 7 per category, saves locally
        ▼
data/mmlu_subset/mmlu_subset.json   ← CREATED HERE, frozen forever after this
        │
        ▼
run_eval.py reads this file and evaluates whatever model you point it at
```

**Key points:**

- `mmlu_subset.json` does **not** exist in the repo on day 1 — that is expected and correct
- The pipeline creates it automatically on first run (internet required, one time only)
- Once created, it is **never changed again** — this guarantees checkpoint-to-checkpoint trends are comparable because everyone is answering the exact same questions
- The same frozen file is used for GPT-2 today and for your trained model next week
- **Commit it to git after building** so all team members share the identical file

**Q: My model isn't trained yet — what can we actually run today?**

Use the open-source baselines already registered in `run_pipeline.py`:

| Model | Size | Download | Why useful |
|-------|------|----------|------------|
| GPT-2 | 117M | ~500 MB, auto | Absolute floor — any trained model should beat this |
| GPT-2 Medium | 345M | ~1.4 GB, auto | Better baseline if GPT-2 is too weak |
| SmolLM2-135M | 135M | ~270 MB, auto | Fast alternative |

Running GPT-2 through the full suite today means:
1. You verify every script works end-to-end
2. You get a real baseline score in every metric
3. When your checkpoint arrives, the trend chart immediately shows whether training helped

---

## 2. Run the Pipeline — the Only File You Need to Edit

**Everything is controlled from one file: [`run_pipeline.py`](run_pipeline.py)**

Open it. You will see two editable sections at the top:

### Section A — CHECKPOINTS list

```python
CHECKPOINTS: list[dict] = [

    # ── Open-source baseline (runs today, no training needed) ──
    {
        "name":     "gpt2_baseline",
        "path":     "gpt2",          # HuggingFace model ID — downloads automatically
        "backend":  "hf",
        "quant":    "fp32",
        "enabled":  True,            # ← True = included in next run
        "notes":    "GPT-2 117M — smoke-test / absolute baseline",
    },

    # ── Your model checkpoints (fill in as training progresses) ──
    {
        "name":     "my_model_step_500",
        "path":     "/path/to/your/checkpoint-500",   # ← UPDATE THIS PATH
        "backend":  "auto",
        "quant":    "int4",
        "enabled":  False,           # ← flip to True when the checkpoint exists
        "notes":    "Step 500",
    },
]
```

**Fields explained:**

| Field | What to put | Example values |
|-------|-------------|----------------|
| `name` | Short label used in plots and reports | `"step_500"`, `"epoch_2"` |
| `path` | HuggingFace model ID **or** local folder path **or** `.gguf` file | `"gpt2"`, `"/data/ckpt-500"`, `"model.gguf"` |
| `backend` | Inference backend — use `"auto"` when unsure | `"auto"`, `"hf"`, `"bitsandbytes"`, `"llama_cpp"` |
| `quant` | Quantization level | `"fp32"`, `"fp16"`, `"int8"`, `"int4"` |
| `enabled` | Whether to run this checkpoint | `True` or `False` |
| `notes` | Free text shown in reports | `"After warmup, lr=3e-4"` |

### Section B — PIPELINE_OPTIONS

```python
PIPELINE_OPTIONS = {
    "skip_if_already_evaluated": True,   # True = don't re-run if result file exists
    "plot_format": "both",               # "png", "html", or "both"
    "verbose_eval": True,                # show per-sample tqdm bars
    "collector_url": None,               # "http://host:5001" to auto-submit results
    "auto_build_mmlu": True,             # build mmlu_subset.json on first run
}
```

### CLI flags (no file editing needed)

```bash
python run_pipeline.py                        # run all enabled checkpoints
python run_pipeline.py --list                 # show all registered checkpoints + status
python run_pipeline.py --name gpt2_baseline   # run exactly one checkpoint
python run_pipeline.py --skip-eval            # aggregate + trend + report only (no model loading)
python run_pipeline.py --force                # re-evaluate even if result file already exists
```

---

## 3. Step-by-Step: First-Time Setup

```bash
# ── 1. Install dependencies ───────────────────────────────────────────────
pip install -r requirements.txt

# Apple Silicon Mac (recommended: Metal GPU acceleration)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA CUDA machine
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# ── 2. Check your hardware (optional) ────────────────────────────────────
python scripts/hardware_survey.py
# → Prints GPU/RAM info and recommended backend/quant settings

# ── 3. See what checkpoints are registered ───────────────────────────────
python run_pipeline.py --list

# ── 4. Run the full pipeline on the GPT-2 baseline ───────────────────────
python run_pipeline.py --name gpt2_baseline
# This will:
#   a) Download + build mmlu_subset.json (one-time, needs internet, ~3-5 min)
#   b) Download GPT-2 from HuggingFace (~500 MB)
#   c) Run MMLU + all 4 custom probes
#   d) Generate trend plots and early-warning report
#   e) Print the report to the terminal

# ── 5. View outputs ───────────────────────────────────────────────────────
cat results/reports/latest_early_warning.md
open results/plots/trend_dashboard.html        # macOS — interactive chart
```

**Expected duration on an M-series MacBook (16 GB RAM):**
- Building MMLU subset: ~3–5 min (one-time internet download)
- GPT-2 fp32 full evaluation: ~10–20 min (CPU only, 399 MMLU questions + 4 probe suites)

---

## 4. Step-by-Step: Adding Your Training Checkpoint

When training saves a new checkpoint, follow these steps:

### Step 1 — Open `run_pipeline.py`, fill in the path, flip `enabled`

```python
# Before (placeholder):
{
    "name":     "my_model_step_500",
    "path":     "/path/to/your/model/checkpoint-500",
    "backend":  "auto",
    "quant":    "int4",
    "enabled":  False,
    "notes":    "Step 500",
},

# After (ready to run):
{
    "name":     "my_model_step_500",
    "path":     "/data/training/runs/exp1/checkpoint-500",  # ← real path
    "backend":  "auto",
    "quant":    "int4",
    "enabled":  True,                                        # ← flipped
    "notes":    "Step 500 — after warmup, lr peaked",
},
```

### Step 2 — Run the pipeline

```bash
python run_pipeline.py --name my_model_step_500
```

The pipeline will:
1. Skip MMLU build (already frozen from first run)
2. Load your checkpoint with INT4 quantization
3. Evaluate on the **exact same** questions GPT-2 was evaluated on
4. Compare to all previous results and flag regressions automatically
5. Write a new early-warning report

### Step 3 — Read the report

```bash
cat results/reports/latest_early_warning.md
```

A healthy run looks like:
```
✅ No regressions, plateaus, or instability detected.
All metrics are trending normally across 2 checkpoints.
```

A regression looks like:
```
| 🔴 HIGH  | REGRESSION | MMLU Accuracy | my_model_step_500 | dropped by 0.062 |
| 🟡 LOW   | PLATEAU    | Code Pass Rate | my_model_step_500 | flat over 3 ckpts |
```

### Which backend and quant to pick for your checkpoint

| Situation | `backend` | `quant` |
|-----------|-----------|---------|
| HuggingFace folder, NVIDIA GPU with 8 GB+ VRAM | `bitsandbytes` | `int4` |
| HuggingFace folder, NVIDIA GPU with 16 GB+ VRAM | `bitsandbytes` | `int8` |
| HuggingFace folder, Mac M-series 16 GB+ | `hf` | `fp16` |
| `.gguf` file (pre-quantized) | `llama_cpp` | `int4` |
| Not sure | `auto` | `int4` |

**Tip:** `"backend": "auto"` auto-detects `.gguf` files by extension and falls back
to `bitsandbytes` for HuggingFace checkpoint folders — safe default for most cases.

---

## 5. MMLU Category Coverage Including Indian Languages

### Why standard MMLU has no Indian languages

The original `cais/mmlu` dataset is **English only**. It covers 57 academic and
professional subject areas (mathematics, physics, law, medicine, etc.) but has zero
Indian language content. This is a gap if your model is trained on multilingual data.

### How we extend MMLU with Indian languages

We use **`sarvamai/mmlu-indic`** — a HuggingFace dataset that provides the same
MMLU questions translated into 10 Indian languages in their native scripts, plus
romanized versions. The build script can pull from both datasets and merge them
into a single frozen `mmlu_subset.json`.

**To enable Indian languages, set them in `configs/eval_config.yaml`:**

```yaml
mmlu:
  questions_per_category: 7
  indic_languages:      # add the language codes you want
    - hi                # Hindi (Devanagari)
    - bn                # Bengali
    - ta                # Tamil
    - te                # Telugu
    - mr                # Marathi (Devanagari)
    - gu                # Gujarati
    - kn                # Kannada
    - ml                # Malayalam
    - pa                # Punjabi (Gurmukhi)
    - or                # Odia
```

Or pass them directly to the build script:

```bash
# Hindi + Tamil only
python scripts/build_mmlu_subset.py --indic-langs hi ta

# All 10 Indian languages
python scripts/build_mmlu_subset.py --indic-langs hi bn ta te mr gu kn ml pa or
```

**Important:** Set `indic_languages` in the YAML config **before** the first run,
because the subset is frozen after building. To change it later you must rebuild
with `--force` and all previous trends become incomparable.

### Complete Indian language coverage table

| Code | Language | Script | Available in |
|------|----------|--------|-------------|
| `hi` | Hindi | Devanagari | `sarvamai/mmlu-indic` |
| `bn` | Bengali | Bengali | `sarvamai/mmlu-indic` |
| `ta` | Tamil | Tamil | `sarvamai/mmlu-indic` |
| `te` | Telugu | Telugu | `sarvamai/mmlu-indic` |
| `mr` | Marathi | Devanagari | `sarvamai/mmlu-indic` |
| `gu` | Gujarati | Gujarati | `sarvamai/mmlu-indic` |
| `kn` | Kannada | Kannada | `sarvamai/mmlu-indic` |
| `ml` | Malayalam | Malayalam | `sarvamai/mmlu-indic` |
| `pa` | Punjabi | Gurmukhi | `sarvamai/mmlu-indic` |
| `or` | Odia | Odia | `sarvamai/mmlu-indic` |
| `hi_roman` | Hindi (romanized) | Latin | `sarvamai/mmlu-indic` |
| `ur` | Urdu | Nastaliq | `LinguaLift/IndicMMLU-Pro` |

### How Indic questions look in the frozen subset

```json
{
  "category": "high_school_mathematics",
  "language": "hi",
  "language_name": "Hindi",
  "source_dataset": "sarvamai/mmlu-indic",
  "question": "यदि x² - 5x + 6 = 0, तो x के मान हैं:",
  "choices": ["2 और 3", "1 और 6", "2 और 4", "3 और 6"],
  "answer": 0
}
```

English questions in the same file look identical except `"language": "en"`.

### Why run Indic MMLU even if your model is English-only

- **Baseline curiosity:** GPT-2 scores ~25% (random chance) on Indic MMLU.
  If your model scores above 25%, it has picked up cross-lingual transfer.
- **Regression detection:** If Indic accuracy drops while English improves, that is
  a sign of catastrophic forgetting of multilingual capacity.
- **Future-proofing:** If your training data includes any Indic text, this tells you
  immediately whether that data is helping.

### Other Indic benchmarks (future expansion)

| Dataset | HuggingFace ID | Languages | Notes |
|---------|---------------|-----------|-------|
| MILU | `ai4bharat/MILU` | 11 Indian + English | 80 k questions, 8 domains — most comprehensive |
| IndicMMLU-Pro | `LinguaLift/IndicMMLU-Pro` | 9 (incl. Urdu) | Good if Urdu matters |
| Global-MMLU | `CohereLabs/Global-MMLU` | hi, te, bn (+ 39 others) | 42-language multilingual benchmark |

---

## 6. Evaluation Suite Details

### Reduced MMLU (Frozen)
- **Source:** `cais/mmlu` (English) + `sarvamai/mmlu-indic` (if configured)
- **Default size:** 7 questions × 57 English categories = 399 questions
- **With all 10 Indic languages:** 399 English + 70 per language = 1099 questions total
- **Scoring method:** Log-likelihood over 4 choices — picks whichever choice the model assigns highest probability to. This is more robust than parsing generated text for quantized models.
- **Metric:** `overall_accuracy` (↑ better), also broken down per category and per language

### Language Modeling Probe
- 15 curated passages: news, literature, science, law, CS, philosophy, history, economics
- **Metric:** `mean_perplexity` = exp(avg NLL per token) — lower is better
- Direct measure of next-token prediction quality; a core training health signal

### Code Continuation Probe
- 12 Python code snippets (sorting, recursion, data structures, generators, decorators)
- Model must complete the snippet; scored by keyword and regex pattern matching
- **Metric:** `pass_rate` (↑ better)

### Math Prose Probe
- 20 natural-language math/reasoning questions spanning easy to hard
- Scored by keyword presence in the generated answer
- **Metric:** `accuracy` (↑ better)

### Consistency Probe
- 10 question groups, each rephrased 4 ways with identical meaning
- All 4 variants must produce the same correct answer
- **Metric:** `mean_agreement_rate` (↑ better, 1.0 = perfectly consistent)

---

## 7. Anomaly Detection

Configured in `configs/eval_config.yaml` under `trend_tracking`:

| Signal | Description | Default threshold | Severity |
|--------|-------------|-------------------|----------|
| **Regression** | Metric drops vs previous checkpoint | > 2% drop | 🔴 HIGH (>5%) / 🟠 MEDIUM |
| **Plateau** | Range over last N checkpoints is tiny | < 0.5% range over 3 ckpts | 🟡 LOW |
| **Instability** | High standard deviation across all checkpoints | std > 5% | 🟠 MEDIUM |

A report is written to `results/reports/latest_early_warning.md` after every run.
Flag any 🔴 HIGH or 🟠 MEDIUM warning to Teams 9 and 12 immediately.

---

## 8. Quantization Backends

| Backend | Model format | Best hardware | Quant options |
|---------|-------------|---------------|---------------|
| `llama_cpp` | `.gguf` file | MacBook (Metal), CPU | `int4`, `int8` |
| `bitsandbytes` | HuggingFace folder | NVIDIA GPU (RTX 3090+) | `int4`, `int8` |
| `hf` | HuggingFace folder | Any (reference quality) | `fp16`, `fp32` |
| `auto` | Either | Picks automatically | matches `quant` field |

**Memory required (approximate, 7B-parameter model):**

| Mode | RAM / VRAM |
|------|-----------|
| fp32 | ~28 GB |
| fp16 | ~14 GB |
| int8 | ~7 GB |
| int4 | ~4 GB |

**For GPT-2 (117M) used in smoke-testing:** fp32 uses ~450 MB — runs on any machine.

---

## 9. Distributed Workflow

### Option A: File-based (recommended for most teams)

1. Each student runs `python run_pipeline.py --name <checkpoint>`
2. Copies `results/raw/<run_id>.json` to a shared folder (Google Drive, Git LFS, etc.)
3. Team lead drops all JSON files into `results/raw/` and runs:
   ```bash
   python run_pipeline.py --skip-eval   # aggregate + trend + report without re-evaluating
   ```

### Option B: Flask server (real-time dashboard)

On the central machine:
```bash
python collector/flask_collector.py --port 5001
```

Set `collector_url` in `run_pipeline.py`:
```python
"collector_url": "http://192.168.1.100:5001",
```

Results auto-submit after every eval. Dashboard at `http://192.168.1.100:5001/`.

---

## 10. Output Files

| File | Created when | Description |
|------|-------------|-------------|
| `data/mmlu_subset/mmlu_subset.json` | First pipeline run | Frozen MMLU questions — never change |
| `results/raw/<run_id>.json` | After each checkpoint eval | All metrics + per-question details |
| `results/aggregated_results.json` | After aggregate step | Summary across all runs |
| `results/trend_db.json` | After trend step | Metric series, deltas, anomaly list |
| `results/plots/trend_*.png` | After trend step | Per-metric matplotlib plots |
| `results/plots/trend_dashboard.html` | After trend step | Interactive Plotly dashboard |
| `results/reports/latest_early_warning.md` | After report step | Current warning report |
| `results/reports/latest_early_warning.html` | After report step | HTML version |

---

## 11. Full Configuration Reference

[`configs/eval_config.yaml`](configs/eval_config.yaml) — every tunable parameter:

```yaml
seed: 42                              # Never change this after the first run

mmlu:
  questions_per_category: 7          # Questions per MMLU category — frozen after build
  subset_file: "data/mmlu_subset/mmlu_subset.json"
  indic_languages: []                 # e.g. [hi, bn, ta, te, mr, gu, kn, ml, pa, or]
  indic_dataset: "sarvamai/mmlu-indic"
  # Full list of 57 English categories is in the file

probes:
  language_modeling:
    enabled: true                     # set false to skip
  code_continuation:
    enabled: true
  math_prose:
    enabled: true
  consistency:
    enabled: true

quantization:
  llama_cpp:
    n_ctx: 2048
    n_threads: 8
    n_gpu_layers: -1                  # -1 = all layers on GPU

evaluation:
  max_new_tokens: 256
  temperature: 0.0                    # greedy = reproducible

trend_tracking:
  regression_threshold: -0.02        # flag drops > 2%
  plateau_window: 3
  plateau_threshold: 0.005
  instability_std_threshold: 0.05

early_warning:
  notify_teams: ["Team_9", "Team_12"]
```

---

## 12. Project Structure

```
Experiment_1/
├── run_pipeline.py               ← THE MAIN FILE — edit checkpoints here
│
├── configs/
│   └── eval_config.yaml          # All thresholds, seeds, probe paths, Indic langs
│
├── data/
│   ├── mmlu_subset/
│   │   └── mmlu_subset.json      # Built on first run, frozen forever after
│   └── probes/
│       ├── language_modeling_probes.json
│       ├── code_probes.json
│       ├── math_prose_probes.json
│       └── consistency_probes.json
│
├── evals/
│   ├── model_loader.py           # Unified loader: bitsandbytes / llama.cpp / HF
│   ├── mmlu/run_mmlu.py          # MMLU subset evaluator
│   ├── probes/
│   │   ├── language_modeling.py
│   │   ├── code_continuation.py
│   │   ├── math_prose.py
│   │   └── consistency.py
│   └── quantized/run_eval.py     # Low-level entrypoint (called by run_pipeline.py)
│
├── collector/
│   ├── collect_results.py        # Aggregate JSON results
│   └── flask_collector.py        # Optional HTTP server
│
├── scripts/
│   ├── build_mmlu_subset.py      # Download + freeze MMLU + Indic subset
│   ├── hardware_survey.py        # Detect local GPU/RAM
│   ├── track_trends.py           # Anomaly detection + matplotlib/plotly plots
│   ├── generate_report.py        # MD/HTML/JSON early-warning report
│   └── submit_result.py          # POST result to Flask collector
│
├── results/                      # All outputs (auto-created, gitignored by default)
│   ├── raw/
│   ├── plots/
│   └── reports/
│
├── utils/config.py
└── requirements.txt
```

---

## 13. Phase Checklist

### Phase 1 — Setup (Days 1–2)
- [ ] `pip install -r requirements.txt`
- [ ] `python scripts/hardware_survey.py` — share JSON output with team lead
- [ ] Edit `configs/eval_config.yaml` — set `indic_languages` if needed
- [ ] `python run_pipeline.py --list` — verify checkpoint registration
- [ ] `python run_pipeline.py --name gpt2_baseline` — full smoke test
- [ ] Confirm `data/mmlu_subset/mmlu_subset.json` was created
- [ ] **Commit `mmlu_subset.json` to git** so all team members share the same file
- [ ] Open `results/plots/trend_dashboard.html` — verify plots render
- [ ] (Optional) Start Flask collector: `python collector/flask_collector.py`

### Phase 2 — Continuous Evaluation
- [ ] New checkpoint saved → open `run_pipeline.py`, update `path`, set `enabled: True`
- [ ] `python run_pipeline.py --name my_model_step_<N>`
- [ ] Check `results/reports/latest_early_warning.md`
- [ ] Flag any 🔴 HIGH or 🟠 MEDIUM warning to Teams 9 and 12 immediately
- [ ] Share `results/raw/<run_id>.json` with team

### Phase 3 — Wind Down
- [ ] `python run_pipeline.py --skip-eval` — final aggregate + report
- [ ] Archive `results/` directory
- [ ] Lock `mmlu_subset.json` — mark as read-only so it cannot be accidentally changed

---

*Team 16 Early Warning — ERAV4 Final Project*
