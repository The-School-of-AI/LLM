# SFT Validation Framework

Manual evaluation framework for comparing **Base vs SFT (Supervised Fine-Tuned)** model performance across 55 diverse prompts.

## Charter Objectives

| Objective | How Addressed |
|-----------|--------------|
| 50+ diverse prompts | 55 prompts across 10 categories and 3 difficulty levels |
| Compare Base vs SFT | Side-by-side scoring with per-prompt deltas |
| Instruction-following rate improved | Automated IF scoring with threshold-based pass/fail |
| No new hallucination patterns | Hallucination detector flags prompts where SFT introduces new risks |

---

## Repository Structure

```
SFT/
├── prompts/
│   └── evaluation_prompts.json       # 55 diverse evaluation prompts with rubrics
├── evaluation/
│   ├── run_evaluation.py             # Main evaluation runner (API or CSV mode)
│   ├── instruction_following.py      # IF scoring engine
│   ├── hallucination_detector.py     # Hallucination risk detector
│   └── metrics.py                    # Aggregate metrics collector & reporter
├── analysis/
│   └── analyze_results.py            # Results analysis & Markdown report generator
├── config/
│   └── config.yaml                   # Model endpoints, thresholds, generation params
├── data/
│   └── model_outputs_template.csv    # Template CSV for manual / pre-generated outputs
├── results/                          # Auto-created; stores all output files
├── tests/
│   └── test_evaluation.py            # pytest unit test suite
└── requirements.txt
```

---

## Prompt Dataset (55 Prompts)

| Category | Count | Examples |
|----------|-------|---------|
| `instruction_following` | 14 | Format, length, tone, keyword, negation, table, role-play constraints |
| `factual_qa` | 11 | History, science, geography, biology, economics, medicine |
| `reasoning` | 7 | Logic, math, probability, ethics, causal, systems thinking |
| `code_generation` | 6 | Python, SQL, JavaScript, debugging, regex, algorithms |
| `creative_writing` | 4 | Haiku+sonnet, narrative, style imitation, constrained story |
| `summarization` | 3 | Abstractive, executive summary, meeting minutes |
| `classification` | 3 | Sentiment, topic, intent detection |
| `rewriting` | 3 | Simplification, formalization, passive→active voice |
| `multi_step` | 4 | Sequential tasks, conditional logic, data analysis, research synthesis |
| `edge_cases` | 7 | Ambiguity, counterfactual, self-referential, uncertainty, numeric precision |

**Difficulty split:** Easy (18) · Medium (26) · Hard (11)

---

## How to Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Choose Your Evaluation Mode

#### Option A: CSV Mode (Recommended for manual evaluation)

1. Copy `data/model_outputs_template.csv` and rename it (e.g. `model_outputs.csv`)
2. For each `prompt_id` in `prompts/evaluation_prompts.json`, run the prompt through your base and SFT models and paste the outputs into the CSV columns `base_output` and `sft_output`
3. Run:

```bash
cd evaluation
python run_evaluation.py --mode csv --input ../data/model_outputs.csv --prompts ../prompts/evaluation_prompts.json --output-dir ../results
```

#### Option B: Auto-Populate CSV via API (`generate_outputs.py` — API mode)

`generate_outputs.py` calls both models for every prompt and writes the results directly to `model_outputs.csv`. This eliminates manual copy-paste and supports resume, parallel inference, and filtering.

**Prerequisites:** A running OpenAI-compatible endpoint for each model (vLLM, Ollama, Together AI, LM Studio, OpenAI, etc.)

1. Edit `config/config.yaml` — set `models.base.model_name`, `models.sft.model_name`, and their `endpoint` URLs
2. Set API keys:

```powershell
$env:BASE_MODEL_API_KEY = "your-base-key"
$env:SFT_MODEL_API_KEY  = "your-sft-key"
```

3. Run inference (populates `data/model_outputs.csv`):

```bash
cd evaluation
python generate_outputs.py --config ../config/config.yaml

# Preview what would run without calling any API
python generate_outputs.py --dry-run

# Resume an interrupted run (already-done rows are skipped automatically)
python generate_outputs.py --config ../config/config.yaml

# Force re-run all prompts (backs up old CSV first)
python generate_outputs.py --force-rerun

# Filter by category / difficulty / specific IDs
python generate_outputs.py --categories instruction_following factual_qa --difficulties hard
python generate_outputs.py --prompt-ids P001 P002 P003
```

4. Once `model_outputs.csv` is populated, evaluate with:

```bash
python run_evaluation.py --mode csv --input ../data/model_outputs.csv --output-dir ../results
```

---

#### Option C: Auto-Populate CSV via Local Models (`generate_outputs.py` — local mode)

Run inference entirely on your own GPU/CPU with HuggingFace Transformers — no API keys or network access required.

**Prerequisites:**

```bash
# Install PyTorch (choose the CUDA build matching your driver from pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install inference dependencies
pip install transformers accelerate

# Optional: bitsandbytes for 4-bit / 8-bit quantization (CUDA only)
pip install bitsandbytes
```

**Config (optional)** — set paths in `config/config.yaml` under `models.base.local` and `models.sft.local`:

```yaml
models:
  base:
    local:
      model_path: "/path/to/base-model"   # HF hub ID or local directory
      device: "auto"
      load_in_4bit: false
  sft:
    local:
      model_path: "/path/to/sft-checkpoint"
      device: "auto"
      load_in_4bit: false
```

**Running:**

```bash
cd evaluation

# Full run — paths via CLI (override config)
python generate_outputs.py \
  --inference-mode local \
  --base-model-path meta-llama/Llama-3.1-8B \
  --sft-model-path  ./checkpoints/sft-v1 \
  --device cuda

# 4-bit quantization (recommended for GPUs with < 24 GB VRAM)
python generate_outputs.py \
  --inference-mode local \
  --base-model-path meta-llama/Llama-3.1-8B \
  --sft-model-path  ./checkpoints/sft-v1 \
  --device cuda --load-in-4bit

# 8-bit quantization (moderate VRAM saving, may be slightly more accurate)
python generate_outputs.py \
  --inference-mode local \
  --base-model-path meta-llama/Llama-3.1-8B \
  --sft-model-path  ./checkpoints/sft-v1 \
  --device cuda --load-in-8bit

# CPU-only (slow, but works without a GPU)
python generate_outputs.py \
  --inference-mode local \
  --base-model-path meta-llama/Llama-3.1-8B \
  --sft-model-path  ./checkpoints/sft-v1 \
  --device cpu

# Dry-run preview
python generate_outputs.py --inference-mode local \
  --base-model-path meta-llama/Llama-3.1-8B \
  --sft-model-path  ./checkpoints/sft-v1 \
  --dry-run
```

> **Single-GPU note:** By default, base and SFT models are loaded and run **sequentially** to avoid OOM errors. On multi-GPU setups you can add `--parallel-local` to run them in parallel threads.

> **Memory tip:** A 7–8B model typically needs ~16 GB VRAM in fp16, ~8 GB with `--load-in-8bit`, and ~5 GB with `--load-in-4bit`.

Once outputs are collected, evaluate with the same command as Option B step 4.

---

#### Option D: Full API Mode — run_evaluation.py (Automated, requires endpoint access)

1. Edit `config/config.yaml` — set `models.base.model_name`, `models.sft.model_name`, and `endpoint` values
2. Set API keys via environment variables:

```bash
$env:BASE_MODEL_API_KEY = "your-base-key"
$env:SFT_MODEL_API_KEY  = "your-sft-key"
```

3. Run:

```bash
cd evaluation
python run_evaluation.py --mode api --config ../config/config.yaml --output-dir ../results
```

**Optional filters:**
```bash
# Evaluate only instruction_following + factual_qa prompts
python run_evaluation.py --mode csv --categories instruction_following factual_qa

# Evaluate only hard prompts
python run_evaluation.py --mode csv --difficulties hard
```

---

### Step 3 — Analyse Results

```bash
cd analysis
# Generate report from result file
python analyze_results.py --results ../results/evaluation_results_*.json

# With charts (requires matplotlib)
python analyze_results.py --results ../results/evaluation_results_*.json --charts
```

This produces:
- **Console table** with all key metrics
- **`results/report_*.md`** — Markdown report with verdict, tables, and recommendations
- **`results/per_prompt_review_*.csv`** — per-prompt scores for spreadsheet review
- **`results/summary_*.json`** — machine-readable metric summary
- **`results/charts/comparison_charts.png`** — bar charts (if `--charts` flag used)

---

### Step 4 — Run Tests

```bash
cd tests
pytest test_evaluation.py -v
```

---

## Scoring

### Instruction-Following (IF) Score

Each prompt has a `scoring_rubric` (must-have / must-not-have checks) and `instruction_constraints` (format, count, word-count, etc.).

- Each check is **binary**: pass (1) or fail (0)
- **IF score = passed checks / total checks** (range: 0.0 – 1.0)
- **Followed = True** if score ≥ threshold (default **0.75**)

Configurable in `config/config.yaml → scoring.threshold`.

### Hallucination Risk Score

Multi-signal heuristic detector:

| Signal | Severity |
|--------|---------|
| Low topic anchor coverage | 0.2 – 0.6 |
| Ground-truth contradiction | 0.8 |
| Fabrication language patterns | 0.35 |
| Generation error (API failure) | 1.0 |

- **Risk score = normalised weighted sum** (range: 0.0 – 1.0)
- **Detected = True** if risk_score ≥ threshold (default **0.5**)
- **New hallucination**: SFT detected=True AND Base detected=False

---

## Validation Pass Criteria

The SFT model **PASSES** validation when **both** conditions are met:

```
✅ SFT instruction-following rate ≥ Base IF rate
✅ Zero new hallucination patterns introduced (SFT clean where base was clean)
```

---

## Manual Annotation Workflow

For highest-quality results, augment automated scoring with human review:

1. After running `run_evaluation.py`, open `results/per_prompt_review_*.csv` in Excel / Google Sheets
2. Add columns: `human_if_score` (0/1), `human_hall_flag` (0/1), `human_notes`
3. Focus manual annotation energy on:
   - Prompts where automated IF score is between 0.5 – 0.85 (borderline cases)
   - All prompts flagged `new_hallucination = True`
   - Hard-difficulty prompts
   - Creative writing and reasoning categories (harder for heuristics)

---

## Key Metrics to Report

| Metric | Charter Requirement |
|--------|-------------------|
| `if_improvement_pp` | Instruction-following rate improved (positive = pass) |
| `sft_if_rate` vs `base_if_rate` | Direct comparison |
| `new_hallucination_count` | Must be **0** to pass |
| `per_category_if_rate` | Breakdown by category |
| `per_difficulty_if_rate` | Breakdown by difficulty |
| `regression_prompts` | Prompts where SFT got worse |

---

## Configuration Reference

Edit `config/config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.base.model_name` | — | Base model identifier |
| `models.sft.model_name` | — | SFT model identifier |
| `generation.temperature` | `0.0` | 0 = deterministic |
| `generation.max_tokens` | `1024` | Max response length |
| `scoring.threshold` | `0.75` | Min IF score to count as "followed" |
| `hallucination.threshold` | `0.50` | Min risk score to count as "detected" |

---

## Extending the Framework

- **Add prompts**: append to `prompts/evaluation_prompts.json` following the existing schema
- **Add constraint checks**: extend `InstructionFollowingScorer._dispatch_constraint()` in `evaluation/instruction_following.py`
- **Add hallucination signals**: extend `HallucinationDetector` with new `_check_*` methods
- **LLM-as-judge**: call an external model (e.g. Claude, GPT-4) inside `evaluate_prompt()` for richer IF/hallucination scoring
