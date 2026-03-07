# Final Datasets and Pipeline — Team 18 SFT

Single reference for: **datasets to source** (and why), **relevant benchmarks**, **decontamination**, **scripts** (source → decontaminate → standardize → chat template), and **chat template** definition.

**Related docs:** [DATASET_SOURCING_STRATEGY.md](./DATASET_SOURCING_STRATEGY.md), [DATASET_BENCHMARK_COVERAGE_MATRIX.md](./DATASET_BENCHMARK_COVERAGE_MATRIX.md), [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md), [scripts/README.md](./scripts/README.md).

---

## 1. Final datasets identified to be sourced

The following datasets are the **agreed set** to source for SFT. For each: **why** we use it, **relevant benchmarks** it supports, and **how to source** (script + IDs).

| Dataset | Why source it | Relevant benchmarks | Tier |
|---------|----------------|----------------------|------|
| **Tulu 3 SFT Mixture** (subsample) | 19-source mixture: code (Evol CodeAlpaca, Persona Python), math (NuminaMath-TIR, Persona MATH/Algebra/GSM), instruction-following (Persona IF). Clear provenance; broad coverage. | MMLU, TriviaQA, MMLU-Pro, GPQA, ARC, GSM8K, BBH, MATH, DROP, IFEval, SimpleQA, TruthfulQA, APPS, MT-Bench, WildBench, L-Eval, RULER | 1 + 3 |
| **OpenMathInstruct-2** (subsample) | Competition-style math with ground-truth answers; directly targets MATH and harder math. | MATH, AIME 2025, GSM8K | 1 |
| **Magicoder OSS-Instruct + CodeFeedback-Filtered** | Code with test suites; mechanically verifiable. | HumanEval, MBPP, APPS, DS1000 | 1 |
| **Tulu 3 instruction-following subset (Persona IF)** | IFEval-style format/length constraints. | IFEval | 1 |
| **NuminaMath-TIR** (subsample) | AIME-level competition math; ground-truth. | AIME 2025, MATH | 1 |
| **Spider + BIRD-SQL filtered** | SQL with expected results; mechanically verifiable. | Spider, BIRD (SQL) | 1 |
| **SWE-smith** (subsample) | SWE-style tasks with testability. | SWE-bench Verified | 1 |
| **PKU-SafeRLHF** (chosen, subsample) | Structured preference/chosen responses; safety and alignment. | TruthfulQA, safety evals | 2 |
| **IndicAlign** (subsample, if Indic in scope) | Indic-language instruction data; structured. | IndicGLUE, IndicQA, Indic-Bias, MMLU-Indic | 2 |
| **General instruction (Tulu 3: FLAN, WildChat, Aya, OpenAssistant)** | Dialogue and general helpfulness; avoid overfitting to only math/code. | MT-Bench, WildBench, broad | 3 |

**Budget note:** For a 50–100K SFT budget, subsample from the above (e.g. Tulu 3 subsample + OpenMathInstruct-2 + Magicoder/CodeFeedback + Tulu 3 IF as core). Document exact mix and sizes.

---

## 2. Script to source the datasets

Use the **sourcing script** to download the identified datasets from canonical sources (Hugging Face). It writes raw data to JSONL (or you can keep in Hugging Face format and convert later).

**Script:** `scripts/source_datasets.py`

**What it does:** For each dataset in the table above, loads from Hugging Face (or specified path), optionally subsamples, and saves to a configurable output directory. You can run it for all datasets or for a subset via arguments.

**Example (after installing `datasets`):**

```bash
cd sft_data/scripts
python source_datasets.py --output-dir /path/to/raw_data --datasets tulu3 openmath magicoder
# Or run for all: --datasets all
```

**Dataset IDs (Hugging Face):**

| Dataset | Hugging Face ID / source |
|---------|---------------------------|
| Tulu 3 SFT Mixture | `allenai/tulu-3-sft-mixture` |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` |
| Magicoder OSS-Instruct | `ise-uiuc/Magicoder-OSS-Instruct-75K` |
| CodeFeedback-Filtered | (check HF for CodeFeedback / Magicoder filtered variants) |
| NuminaMath-TIR | `AI-MO/NuminaMath-TIR` |
| Spider (filtered) | `xlangai/spider` (use train split) |
| BIRD-SQL filtered | `birdsql/bird23-train-filtered` or `PipableAI/pip-txt-to-sql-spider-bird-dataset` |
| SWE-smith | (check HF: SWE-GYM / SWE-smith) |
| PKU-SafeRLHF | `PKU-Alignment/PKU-SafeRLHF` (use chosen responses) |
| IndicAlign | (check HF: IndicAlign instruction data) |

If a dataset is not on HF or has a different ID, edit `source_datasets.py` or the config it uses. Sourcing can also be done manually via `datasets.load_dataset("org/name")` and export to JSONL.

---

## 3. Decontamination (no benchmark contamination)

We must ensure **no benchmark test-set content** appears in SFT data. Use the following scripts.

### 3.1 Build benchmark test-set hashes

**Script:** `scripts/build_benchmark_hashes.py`

Builds a file of hashes (one per line) from each benchmark **test** set JSONL. Same hash function (SHA256 of normalized text) as decontamination and dedup.

```bash
python build_benchmark_hashes.py /path/to/math_test.jsonl benchmark_hashes/math_test.txt --text-field problem
python build_benchmark_hashes.py /path/to/gsm8k_test.jsonl benchmark_hashes/gsm8k_test.txt --text-field question
# Repeat for HumanEval, IFEval, MMLU, etc., using the correct --text-field for each benchmark.
```

### 3.2 Decontaminate SFT data against benchmarks

**Script:** `scripts/decontaminate_against_benchmarks.py`

Removes SFT examples whose **prompt** (user content) hash matches any benchmark test-set hash. Run **after** standardize (conversation format).

```bash
# Using a directory of hash files (one per benchmark)
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes-dir benchmark_hashes/

# Or individual hash files
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes benchmark_hashes/math_test.txt \
  --benchmark-hashes benchmark_hashes/gsm8k_test.txt

# Optional: write removed examples for audit
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes-dir benchmark_hashes/ --removed-out removed_contaminated.jsonl
```

**Default:** `--hash-mode prompt` (user content only). Use `--hash-mode full` to match on full conversation. See [DATASET_BENCHMARK_COVERAGE_MATRIX.md](./DATASET_BENCHMARK_COVERAGE_MATRIX.md) for the full contamination checklist.

---

## 4. Script to standardize data format (system / user / assistant)

All SFT data must be in **conversation format**: turns with `role` and `content` (`system`, `user`, `assistant`).

**Script:** `scripts/standardize_conversation_format.py`

**What it does:** Converts input JSONL from Alpaca-style (instruction/input/output), ShareGPT-style, or already-conversation into a single schema: `{"conversations": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}`.

```bash
python standardize_conversation_format.py input.jsonl standardized.jsonl --format alpaca
# or --format sharegpt | already_conversation
```

- **`--format alpaca`** — Fields: `instruction`, `input`, `output` (optional `system`).
- **`--format sharegpt`** — Conversation list with `from`/`value` or `role`/`content` (human/gpt → user/assistant).
- **`--format already_conversation`** — Normalizes `conversations` or `messages` to standard roles.

**Pipeline place:** Run first on raw or sourced data. Output `standardized.jsonl` is the input for decontamination, then chat template.

---

## 5. Chat template defined and documented

One chat template is used **consistently** for all SFT data and evaluation (no per-benchmark prompts). The base model / tokenizer (Team 6) determines which option applies.

**Full definition:** [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md)

### Option A: ChatML (OpenAI-style)

- **Format:** `<|im_start|>role\ncontent<|im_end|>` for each turn (system, user, assistant).
- **Special tokens:** `<|im_start|>`, `<|im_end|>`, optionally `<|endoftext|>`.
- **Use when:** Base model uses ChatML (e.g. many OpenAI-style models).

### Option B: Llama 3 / Llama 3.1

- **Format:** `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`; optional `<|begin_of_text|>` at start.
- **Special tokens:** `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`.
- **Use when:** Base model is Llama 3 family (usually already in tokenizer).

### Option C: Custom

- Document the exact template string and special tokens in [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md) (Option C section).

**Application:** The chosen template is applied in `scripts/apply_chat_template.py` (see below). Use the same template in the training collator and in evaluation.

---

## 6. Script to apply the chat template

**Script:** `scripts/apply_chat_template.py`

Takes **standardized** (or decontaminated) conversation JSONL and produces JSONL with a `text` field: the conversation rendered with the chosen chat template (ChatML or Llama).

```bash
python apply_chat_template.py decontaminated.jsonl templated.jsonl --template chatml
# or --template llama
# optional: --tokenizer path/to/model --max-length 2048
```

Use `decontaminated.jsonl` (or `standardized.jsonl` if you skip decontamination). Downstream: quality sample, dedup vs pretrain, train/val split, verify loss masking.

---

## 7. End-to-end pipeline order

| Step | Script | Input → Output |
|------|--------|-----------------|
| 1 | **source_datasets.py** | — → raw JSONL (per dataset or combined) |
| 2 | **standardize_conversation_format.py** | raw JSONL → **standardized.jsonl** (system/user/assistant) |
| 3 | **build_benchmark_hashes.py** | benchmark test JSONL → hash files (one per benchmark) |
| 4 | **decontaminate_against_benchmarks.py** | standardized.jsonl + hash files → **decontaminated.jsonl** |
| 5 | **apply_chat_template.py** | decontaminated.jsonl + **CHAT_TEMPLATE** → **templated.jsonl** |
| 6 | sample_for_quality_review.py | (standardized or templated) → review sample |
| 7 | dedup_against_pretrain.py | decontaminated or standardized → deduped.jsonl (requires Team 5 hashes) |
| 8 | train_val_split.py | deduped.jsonl → train.jsonl, val.jsonl |
| 9 | verify_loss_masking.py | train.jsonl → verify labels (assistant-only, -100 elsewhere) |

---

## 8. Quick reference — script summary

| Purpose | Script | Key args |
|---------|--------|----------|
| **Source datasets** | `scripts/source_datasets.py` | `--output-dir`, `--datasets` |
| **Build benchmark hashes** | `scripts/build_benchmark_hashes.py` | `input.jsonl`, `output.txt`, `--text-field` |
| **Decontaminate vs benchmarks** | `scripts/decontaminate_against_benchmarks.py` | `standardized.jsonl`, `decontaminated.jsonl`, `--benchmark-hashes-dir` |
| **Standardize format** | `scripts/standardize_conversation_format.py` | `input.jsonl`, `standardized.jsonl`, `--format alpaca\|sharegpt\|already_conversation` |
| **Apply chat template** | `scripts/apply_chat_template.py` | `decontaminated.jsonl`, `templated.jsonl`, `--template chatml\|llama` |

Chat template is **defined and documented** in [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md) and **applied** by `apply_chat_template.py`.
