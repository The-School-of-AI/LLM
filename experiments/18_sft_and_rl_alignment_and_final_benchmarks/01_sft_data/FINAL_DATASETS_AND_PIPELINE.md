# Final Datasets and Pipeline — Team 18 SFT

Single reference for: **datasets to source** (and why), **relevant benchmarks**, **decontamination**, **scripts** (source → decontaminate → standardize → chat template), and **chat template** definition.

**Related docs:** [DATASET_SOURCING_STRATEGY.md](./DATASET_SOURCING_STRATEGY.md), [DATASET_BENCHMARK_COVERAGE_MATRIX.md](./DATASET_BENCHMARK_COVERAGE_MATRIX.md), [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md), [scripts/README.md](./scripts/README.md).

---

## 1. Final datasets — confirmed list for this 70B MoE

**Design principle:** Tulu 3 is the backbone. It already covers general chat, safety, instruction following, code, and math QA. Everything added on top fills a specific gap Tulu 3 doesn't address. No dataset is added speculatively.

### IN — the four datasets we are using

| Dataset | Count | What gap it fills | Benchmarks | Status |
|---------|-------|-------------------|------------|--------|
| **Tulu 3 SFT Mixture** | **939K (full)** | The validated 70B SFT base. Covers: general chat (WildChat, No Robots, OASST, UltraChat), safety (WildGuardMix 50K + WildJailbreak 50K + CoCoNot 11K), instruction following (Persona IF), code (Evol CodeAlpaca, Persona Python), math QA (NuminaMath-TIR 64K), FLAN, StackExchange. Don't subsample — it's already curated for 70B. | MMLU, GSM8K, MATH, IFEval, TruthfulQA, HumanEval, MT-Bench, WildBench, ARC, BBH | Decontaminated ✓ |
| **OpenThoughts3-1.2M** | **150K subsample** | The one critical gap in Tulu 3: **reasoning behavior**. R1-style long-form reasoning traces — teaches the model to reason step-by-step. 850K math + 250K code + 100K science; already includes AIME/AMC/Olympiad-level problems. Subsample proportionally: ~106K math + ~31K code + ~13K science. ⚠️ Must be mixed with Tulu 3 — SFT on reasoning traces alone degrades instruction following (arXiv:2507.00432). | MATH, AIME 2025, GSM8K, HumanEval, GPQA | Pending sourcing |
| **IndicAlign** | **50K subsample** | The model-specific gap: **Indic chat behavior**. The model understands Indic languages (from sangraha + ai-bharath pretraining) but hasn't learned to respond in them. All 14 IndicAlign languages exist in pretraining — no wasted data. Mostly machine-translated via IndicTrans2; quality is strong for major languages (Hindi, Bengali, Tamil, Telugu). | IndicGLUE, IndicQA, MMLU-Indic | Pending decontamination |
| **SWE-smith** | **25K subsample** | Real-world software engineering tasks — the one code type Tulu 3 doesn't cover. Given 22% code pretraining (StarCoder), the model has strong code foundations; SWE-style tasks will have high ROI. | SWE-bench Verified | Pending decontamination |

**Total: ~1.16M examples, 2 epochs.**

### OUT — dropped and why

| Dataset | Reason dropped |
|---------|---------------|
| **OpenMathInstruct-2** | Math over-allocation. Tulu 3 already has NuminaMath-TIR 64K; OpenThoughts3 adds 106K math reasoning traces. Total math is already ~170K on a model with 5.4% math pretraining. Adding 80K more standard math QA biases the model's behavior without proportional pretraining support. Hold for a targeted second-round SFT if post-eval shows math is weak. |
| **NuminaMath-TIR (standalone)** | Already inside Tulu 3 (64K). Pure duplication. |
| **Magicoder OSS-Instruct + CodeFeedback-Filtered** | Tulu 3 already has code (Evol CodeAlpaca, Persona Python). Add back only if post-SFT HumanEval/MBPP eval shows code is specifically weak. Both are decontaminated and ready if needed. |
| **Spider + BIRD-SQL** | No SQL in pretraining. SFT cannot teach knowledge the model doesn't have. Drop unless explicit product requirement. |
| **PKU-SafeRLHF** | DPO/preference dataset, not SFT. Hold for DPO stage. Safety already covered by Tulu 3. |

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
| OpenThoughts3-1.2M | `open-thoughts/OpenThoughts3-1.2M` (subsample ~150K: ~106K math + ~31K code + ~13K science) |
| IndicAlign | `ai4bharat/indic-align` (subsample 50K across 14 languages) |
| SWE-smith | `SWE-bench/SWE-smith` (subsample 25K) |

**On-deck (add only if post-SFT eval shows specific weakness):**

| Dataset | HF path | Add if... |
|---------|---------|-----------|
| Magicoder OSS-Instruct | `ise-uiuc/Magicoder-OSS-Instruct-75K` | HumanEval/MBPP weak post-eval |
| CodeFeedback-Filtered | `m-a-p/CodeFeedback-Filtered-Instruction` | HumanEval/MBPP weak post-eval |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | MATH/AIME weak after first SFT round |

If a dataset is not on HF or has a different ID, edit `source_datasets.py` or the config it uses. Sourcing can also be done manually via `datasets.load_dataset("org/name")` and export to JSONL.

---

## 3. Decontamination (no benchmark contamination)

We must ensure **no benchmark test-set content** appears in SFT data. **Decontamination is a blocker, not a nice-to-have** — without it, eval numbers are meaningless.

### Decontamination status

| Dataset | Status |
|---------|--------|
| Tulu 3 SFT Mixture | ✓ Done |
| OpenThoughts3-1.2M (subsample ~150K) | ❌ Pending |
| IndicAlign (50K) | ❌ Pending |
| SWE-smith (25K) | ❌ Pending |
| Magicoder OSS-Instruct *(on-deck)* | ✓ Done (ready if needed) |
| CodeFeedback-Filtered *(on-deck)* | ✓ Done (ready if needed) |
| OpenMathInstruct-2 *(on-deck)* | ❌ Not started (decontaminate before use) |

**Priority:** OpenThoughts3 contains NuminaMath-derived math problems — run 13-gram overlap check against MATH + GSM8K test sets before training.

### Decontamination method

Two complementary checks are required:

1. **Exact hash match** (current scripts) — SHA256 of normalized text. Catches verbatim copies.
2. **13-gram overlap** — Remove any SFT example where >10% of 13-grams overlap with any test-set example. Required for math datasets where problems are paraphrased but structurally identical. Run against: GSM8K, MATH, MMLU, HumanEval, IFEval, TruthfulQA, ARC, HellaSwag, WinoGrande, AlpacaEval, MT-Bench.

The existing scripts cover (1). Add an n-gram overlap pass for (2) on all math datasets before training.

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

## 8. Target data mix — 70B MoE (4 datasets, ~1.16M, 2 epochs)

This is the actual mix for this model.

| Dataset | Count | % of total | What it provides |
|---------|-------|-----------|-----------------|
| **Tulu 3 SFT Mixture** | 939K | 81% | General chat, safety, IF, code, math QA, FLAN — the complete base |
| **OpenThoughts3-1.2M subsample** | 150K | 13% | Reasoning behavior (R1-style traces): ~106K math + ~31K code + ~13K science |
| **IndicAlign** | 50K | 4% | Indic chat behavior across 14 languages |
| **SWE-smith** | 25K | 2% | Real-world software engineering tasks |
| **Total** | **~1.16M** | **100%** | |

**2 epochs. Hard limit — MoE models overfit faster than dense models.**

**Why this allocation makes sense for this specific model:**

| Pretraining domain | % of pretraining | SFT allocation | Reasoning |
|---|---|---|---|
| General web (B1) | 75% | ~81% via Tulu 3 | Tulu 3's general chat activates the broad web knowledge |
| Code (B3) | 22% | ~15% (Tulu 3 code + SWE-smith + OpenThoughts3 code traces) | High ROI — deep code foundation already exists |
| Indic (B0+B1+B2) | significant | ~4% IndicAlign | Differentiator; model understands, needs to learn to respond |
| Math (B4+B5) | 5.4% | ~13% OpenThoughts3 + Tulu 3 NuminaMath | Reasoning behavior, not raw math QA — proportional to pretraining |

**What's deliberately excluded:**
- OpenMathInstruct-2, NuminaMath-TIR standalone: math is already ~170K examples in this mix (Tulu 3's 64K + OpenThoughts3's 106K). That's proportional to 5.4% math pretraining. More would over-index.
- Magicoder + CodeFeedback: Tulu 3 has code. Add only if HumanEval/MBPP show specific weakness post-eval.
- All preference/DPO data: held for DPO stage.

---

## 9. Quick reference — script summary

| Purpose | Script | Key args |
|---------|--------|----------|
| **Source datasets** | `scripts/source_datasets.py` | `--output-dir`, `--datasets` |
| **Build benchmark hashes** | `scripts/build_benchmark_hashes.py` | `input.jsonl`, `output.txt`, `--text-field` |
| **Decontaminate vs benchmarks** | `scripts/decontaminate_against_benchmarks.py` | `standardized.jsonl`, `decontaminated.jsonl`, `--benchmark-hashes-dir` |
| **Standardize format** | `scripts/standardize_conversation_format.py` | `input.jsonl`, `standardized.jsonl`, `--format alpaca\|sharegpt\|already_conversation` |
| **Apply chat template** | `scripts/apply_chat_template.py` | `decontaminated.jsonl`, `templated.jsonl`, `--template chatml\|llama` |

Chat template is **defined and documented** in [CHAT_TEMPLATE.md](./CHAT_TEMPLATE.md) and **applied** by `apply_chat_template.py`.
