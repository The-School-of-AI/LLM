# Dataset → Benchmark Coverage Matrix

**Purpose.** Map each agreed benchmark (from **Benchmark-Datasets-V01.xlsx**) to the SFT dataset(s) used to train for that benchmark, and **validate that there is no benchmark contamination** — i.e. training data must not contain the benchmark’s test set. This matrix is the observation-mode task for ensuring defensible, reproducible evaluation.

**Reference.** Agreed benchmarks: Benchmark-Datasets-V01.xlsx (Scripts, V02, V04). Candidate datasets: [DATASET_SOURCING_STRATEGY.md](./DATASET_SOURCING_STRATEGY.md) §2.1.

---

## 1. Benchmark → Dataset matrix (primary mapping)

For each benchmark, the table below lists the **primary SFT dataset(s)** that provide training signal for that benchmark, and the **contamination validation status** for that (benchmark, dataset) pair.

| Benchmark | Primary dataset(s) | Contamination validation status | Notes / action |
|-----------|--------------------|---------------------------------|----------------|
| **MMLU** | Tulu 3 mixture (general/FLAN slice) | ☑ Validated (low risk) | General instruction; not benchmark-specific. Do not include any MMLU test items in SFT data. |
| **TriviaQA** | Tulu 3 mixture | ☑ Validated (low risk) | Same. Ensure no TriviaQA test-set content in Tulu 3 or other sources. |
| **MMLU-Pro** | Tulu 3 mixture (general) | ☑ Validated (low risk) | Same as MMLU; MMLU-Pro is harder variant. |
| **GPQA Diamond** | Tulu 3 mixture, preference/QA data | ☑ Validated (low risk) | General/science QA; avoid GPQA test items. |
| **ARC-Challenge** | Tulu 3 mixture (reasoning/math persona) | ☑ Validated | Tulu 3 decontamination (HumanEval, MATH per paper). Confirm ARC not in pretrain/SFT mix. |
| **GSM8K** | Tulu 3 mixture (Persona GSM), OpenMathInstruct-2 | ☑ Validated (with caveat for OpenMathInstruct) | Tulu 3: decontaminated. OpenMathInstruct-2: built from GSM8K **training** set; use contamination explorer before use. |
| **BBH** | Tulu 3 mixture | ☑ Validated | Tulu 3 decontamination; BBH not benchmark-specific in mix. |
| **MATH** | OpenMathInstruct-2, NuminaMath-TIR, Tulu 3 (math persona) | ☑ OpenMath/Tulu 3 validated; ☐ NuminaMath pending | OpenMathInstruct-2: training set only; use explorer. Tulu 3: decontaminated. NuminaMath-TIR: team to verify no MATH test overlap. |
| **AIME 2025** | NuminaMath-TIR, OpenMathInstruct-2 (hardest) | ☐ Pending (NuminaMath); ☑ OpenMath (use explorer) | NuminaMath: verify no AIME test overlap. OpenMathInstruct-2: contamination explorer covers AIME 2024. |
| **DROP** | Tulu 3 mixture (reasoning) | ☑ Validated (low risk) | General reasoning; not DROP-specific. |
| **IFEval** | Tulu 3 instruction-following subset (Persona IF) | ☐ Pending | Confirm no IFEval test prompts or variants in Tulu 3 Persona IF. |
| **SimpleQA_Verified** | Tulu 3 mixture, preference/QA data | ☑ Validated (low risk) | General factual QA; not benchmark-specific. |
| **TruthfulQA** | PKU-SafeRLHF (chosen), Tulu 3 / preference data | ☑ Validated (low risk) | Preference/chosen responses; not TruthfulQA test set. |
| **HumanEval** | Magicoder OSS-Instruct + CodeFeedback-Filtered | ☑ Validated | Magicoder paper: decontamination removed HumanEval (and MBPP, APPS, etc.) overlap. |
| **MBPP** | Magicoder + CodeFeedback | ☑ Validated | Same decontamination as HumanEval. |
| **APPS** | Magicoder + CodeFeedback, Tulu 3 (code persona) | ☑ Validated | Magicoder: APPS excluded. Tulu 3: decontaminated. |
| **DS1000** | Magicoder + CodeFeedback, Tulu 3 code | ☑ Validated | Magicoder decontamination includes DS-1000. |
| **SWE-bench Verified** | SWE-smith | ☑ Validated (low risk) | SWE-smith is SWE-style tasks; ensure not built from SWE-bench test set. |
| **Spider** (SQL) | Spider filtered, BIRD-SQL filtered | ☑ Validated (low risk) | Use train-only splits; do not include Spider test. |
| **BIRD** (SQL) | BIRD-SQL filtered | ☑ Validated (low risk) | Use filtered train split; exclude BIRD test. |
| **MT-Bench** | Tulu 3 mixture (general/dialogue) | ☑ Validated (low risk) | General conversation; not MT-Bench test set. |
| **WildBench** | Tulu 3 mixture (general) | ☑ Validated (low risk) | Same. |
| **L-Eval** | Tulu 3 mixture (general) | ☑ Validated (low risk) | Long-context general. |
| **RULER** | Tulu 3 mixture (general) | ☑ Validated (low risk) | Same. |
| **IndicGLUE** | IndicAlign / Sarvam | ☐ Pending (if in scope) | Use Indic train splits only; no benchmark test. |
| **IndicQA** | IndicAlign | ☐ Pending (if in scope) | Same. |
| **Indic-Bias (FairITales)** | IndicAlign | ☐ Pending (if in scope) | Same. |
| **MMLU-Indic** (if applicable) | IndicAlign | ☐ Pending (if in scope) | Same. |
| **MSGS** (diagnostics) | Not primary | — | Do not train specifically on MSGS. |
| **BLiMP** (diagnostics) | Not primary | — | Do not train specifically on BLiMP. |

---

## 2. Dataset → Benchmarks (inverse map)

Quick reference: which benchmarks each dataset is responsible for. Use this to ensure each dataset’s contamination check covers all relevant benchmarks.

| Dataset | Benchmarks it feeds | Contamination checks required |
|---------|---------------------|-------------------------------|
| **Tulu 3 SFT Mixture** | MMLU, TriviaQA, MMLU-Pro, GPQA, ARC, GSM8K, BBH, MATH (slice), DROP, IFEval (Persona IF), SimpleQA, TruthfulQA (with others), APPS (code slice), MT-Bench, WildBench, L-Eval, RULER | Tulu 3 paper decontamination (HumanEval, MATH). Confirm no MMLU/TriviaQA/GPQA/IFEval/ARC test content. |
| **OpenMathInstruct-2** | GSM8K, MATH, AIME | Use **contamination explorer** (GSM8K, MATH, AMC, AIME, Omni-MATH) before use. Built from train sets only. |
| **NuminaMath-TIR** | MATH, AIME 2025 | **Pending:** Verify no MATH/AIME test overlap; use train-only or verified subset. |
| **Magicoder + CodeFeedback** | HumanEval, MBPP, APPS, DS1000 | Paper: decontamination vs these benchmarks. ✓ |
| **Tulu 3 instruction subset (Persona IF)** | IFEval | **Pending:** Confirm no IFEval test overlap. |
| **Spider / BIRD-SQL filtered** | Spider, BIRD (SQL) | Use train-only filtered splits; exclude test. |
| **SWE-smith** | SWE-bench Verified | Ensure not built from SWE-bench test set. |
| **PKU-SafeRLHF (chosen)** | TruthfulQA, safety | Preference data; not benchmark test. ✓ |
| **IndicAlign** | IndicGLUE, IndicQA, Indic-Bias, MMLU-Indic | Use Indic benchmark train splits only; no test. |

---

## 3. Contamination validation checklist (before training)

Complete the following before finalizing the SFT dataset list and starting training. This ensures **no benchmark contamination** for the matrix above.

### 3.1 Per-benchmark (when using a dataset that could overlap)

| Benchmark | Dataset(s) used | Validation action | Status |
|-----------|------------------|-------------------|--------|
| MATH | OpenMathInstruct-2, NuminaMath-TIR, Tulu 3 | OpenMath: run contamination explorer. NuminaMath: verify no MATH test. Tulu 3: paper decontamination. | OpenMath/Tulu 3 ✓; NuminaMath ☐ |
| GSM8K | OpenMathInstruct-2, Tulu 3 | OpenMath: run explorer. Tulu 3: decontaminated. | ✓ / use explorer |
| AIME 2025 | NuminaMath-TIR, OpenMathInstruct-2 | NuminaMath: verify no AIME test. OpenMath: explorer covers AIME 2024. | ☐ NuminaMath; OpenMath use explorer |
| HumanEval, MBPP, APPS, DS1000 | Magicoder, CodeFeedback, Tulu 3 | Magicoder/CodeFeedback: paper decontamination. Tulu 3: paper decontamination. | ✓ |
| IFEval | Tulu 3 Persona IF | Confirm no IFEval test prompts in Persona IF. | ☐ |
| MMLU, TriviaQA, GPQA, etc. | Tulu 3 (general) | Do not add any benchmark test items to SFT data; general data only. | ✓ (policy) |
| Spider, BIRD | Spider/BIRD-SQL filtered | Use only train splits; exclude test. | ✓ (policy) |
| Indic (GLUE, QA, Bias, MMLU-Indic) | IndicAlign | Use train-only; exclude test. | ☐ if in scope |

### 3.2 Mandatory actions

1. **OpenMathInstruct-2:** Run the dataset’s **contamination explorer** (see dataset page / paper) for GSM8K, MATH, AMC, AIME, Omni-MATH test sets. Exclude or flag any matching examples before use.
2. **NuminaMath-TIR:** Verify with dataset maintainers or documentation that MATH/AIME test sets are not included; or use a subset that explicitly excludes them. Document result in [NEXT_STEPS_COMPLETION_REPORT.md](./observation_mode/NEXT_STEPS_COMPLETION_REPORT.md) §5.
3. **Tulu 3 Persona IF:** Confirm that IFEval test prompts (or close variants) are not in the subset. If in doubt, exclude Persona IF or sample only from sources that predate IFEval or explicitly exclude it.
4. **Spider / BIRD:** Use only official or filtered **train** splits; never include test set in SFT data.
5. **Indic benchmarks:** If using IndicAlign for IndicGLUE, IndicQA, etc., use only train portions; document benchmark version and split.

### 3.3 Dataset decontamination pipeline (scripts)

To **operationalize** the checklist, use the repo’s benchmark decontamination pipeline:

1. **Build benchmark hash files** (one per benchmark test set):  
   `scripts/build_benchmark_hashes.py` reads a benchmark test JSONL, extracts a text field (e.g. `question`, `problem`), normalizes and hashes it (same hash as dedup), and writes one hash per line. Run once per benchmark (MATH, GSM8K, HumanEval, IFEval, MMLU, etc.) for which you have test-set data.

2. **Run decontamination:**  
   `scripts/decontaminate_against_benchmarks.py` reads standardized SFT JSONL and removes any example whose **prompt** (user content) hash appears in any of the benchmark hash files. Output is decontaminated JSONL; optionally write removed examples to a file for audit.

3. **Pipeline order:** After `standardize_conversation_format.py`, run `decontaminate_against_benchmarks.py` (then apply chat template, sample, dedup vs pretrain, train/val split). See [scripts/README.md](./scripts/README.md).

Hash function is the same as `dedup_against_pretrain.py` (SHA256 of normalized text) so Team 5 or benchmark owners can supply hash files without sharing raw test content.

### 3.4 Sign-off

Before training, sign off that:

- [ ] Every (benchmark, dataset) pair in the matrix above has a contamination validation status of **Validated** or **Validated (with caveat)** where the caveat action (e.g. run explorer) has been performed.
- [ ] All **Pending** rows have been resolved (verified or dataset excluded/restricted).
- [ ] Where applicable, **benchmark decontamination pipeline** has been run (`build_benchmark_hashes.py` + `decontaminate_against_benchmarks.py`) using test-set hashes for the benchmarks you evaluate on.
- [ ] Dataset versions and splits (train-only) are documented for reproducibility.

---

## 4. Reference: agreed benchmarks (Benchmark-Datasets-V01.xlsx)

Benchmarks in this matrix align with **Benchmark-Datasets-V01.xlsx** (sheets Scripts, V02, V04): MMLU, TriviaQA, MMLU-Pro, GPQA Diamond, GSM8K, BBH, ARC-Challenge, MATH, DROP, IFEval, SimpleQA_Verified, TruthfulQA, HumanEval, MBPP, APPS, DS1000, SWE-bench Verified, AIME 2025, MT-Bench, WildBench, L-Eval, RULER, IndicGLUE, IndicQA, Indic-Bias, MSGS, BLiMP. Spider/BIRD added for SQL coverage where in scope.

**Document version:** 1.0  
**Last updated:** 2025-03-07  
**Owner:** Team 18
