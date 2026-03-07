# Next Steps Completion Report — Team 18 SFT

This document records the **performance and results** of the next steps from the Dataset Sourcing Strategy (Study, Observation mode, Contamination). Fill each section as the team completes the corresponding runbook steps. Keep this report updated so that observation mode completion and readiness to finalize the dataset list are clearly documented.

**Runbook:** [OBSERVATION_MODE_RUNBOOK.md](./OBSERVATION_MODE_RUNBOOK.md)  
**Strategy:** [DATASET_SOURCING_STRATEGY.md](../DATASET_SOURCING_STRATEGY.md)

---

## Execution summary (as of 2025-03-07)

| Step | Status | Finding |
|------|--------|--------|
| **1. Study (Part 1)** | Done | Part 1 read; all 6 findings in Table 1.5 checked against mapping and pipeline; implications reflected. |
| **2. Candidate mapping** | Done | Accepted as in strategy doc (§2.1); no changes. |
| **2. Benchmark coverage** | Done | Checked against Benchmark-Datasets-V01.xlsx; all categories covered; accepted. |
| **3. Dry-run evaluations** | Pending | Requires base model and eval harness (e.g. OLMES). Blocking observation-mode complete. |
| **4. Pipeline validation** | Done | All four steps Pass (standardize → template → split → verify_loss_masking); outputs verified. |
| **5. Contamination checks** | Partial | OpenMathInstruct-2 (use explorer), Tulu 3 (decontamination per paper), Magicoder (benchmark exclusion per paper) documented. NuminaMath-TIR and IFEval overlap pending team verification. |
| **6. Final status** | Not complete | Observation mode complete = No. Ready to finalize dataset list = No. Blockers: dry-run, contamination verification. |

**Next:** Run dry-run evals; complete contamination verification for NuminaMath-TIR and IFEval; then set §6 to complete and proceed to finalize dataset list.

---

## 1. Study completion (Part 1)

| Item | Status | Date | Notes |
|------|--------|------|-------|
| Part 1 (sections 1.1–1.5) read and summarized | ☑ Done | 2025-03-07 | |
| Findings table (1.5) implications checked against mapping and pipeline | ☑ Done | 2025-03-07 | All 6 rows cross-checked |
| Study sign-off | ☑ Done | 2025-03-07 | Observation mode execution |

**Summary:**  
Part 1 (DeepSeek, Qwen, Gemini, SFT+RL, stability) was read and summarized. **Findings vs mapping/pipeline:** (1) SoTA provenance → candidate list uses canonical HF/official sources and versioning is required in completion report §5. (2) Verifiable data → Tier 1 (60–70%) is code/math/SQL/IFEval; Tier 2/3 balanced per strategy. (3) Single chat template → CHAT_TEMPLATE.md and apply_chat_template.py (chatml/llama) used in pipeline. (4) Forgetting/verbosity → mixed balanced data and optional preference signals in Tier 2. (5) Contamination → Part 3 and §5 contamination checks mandatory. (6) Pipeline and eval before training → observation mode runbook and this completion report. Study accepted; implications reflected in candidate mapping, benchmark coverage, and contamination plan.

---

## 2. Mapping outputs (Part 2.1 and 2.2)

### 2.1 Candidate dataset mapping

| Item | Status | Date | Notes |
|------|--------|------|-------|
| Candidate mapping (Tier 1/2/3) reviewed | ☑ Done | 2025-03-07 | |
| Outcome | ☑ Accepted as in strategy doc ☐ Updated (see notes) | | |

**Notes:**  
Accepted as in DATASET_SOURCING_STRATEGY.md §2.1. Tier 1: Tulu 3 subsample, OpenMathInstruct-2, Magicoder+CodeFeedback, Tulu 3 Persona IF, Spider+BIRD-SQL, NuminaMath-TIR, SWE-smith. Tier 2: PKU-SafeRLHF chosen, IndicAlign, DeepSeek-R1 distilled. Tier 3: General instruction (e.g. Tulu 3 FLAN/WildChat/Aya). No changes.

### 2.2 Benchmark coverage map

| Item | Status | Date | Notes |
|------|--------|------|-------|
| Coverage map checked against Benchmark-Datasets-V01.xlsx | ☑ Done | 2025-03-07 | |
| Outcome | ☑ Accepted ☐ Gaps/updates (see notes) | | |

**Notes:**  
Strategy §2.2 coverage map checked against Benchmark-Datasets-V01.xlsx (Scripts, V02, V04). All listed benchmark categories (Knowledge, Reasoning, Instruction, Code, Math elite, Conversation, Indic, Diagnostics) have primary candidate datasets and tier assigned. No gaps; accepted as-is.

---

## 3. Dry-run evaluations (Part 2.3)

| Item | Value |
|------|--------|
| Base model (name and version) | *Pending* — to be set when dry-run is executed (e.g. Gemma 3 1B, Llama 3.2 1B, or agreed base per training plan) |
| Evaluation framework (e.g. OLMES) and version | *Pending* — use framework from Benchmark-Datasets-V01.xlsx Scripts sheet (e.g. OLMES) |
| Date of dry-run | *Not yet run* |

**Benchmarks run (list):**  
*To be filled when run.* Per Benchmark-Datasets-V01.xlsx: MMLU, TriviaQA, MMLU-Pro, GPQA Diamond, GSM8K, BBH, ARC-Challenge, MATH, IFEval, SimpleQA_Verified, HumanEval, etc. Run at least a subset (e.g. MMLU-Pro, GSM8K, MATH, IFEval, HumanEval) to establish baseline.

**Results (fill table):**

| Benchmark | Metric | Value | Notes |
|-----------|--------|-------|--------|
| *(fill when dry-run executed)* | | | |

**Harness issues or failures (if any):**  
*None yet — dry-run not executed.*

**Confirmation:** ☐ No weight updates were performed during this step (eval-only). *When you run the dry-run, confirm eval-only and check this.*

**Blocker:** Dry-run requires (1) chosen base model, (2) installed eval harness (e.g. OLMES), (3) benchmark task names from Scripts sheet. Complete this step before marking observation mode complete.

---

## 4. Pipeline validation (Part 2.4)

| Item | Status | Notes |
|------|--------|--------|
| Standardize (standardize_conversation_format.py) | ☑ Pass | 5 conversations → standardized.jsonl |
| Apply chat template (apply_chat_template.py) | ☑ Pass | 5 examples with `text` (chatml) |
| Train/val split (train_val_split.py) | ☑ Pass | train 4, val 1 (seed 42, val_ratio 0.2) |
| Verify loss masking (verify_loss_masking.py) | ☑ Pass | Sample 5; turn counts printed |
| Training collator aligned with checklist items 8–9 | ☑ Done | Per SFT_DATA_CHECKLIST_7.1.md items 8–9 |

**Commands run (paste or reference):**  
From `sft_data/observation_mode`: `bash run_pipeline_validation.sh` (executed 2025-03-07; python3).  
Or run manually from `sft_data/scripts`: see OBSERVATION_MODE_RUNBOOK.md “Quick reference — Pipeline validation commands”.

**Errors (if any):**  
None. All four steps completed successfully.

**Output checks (verified):**  
- standardized.jsonl: 5 lines; each has `conversations` with role/content; 2 turns per example.  
- templated.jsonl: 5 lines; each has `text` (ChatML).  
- train.jsonl: 4 lines; val.jsonl: 1 line; no overlap.  
- verify_loss_masking.py: printed Example 1–4, 2 turns each; reminder for --tokenizer.

---

## 5. Contamination checks (Part 3)

**High-risk candidates (from Strategy §3.3):**

| Dataset | Risk category | Version / source | Check performed | Result | Date |
|---------|---------------|------------------|------------------|--------|------|
| OpenMathInstruct-2 | Math (MATH, AIME) | nvidia/OpenMathInstruct-2 (HF) | Built from MATH/GSM8K **training** set; paper states contamination explorer for test sets (GSM8K, MATH, AMC, AIME, Omni-MATH). Use explorer before use. | ☑ Pass (with caveat) | 2025-03-07 |
| NuminaMath-TIR | Math (MATH, AIME) | AI-MO/NuminaMath-TIR (HF) | Built from NuminaMath-CoT (~70k problems); no explicit AIME/MATH test exclusion in public docs. | ☐ Pending | Team to verify |
| Tulu 3 (math slices) | Math | allenai/tulu-3-sft-mixture | Tulu 3 paper: "prompt decontamination" for HumanEval and MATH (Section 3.2). | ☑ Pass | 2025-03-07 |
| Magicoder / CodeFeedback | Code (HumanEval, MBPP, APPS) | ise-uiuc/Magicoder-OSS-Instruct-75K etc. | Paper: decontamination removed overlap with HumanEval, HumanEval+, MBPP, MBPP+, DS-1000, APPS, GSM8K (9 samples filtered). | ☑ Pass | 2025-03-07 |
| Tulu 3 Persona IF | IFEval | (subset of Tulu 3) | Same Tulu 3 decontamination; IFEval not explicitly listed — low volume. | ☐ Pending | Team to confirm no IFEval test overlap |
| General / knowledge sets | MMLU, TriviaQA, GPQA | Various (FLAN, WildChat, Aya) | General instruction data; not benchmark-specific. Lower risk if no MMLU/TriviaQA/GPQA test items. | ☑ Pass (low risk) | 2025-03-07 |

**All candidates (for final list):**  
*Fill when final list is fixed. For each dataset used: record version/source (HF commit or date) and contamination check result (Pass/Fail/Not yet done).*

**Exclusions or restrictions:**  
None so far. Before finalizing: (1) Run OpenMathInstruct-2 contamination explorer if using it. (2) Verify NuminaMath-TIR has no AIME/MATH test overlap or use train-only subset. (3) Confirm Tulu 3 Persona IF / IFEval separation.

**Full matrix and validation checklist:** The **Dataset → Benchmark Coverage Matrix** and per-benchmark contamination validation checklist are in [DATASET_BENCHMARK_COVERAGE_MATRIX.md](../DATASET_BENCHMARK_COVERAGE_MATRIX.md). Use that document to ensure no benchmark contamination for every (benchmark, dataset) pair before training.

---

## 6. Final status

| Item | Value |
|------|--------|
| **Observation mode complete** | ☐ Yes ☑ No |
| **Ready to finalize dataset list** | ☐ Yes ☑ No |

**Blockers (if not complete):**  
1. **Dry-run evaluations not yet run** — Requires base model and eval harness (e.g. OLMES). Run per runbook Step 4 and fill §3 results.  
2. **Contamination verification pending for some candidates** — NuminaMath-TIR and Tulu 3 Persona IF / IFEval need team verification (see §5). Before training, run OpenMathInstruct-2 contamination explorer if using that dataset.

**Next actions:**  
- Run dry-run evaluations on base model (agreed benchmarks or subset); document results in §3.  
- Complete contamination verification for NuminaMath-TIR and IFEval overlap; run OpenMathInstruct-2 explorer if using.  
- Once §3 and §5 are complete and blockers cleared, set Observation mode complete = Yes and Ready to finalize dataset list = Yes; then finalize dataset list, obtain Team 5 approval, and run training.

*Once both checkboxes above are Yes, the team can proceed to finalize the dataset list (within candidate set and contamination constraints), obtain Team 5 approval, and run training.*

---

**Document version:** 1.0  
**Last updated:** 2025-03-07  
**Owner:** Team 18
