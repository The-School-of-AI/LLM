# Dataset Sourcing Strategy — Team 18 SFT  
## Comprehensive Report

**Purpose.** This report documents (1) the mandatory study of state-of-the-art post-training methods, (2) the observation-mode phase before any training—candidate SFT dataset mapping, benchmark coverage mapping, dry-run evaluations, and pipeline validation—and (3) measures to avoid benchmark contamination. Method and data choices are to be informed by this study and these mappings; no weight updates are permitted until observation mode is complete and the pipeline is validated.

**Reference.** Agreed benchmarks are documented in **Benchmark-Datasets-V01.xlsx** (sheets: V01-Archived, V02, V04, Scripts, Scripts-ongoing). This report aligns candidate datasets and coverage to that benchmark set.

---

# Part 1 — Study: State-of-the-Art Post-Training (Mandatory)

Before selecting methods or finalizing datasets, the team must study the following. Findings below summarize current practice and pitfalls; they must **inform method choice**, not drive ad-hoc experimentation.

---

## 1.1 DeepSeek post-training strategies

**Summary.** DeepSeek uses a multi-stage pipeline combining reinforcement learning and supervised fine-tuning. Key points relevant to SFT data strategy:

- **RL-first path (DeepSeek-R1):** Large-scale RL (GRPO) can be applied directly to a base model and yield large gains on reasoning (e.g. AIME 2024: 15.6% → 71.0%). This shows that strong reasoning can be incentivized without SFT-first, but it does not replace the need for high-quality SFT data when an SFT stage is used.
- **SFT stages:** When SFT is used, it appears in two roles: (1) improving response quality and consistency, (2) safety alignment and behavioral stability. Data for these stages is curated for clear provenance and task coverage.
- **Distillation:** Smaller models (e.g. 1.5B–70B) are produced via distillation from larger RL/SFT models. SFT data quality and diversity matter for both the main model and downstream distillation.
- **Implication for dataset strategy:** Prioritize data with clear provenance, task diversity aligned to target benchmarks, and a mix that supports both capability (e.g. math, code, instruction following) and stability/safety. Mechanically verifiable data (code with tests, math with answers) supports reproducible gains and aligns with DeepSeek-style emphasis on measurable reasoning.

---

## 1.2 Qwen / Gemini alignment methods

**Qwen.** Mixture-of-Instructions (MoI), diverse system prompts, and scalable data generation (e.g. AutoIF with code-verifiable quality) are used. Findings: (1) System-level instruction diversity improves multi-task alignment. (2) Transforming quality checks into code verification (AutoIF) scales and improves SFT and preference-based methods. (3) SFT alone can exhibit verbosity bias; preference-based or length-aware techniques can complement it. **Implication:** Prefer instruction data that is structured (system/user/assistant), verifiable where possible, and drawn from diverse task definitions to avoid narrow overfitting.

**Gemini (Google).** Post-training uses pedagogical instruction-following in mixture form, with system-level instructions describing desired behavior. Tuning data is integrated into post-training mixtures for robustness. Knowledge distillation is used for smaller models. **Implication:** Align SFT data format with a single, documented chat template (e.g. ChatML or Llama-style); include clear system-instruction variety; ensure data is suitable for mixture-based training rather than single-domain batches.

---

## 1.3 Recent SFT and lightweight RL-style work

- **Unified SFT + RLHF:** Intuitive Fine-Tuning (IFT) and similar work treat SFT as a special case of RLHF, with potential for better optimization. For a LoRA/QLoRA-only SFT phase, the takeaway is that **data quality and reward structure implicit in the data** (e.g. correct vs incorrect code, strict instruction adherence) matter as much as the training algorithm.
- **Preference-based methods:** DPO, GRPO, and related methods are often applied after or alongside SFT. Preference data (e.g. chosen/rejected pairs) can address verbosity and alignment; for a data-sourcing report, this motivates including **structured preference or “chosen-only” high-quality responses** where appropriate (Tier 2 in the verifiability framing), without requiring the team to run RL in observation mode.
- **Distillation from strong reasoners:** DeepSeek-R1–style distillation and “reasoning trace” datasets suggest that **curated reasoning data** (e.g. step-by-step math, code with tests) is a high-lever way to transfer capability. Dataset mapping should explicitly tag which candidates provide such data.

---

## 1.4 Stability and regression pitfalls

- **Catastrophic forgetting:** SFT can degrade general capabilities when data is narrow or when distribution shift is large. Mitigations in the literature: (1) **Mix general and domain-specific data** (rehearsal); (2) **Answer-style diversification** so the model does not overfit to one response style; (3) **Regularization** (e.g. RegLoRA) over critical parameters; (4) **Mixup-style regularization** to smooth confidence and reduce overfitting to hard examples.
- **Regression in benchmarks:** Overfitting to a single benchmark or to data that overlaps with test sets causes **inflated scores and regression elsewhere**. This directly motivates: (1) **Benchmark contamination controls** (see Part 3), and (2) **Balanced dataset mix** aligned to the full agreed benchmark set, not a single task.
- **Implication for observation mode:** Before training, validate that the **pipeline** (format, template, loss masking, padding) is correct and that **dry-run evaluations** are possible on a subset of benchmarks. This reduces the risk of discovering format or evaluation bugs only after weight updates.

---

## 1.5 Findings that must inform method and data choice

| Finding | Implication for dataset strategy |
|--------|----------------------------------|
| SoTA pipelines use multi-stage SFT/RL and clear data provenance | Prefer canonical, well-documented datasets; document versions and subsampling. |
| Verifiable data (code tests, math answers, SQL results) supports reproducible gains | Weight candidate set toward mechanically verifiable data (Tier 1). |
| System-instruction diversity and format consistency matter (Qwen, Gemini) | Standardize to one chat template; include diverse system prompts in candidate set. |
| SFT can cause forgetting and verbosity bias | Plan for mixed, balanced data; consider preference or length-aware signals later. |
| Contamination resurfaces in SFT and inflates metrics | Explicit contamination avoidance is mandatory (Part 3). |
| Pipeline and eval must be validated before training | Observation mode: dry-run evals and pipeline validation with no weight updates. |

---

# Part 2 — Observation Mode (Early Phase)

Before any training (no weight updates), the team must: (1) map candidate SFT datasets, (2) map benchmark coverage, (3) run dry-run evaluations, and (4) validate pipeline correctness. This section provides the mapping and reasoning; execution (dry-runs, pipeline validation) uses the scripts and checklist in this repo.

---

## 2.1 Mapping candidate SFT datasets

Candidates are listed with **reasoning** tied to the agreed benchmarks (Benchmark-Datasets-V01.xlsx) and to the study (Part 1). Verifiability tiers are used to prioritize data where correctness is mechanically checkable (Tier 1), then model-judged but structured (Tier 2), then subjective (Tier 3).

### Tier 1 — Mechanically verifiable (target 60–70% of mix)

| Candidate dataset | Rationale | Relevant benchmarks | Notes |
|------------------|-----------|----------------------|-------|
| **Tulu 3 SFT Mixture** (subsample) | 19-source mixture with code (Evol CodeAlpaca, Persona Python), math (NuminaMath-TIR, Persona MATH/Algebra/GSM), and instruction-following (Persona IF). Clear provenance; supports MATH, GSM8K, HumanEval-style code, IFEval. | MATH, GSM8K, BBH, HumanEval, APPS, IFEval | Subsample to stay within budget (e.g. 80–100K); prefer code/math/IF-heavy slices. |
| **OpenMathInstruct-2** (subsample) | Competition-style math with ground-truth answers; directly targets MATH and harder math. | MATH, AIME 2025 | Subsample hardest problems (e.g. ~50K) for efficiency. |
| **Magicoder OSS-Instruct + CodeFeedback-Filtered** | Code with test suites; mechanically verifiable. Targets HumanEval, MBPP, APPS. | HumanEval, MBPP, APPS | ~50K from combined 75K+157K. |
| **Tulu 3 instruction-following subset (Persona IF)** | IFEval-style constraints; format/length constraints. | IFEval | ~30K. |
| **Spider + BIRD-SQL filtered** | SQL with expected results; mechanically verifiable. | (SQL coverage if in scope) | ~15K; use filtered versions to avoid low quality. |
| **NuminaMath-TIR** | AIME-level competition math; ground-truth. | AIME 2025, MATH | 64K available; subsample as needed. |
| **SWE-smith** (subsample) | SWE-style tasks with testability. | SWE-bench Verified | ~10K. |

### Tier 2 — Model-judged but structured (target 20–30%)

| Candidate dataset | Rationale | Relevant benchmarks | Notes |
|------------------|-----------|----------------------|-------|
| **Preference/chosen-only (e.g. PKU-SafeRLHF chosen)** | Structured preference or chosen responses; supports safety and alignment without requiring live RL. | TruthfulQA, safety-related evals | Subsample ~20K. |
| **IndicAlign** (subsample) | Indic-language instruction data; structured. | IndicGLUE, IndicQA, Indic-Bias | ~25K if Indic benchmarks are in scope. |
| **DeepSeek-R1 distilled / reasoning traces** | Reasoning-step data from strong models; supports CoT and transfer. | MATH, GSM8K, BBH, AIME | TBD size; validate license and overlap with benchmarks. |

### Tier 3 — Subjective quality (target ~10%)

| Candidate dataset | Rationale | Relevant benchmarks | Notes |
|------------------|-----------|----------------------|-------|
| **General instruction subset (e.g. from Tulu 3: FLAN, WildChat, Aya, OpenAssistant)** | Dialogue and general helpfulness; avoids overfitting to only math/code. | MT-Bench, WildBench, broad coverage | Keep small share; use cleaned/general slices. |

**Total scale.** The above can sum to ~300–400K samples before subsampling. For a **50–100K SFT budget**, the team should select a subset (e.g. Tulu 3 subsample + OpenMathInstruct-2 + Magicoder/CodeFeedback + Tulu 3 IF subset as core, then add SQL / safety / Indic / SWE / reasoning as needed) and document the exact mix and subsampling in the pipeline.

---

## 2.2 Mapping benchmark coverage

Agreed benchmarks (Benchmark-Datasets-V01.xlsx) are grouped below. The **coverage map** states which candidate dataset(s) and tier(s) support each category so that gaps and over-representation are explicit.

| Benchmark category | Benchmarks (from xlsx) | Primary candidate datasets | Tier | Coverage note |
|--------------------|------------------------|----------------------------|------|----------------|
| Knowledge | MMLU, TriviaQA, MMLU-Pro, GPQA Diamond | Tulu 3 (general/FLAN slice), preference/QA data | 2–3 | Broader knowledge from mixed and preference data; avoid training on benchmark questions. |
| Reasoning | GSM8K, BBH, ARC-Challenge, MATH, DROP | Tulu 3 (math persona), OpenMathInstruct-2, NuminaMath-TIR, reasoning traces | 1–2 | Strong Tier 1 coverage for MATH/GSM8K; BBH/ARC via Tulu 3 and reasoning data. |
| Instruction | IFEval, SimpleQA_Verified, TruthfulQA | Tulu 3 Persona IF, Tulu 3 mixture, preference/safety data | 1–2 | IFEval explicitly covered by IF subset; SimpleQA/TruthfulQA via general + preference. |
| Code | HumanEval, MBPP, APPS, DS1000, SWE-bench Verified | Magicoder + CodeFeedback, Tulu 3 code persona, SWE-smith | 1 | Direct Tier 1 coverage; ensure no test-set leakage (see Part 3). |
| Math (elite) | AIME 2025 | OpenMathInstruct-2, NuminaMath-TIR | 1 | AIME-level data from NuminaMath-TIR and hardest OpenMathInstruct. |
| Conversation / long-context | MT-Bench, WildBench, L-Eval, RULER | Tulu 3 general slice (Tier 3) | 3 | Light Tier 3 to avoid regression on dialogue. |
| Indic | IndicGLUE, IndicQA, Indic-Bias | IndicAlign | 2 | Only if Indic benchmarks are in scope. |
| Diagnostics | MSGS, BLiMP | Not primary; avoid skewing data toward diagnostics | — | Optional; do not over-weight training data for these. |

**Reasoning.** (1) Tier 1–heavy coverage for code and math ensures that gains on HumanEval, APPS, MATH, GSM8K, AIME are **verifiable and reproducible**. (2) IFEval is explicitly targeted by an instruction-following subset to support “real gains” on constraint following. (3) Knowledge benchmarks are supported by mixed and preference data without requiring benchmark-specific training data. (4) Conversation and long-context get a small share to limit regression and contamination risk. (5) Any candidate that might overlap with a benchmark (e.g. MATH, HumanEval, GSM8K) must pass contamination checks (Part 3).

---

## 2.3 Dry-run evaluations

- **Objective.** Before weight updates, run evaluations on a **fixed set of benchmarks** (from Benchmark-Datasets-V01.xlsx) using the **base model** (no SFT). This establishes a baseline and validates that the evaluation harness and metrics run correctly.
- **Scope.** Use the same benchmarks (and versions) that will be used post-SFT. Prefer the framework already agreed (e.g. OLMES per Scripts sheet) so that post-SFT comparisons are comparable.
- **Constraint.** No training or weight updates in this phase. Dry-runs are **evaluation-only** to confirm pipeline and baseline.

---

## 2.4 Pipeline validation

- **Objective.** Ensure that the full data pipeline—standardize format → apply chat template → sample for quality review → dedup (when hashes available) → train/val split—produces outputs that are correct for training (e.g. labels, loss masking, padding).
- **Checks.** (1) Run scripts in order (see `scripts/README.md`) on a small sample. (2) Run `verify_loss_masking.py` with the chosen tokenizer and confirm that only assistant tokens receive non–ignore_index labels and that padding is right-side with -100. (3) Confirm that the training collator (in training code) uses the same loss masking and padding policy (checklist items 8–9 in SFT_DATA_CHECKLIST_7.1.md).
- **Constraint.** No weight updates. Validation is to ensure that when training is enabled, the data and loss are correct.

---

# Part 3 — Benchmark contamination avoidance

Contamination inflates benchmark scores and undermines claims of real capability gains. SFT can **resurface** leaked information from pre-training, so prevention and checks are mandatory.

---

## 3.1 How contamination happens

- **Direct leakage:** Benchmark questions or answers appear in training data (e.g. from web scrape, GitHub, papers).
- **Indirect:** Paraphrased, translated, or semantically equivalent content; n-gram overlap checks alone are insufficient.
- **Temporal and distributional:** Same underlying patterns or sources in both train and test.

For SFT, any candidate dataset that might contain or be derived from benchmark material (MMLU, MATH, HumanEval, GSM8K, IFEval, etc.) must be treated as high risk until checked.

---

## 3.2 Mitigation before training

| Measure | Action |
|--------|--------|
| **Exclude benchmark-derived data** | Do not include any dataset that is built from or includes benchmark test sets. Prefer datasets with explicit “train-only” splits or clear exclusion policies. |
| **Dedup against benchmark content** | Where feasible, maintain a set of hashes (or normalized fingerprints) of benchmark items (per benchmark) and exclude SFT examples that match (exact or near-duplicate). Coordinate with Team 5 / benchmark owners for canonical test IDs or hashes. |
| **Provenance and versioning** | Use only canonical, versioned datasets (e.g. Hugging Face with commit/dataset card). Document dataset version and date so that future benchmark updates can be checked for overlap. |
| **Subsampling** | When subsampling, avoid any criterion that could bias toward benchmark-like items (e.g. do not select “hard” examples by similarity to a benchmark). Prefer random or stratified sampling by source/task. |

---

## 3.3 High-risk candidates and checks

- **Math (MATH, GSM8K, AIME):** OpenMathInstruct-2, NuminaMath-TIR, and any “competition math” set may overlap with public benchmarks. Verify that they are not built from MATH/GSM8K/AIME test sets; use train-only splits where available.
- **Code (HumanEval, MBPP, APPS):** Magicoder, CodeFeedback, and code-heavy Tulu 3 slices. Ensure no test-set problems are included; prefer datasets that explicitly exclude HumanEval/MBPP/APPS test sets.
- **IFEval / instruction:** Tulu 3 Persona IF and similar. Confirm no overlap with IFEval test prompts or variants.
- **Knowledge (MMLU, TriviaQA, etc.):** Avoid any SFT set built from or including MMLU/TriviaQA/GPQA test items. General instruction data is lower risk if it is not benchmark-specific.

If any candidate cannot be verified as clean, exclude it or restrict to a subset that has been verified.

---

## 3.4 Post-training and reporting

- **Do not train on benchmark test sets.** Report only on held-out benchmark splits.
- **Document data cutoff and versions.** In the final report, state SFT dataset names, versions, and subsampling so that reproducibility and future contamination checks are possible.
- **Intervention-based checks (optional).** Literature suggests that intervention-based detection (e.g. observing model behavior when deliberately fine-tuning on test data) can reveal paraphrased or indirect contamination; consider such checks if resources allow.

---

# Summary and next steps

1. **Study (Part 1).** DeepSeek, Qwen, and Gemini post-training strategies, plus SFT/RL literature and stability pitfalls, are summarized above. These findings must **inform** method choice and dataset mix (verifiability tiers, balance, format, contamination).
2. **Observation mode (Part 2).** Before any training: **map** candidate SFT datasets (with reasoning and tier), **map** benchmark coverage to those candidates, **run** dry-run evaluations on the base model, and **validate** the data and training pipeline (loss masking, padding). No weight updates in this phase.
3. **Contamination (Part 3).** Avoid benchmark-derived data; dedup against benchmark content where feasible; use only versioned, canonical datasets; verify high-risk candidates (math, code, IFEval, knowledge); document datasets and versions for reporting.

After observation mode is complete and the pipeline is validated, the team can proceed to finalize the dataset list (within the candidate set and contamination constraints), obtain approvals, and run training.

**How to perform these steps:** Use the **Observation Mode Runbook** and **Next Steps Completion Report** in `sft_data/observation_mode/`:
- **[OBSERVATION_MODE_RUNBOOK.md](observation_mode/OBSERVATION_MODE_RUNBOOK.md)** — Step-by-step actions for Study (§1), Mapping (§2.1–2.2), Dry-run evals (§2.3), Pipeline validation (§2.4), and Contamination checks (§3). Includes pipeline validation commands and optional `run_pipeline_validation.sh`.
- **[NEXT_STEPS_COMPLETION_REPORT.md](observation_mode/NEXT_STEPS_COMPLETION_REPORT.md)** — Template to document completion of each step (study sign-off, mapping outcomes, dry-run results, pipeline validation, contamination checks, final status). Fill this report as you complete the runbook; final status determines readiness to finalize the dataset list.

The checklist in SFT_DATA_CHECKLIST_7.1.md and the scripts in `scripts/README.md` support execution.

---

## References

- DeepSeek-R1 / GRPO and multi-stage SFT+RL pipeline (e.g. arxiv 2501.12948; reproduction guides).
- Qwen Mixture-of-Instructions, AutoIF; IFT (SFT+RLHF unification).
- Gemini alignment (Gemma 2 report; LearnLM; Google alignment docs).
- Contamination: “The Impact of Post-training on Data Contamination”; “Test Set Contamination: The Silent Killer of LLM Benchmarks”; mitigation and intervention-based detection (arxiv 2503.16402, OpenReview).
- Stability: catastrophic forgetting and SFT (e.g. RegLoRA, SFTMix, rehearsal/mixup).
- **Benchmark-Datasets-V01.xlsx** — agreed benchmarks (V01–V04, Scripts, Scripts-ongoing).
- [SFT_DATA_CHECKLIST_7.1.md](./SFT_DATA_CHECKLIST_7.1.md) — pipeline and checklist.
- [scripts/README.md](./scripts/README.md) — script order and usage.
