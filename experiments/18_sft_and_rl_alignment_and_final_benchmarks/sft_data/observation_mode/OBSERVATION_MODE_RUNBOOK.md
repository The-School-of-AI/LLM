# Observation Mode Runbook — Team 18 SFT

This runbook performs the **next steps** from [DATASET_SOURCING_STRATEGY.md](../DATASET_SOURCING_STRATEGY.md) (Summary and next steps). Complete each section in order; **no weight updates** until the runbook is complete and the completion report is filled.

**Reference:** [DATASET_SOURCING_STRATEGY_v1.md](../DATASET_SOURCING_STRATEGY_v1.md) (or DATASET_SOURCING_STRATEGY.md)

---

## Step 1 — Study completion (Part 1)

**Objective.** Confirm that the team has studied state-of-the-art post-training (DeepSeek, Qwen, Gemini, SFT+RL, stability/regression) and that findings inform method and data choice.

**Actions.**

1. Read **Part 1** of the Dataset Sourcing Strategy report (sections 1.1–1.5).
2. For each finding in **Table 1.5**, confirm that the corresponding implication is reflected in:
   - Candidate dataset mapping (Tier 1/2/3, verifiability),
   - Chat template and format choice (single template, diverse system prompts),
   - Contamination and pipeline validation plans.
3. Document completion in [NEXT_STEPS_COMPLETION_REPORT.md](./NEXT_STEPS_COMPLETION_REPORT.md) **§1 Study completion**.

**Checklist.**

- [ ] Part 1 (1.1–1.5) read and summarized.
- [ ] Findings table implications checked against mapping and pipeline.
- [ ] Study completion date and sign-off recorded in completion report.

---

## Step 2 — Map candidate SFT datasets (Part 2.1)

**Objective.** Treat the candidate dataset mapping as the agreed reference for observation mode. No new mapping is required if the strategy doc is accepted; otherwise update the strategy doc and record the outcome.

**Actions.**

1. Open **§2.1 Mapping candidate SFT datasets** in the Dataset Sourcing Strategy (Tier 1, Tier 2, Tier 3 tables).
2. Confirm that each candidate has a clear rationale and relevant benchmarks.
3. If the team adds/removes candidates or changes tiers, update the strategy doc and note changes in the completion report **§2 Mapping outputs**.
4. Record in the completion report: “Candidate mapping: [Accepted as in strategy doc | Updated: list changes]” and the date.

**Checklist.**

- [ ] Candidate mapping reviewed.
- [ ] Any changes documented in strategy doc and completion report §2.

---

## Step 3 — Map benchmark coverage (Part 2.2)

**Objective.** Ensure every agreed benchmark (from Benchmark-Datasets-V01.xlsx) has a mapped candidate dataset and that gaps/over-coverage are explicit.

**Actions.**

1. Open **§2.2 Mapping benchmark coverage** in the Dataset Sourcing Strategy (benchmark category vs primary candidate datasets).
2. Cross-check against **Benchmark-Datasets-V01.xlsx** (Scripts, V02/V04 sheets) so no agreed benchmark is missing from the coverage map.
3. If any benchmark has no candidate or is under/over-covered, add a note in the completion report **§2** and, if needed, update the strategy doc.
4. Record in completion report: “Benchmark coverage map: [Accepted | Gaps/updates: …]” and the date.

**Checklist.**

- [ ] Coverage map checked against Benchmark-Datasets-V01.xlsx.
- [ ] Gaps or updates documented in completion report §2.

---

## Step 4 — Dry-run evaluations (Part 2.3)

**Objective.** Run evaluations on the **base model** (no SFT) on the agreed benchmarks to establish a baseline and validate the evaluation harness.

**Actions.**

1. Identify the **base model** (e.g. from training plan) and the **evaluation framework** (e.g. OLMES per Benchmark-Datasets-V01.xlsx Scripts sheet).
2. Run the agreed benchmark tasks (or a subset if full run is expensive) with the base model. Example (OLMES-style; adjust to your harness):
   ```bash
   # Example — replace with your actual eval command and benchmark list
   # olmes --model <base_model> --task mmlu:mc::olmes
   # olmes --model <base_model> --task gsm8k::olmes
   # ... (one per benchmark from Scripts sheet)
   ```
3. Record in completion report **§3 Dry-run evaluations**:
   - Base model name and version
   - Eval framework and version
   - List of benchmarks run
   - Results table (benchmark | metric | value)
   - Any failures or harness issues
4. Confirm: no training or weight updates were performed; eval-only.

**Checklist.**

- [ ] Base model and eval framework identified.
- [ ] Dry-run evals executed; results recorded in completion report §3.
- [ ] No weight updates during this step.

---

## Step 5 — Pipeline validation (Part 2.4)

**Objective.** Validate the full data pipeline (standardize → template → … → verify loss masking) and confirm the training collator will use correct loss masking and padding.

**Actions.**

1. **Run the pipeline on the sample data** (this directory contains `sample_input_alpaca.jsonl`):
   ```bash
   cd sft_data/scripts
   # 1. Standardize
   python standardize_conversation_format.py ../observation_mode/sample_input_alpaca.jsonl ../observation_mode/standardized.jsonl --format alpaca
   # 2. Apply chat template
   python apply_chat_template.py ../observation_mode/standardized.jsonl ../observation_mode/templated.jsonl --template chatml
   # 3. (Optional) Sample for quality — skip for tiny sample
   # 4. (Optional) Dedup — requires pretrain hashes; skip if not available
   # 5. Train/val split (use standardized or templated as input)
   python train_val_split.py ../observation_mode/standardized.jsonl --train-out ../observation_mode/train.jsonl --val-out ../observation_mode/val.jsonl --val-ratio 0.2 --seed 42
   # 6. Verify loss masking (use train output; --tokenizer optional)
   python verify_loss_masking.py ../observation_mode/train.jsonl --sample 5
   # With tokenizer (if transformers installed and model path set):
   # python verify_loss_masking.py ../observation_mode/train.jsonl --tokenizer path/to/model --sample 5
   ```
   Or run the helper script from `sft_data/observation_mode`:
   ```bash
   cd sft_data/observation_mode
   bash run_pipeline_validation.sh
   ```
2. **Verify outputs:** Check that `standardized.jsonl` has `conversations` with `role` and `content`; `templated.jsonl` has `text` (and optionally `input_ids`); `train.jsonl` / `val.jsonl` exist and have no overlap; `verify_loss_masking.py` prints token counts and assistant vs ignored positions.
3. **Training collator:** In your actual training code, confirm (per SFT_DATA_CHECKLIST_7.1.md items 8–9):
   - Labels: only assistant token positions have token id; all others = -100.
   - Padding: right-pad; pad positions in labels = -100.
   - Loss: `CrossEntropyLoss(ignore_index=-100)`.
4. Record in completion report **§4 Pipeline validation**: commands run, pass/fail per step, any errors; training collator checklist (done/pending).

**Checklist.**

- [ ] Pipeline run on sample (standardize → template → split → verify_loss_masking).
- [ ] Outputs checked; results recorded in completion report §4.
- [ ] Training collator alignment with items 8–9 confirmed and documented.

---

## Step 6 — Contamination checks (Part 3)

**Objective.** Verify high-risk candidates (math, code, IFEval, knowledge) and document dataset versions and contamination status before finalizing the dataset list.

**Actions.**

1. For each **high-risk candidate** you plan to use (see Strategy §3.3):
   - **Math (MATH, GSM8K, AIME):** OpenMathInstruct-2, NuminaMath-TIR, math slices of Tulu 3. Verify: not built from benchmark test sets; use train-only splits where available. Record source, version, and check result in completion report **§5 Contamination checks**.
   - **Code (HumanEval, MBPP, APPS):** Magicoder, CodeFeedback, Tulu 3 code. Verify: no test-set problems included; prefer datasets that explicitly exclude HumanEval/MBPP/APPS test sets. Record in §5.
   - **IFEval:** Tulu 3 Persona IF and similar. Confirm no overlap with IFEval test prompts. Record in §5.
   - **Knowledge (MMLU, TriviaQA, GPQA):** Avoid SFT sets built from benchmark test items. General instruction data: lower risk if not benchmark-specific. Record in §5.
2. For **all** candidates that will be used, document in §5: dataset name, version/source (e.g. Hugging Face commit or date), and “Contamination check: [Pass | Fail | Not yet done]”.
3. If any candidate cannot be verified as clean, exclude it or restrict to a verified subset and document in §5.

**Checklist.**

- [ ] High-risk candidates checked per Strategy §3.3.
- [ ] All planned candidates have version and contamination status in completion report §5.
- [ ] Unverified or failed candidates excluded or restricted and documented.

---

## Step 7 — Final status and next actions

**Objective.** Decide whether observation mode is complete and the team can proceed to finalize the dataset list and training.

**Actions.**

1. In completion report **§6 Final status**, set:
   - **Observation mode complete:** [Yes | No]. If No, list blockers (e.g. dry-run not run, pipeline failed, contamination not checked).
   - **Ready to finalize dataset list:** [Yes | No].
   - **Next actions:** Bullet list (e.g. “Finalize dataset list from candidates”; “Obtain Team 5 approval”; “Run training”).
2. Only if **Observation mode complete** and **Ready to finalize dataset list** are Yes, proceed to finalize the dataset list (within candidate set and contamination constraints), obtain approvals, and run training.

**Checklist.**

- [ ] §6 Final status filled.
- [ ] If complete: dataset list finalization and approval steps scheduled.

---

## Quick reference — Pipeline validation commands

From repo root:

```bash
cd sft_data/scripts
python standardize_conversation_format.py ../observation_mode/sample_input_alpaca.jsonl ../observation_mode/standardized.jsonl --format alpaca
python apply_chat_template.py ../observation_mode/standardized.jsonl ../observation_mode/templated.jsonl --template chatml
python train_val_split.py ../observation_mode/standardized.jsonl --train-out ../observation_mode/train.jsonl --val-out ../observation_mode/val.jsonl --val-ratio 0.2 --seed 42
python verify_loss_masking.py ../observation_mode/train.jsonl --sample 5
```

Optional with tokenizer: add `--tokenizer <model_path>` to `verify_loss_masking.py`.
