# 🛰️ Data Radar & Acquisition: Project "Golden Corpus"

**Target:** ~2T Effective Pretraining Tokens  
**Date:** January 28, 2026  
**Status:** 🟢 Ready for Execution (Jan 29)

---

## 📋 Executive Summary
Based on technical post-mortems of **DeepSeek-V3**, **Llama 3**, and **Qwen**, this project shifts focus from raw volume to **High-Signal Density**. Our strategy involves distilling a >20T raw token pool into a **~2T "Golden Corpus"** using aggressive Perplexity (PPL) filtering, ML-based quality scoring, and reasoning-heavy domain weighting.

---

## 📊 Curated Dataset Composition
We have selected 6 primary sources to hit our target, prioritizing "Reasoning Heavy" ratios (Code/Math/Edu).

| Rank | Dataset | Est. Effective Tokens | License | Type | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **FineWeb** | 600B | ODC-By 1.0 | General Web | 50% General Knowledge base |
| 2 | **FineWeb-Edu** | 500B | ODC-By 1.0 | Reasoning | Educational/High-quality signal |
| 3 | **The Stack v2** | 350B | Permissive | Code | Logic & Programming reasoning |
| 4 | **Dolma v1.7** | 300B | ODC-By | Books/Academic | Long-context depth & formal text |
| 5 | **DCLM-Baseline** | 200B | ODC-By 1.0 | Web | High-quality control group |
| 6 | **OpenMathInstruct-2**| 5B | CC-BY 4.0 | Math | **Annealing Phase** booster |

**Total Estimated Volume:** ~1.955T Effective Tokens

---

## 🛠️ Data Strategy & Alignment

### 1. Reasoning-First Ratios
Unlike early-gen models, we are overweighting high-signal content to maximize "Intelligence-per-Token":
* **Reasoning Core:** (Edu + Code + Math) = **~65%** of the total mix.
* **Logic Transfer:** Following DeepSeek-V3’s findings, we leverage code/math to improve general-purpose reasoning.

### 2. Quality & Filtering (Llama 3 Methodology)
* **PPL Filtering:** Aggressive removal of low-information content and repetitive web-spam.
* **Safety:** Llama Guard-style classifiers for hate speech and adult content.
* **PII Redaction:** Multi-pass scrubbing using the **Presidio analyzer**.

### 3. Licensing & Traceability
* **Code:** Strictly limited to the "Permissive-Only" subset of *The Stack v2* (MIT/Apache 2.0/BSD). 
* **Verification:** Automatic "dirty word" filters scan for copyleft (GPL/AGPL) license headers.

---

## 🛡️ Benchmark Decontamination Protocol
To ensure valid performance metrics, we implement the following "Canary" checks:

1.  **N-Gram Scrubbing:** 13-gram overlap removal (Qwen style) and 8-gram character checks (Llama 3 style).
2.  **Mandatory Targets:** Exact string matching against **MMLU, GSM8K, HumanEval, and MBPP** test sets.
3.  **Hold-outs:** Creation of a "clean" validation set from held-out FineWeb subsets to monitor perplexity during training.

---

## 🚀 Next Steps
- [ ] **Acquire:** Download manifests for FineWeb, Stack v2, and Dolma.
- [ ] **Verify:** Execute license verification scripts on code samples.
- [ ] **Coordinate:** Hand off token statistics to the **Benchmarking Team** for final sanity check evals.
- [ ] **Augment:** Fresh crawl of **arXiv (up to Jan 2026)** to include latest SOTA papers.

> **Status:** Ready for "Peak Execution" starting Jan 29.
