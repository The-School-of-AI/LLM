
## 1. FineWeb-Edu

* **Dataset type/domain:** Educational Web (High Quality)
* **Raw + effective token scale:** 15T (Raw) / 1.3T (Edu)
* **License + legal usability:** ODC-By 1.0 (High)
* **Benchmark contamination risk:** Low (Aggressive Decontam)
* **Engineering filters required:** Minimal
* **Recommended weighting:** 40–50% (Base)
* **Exact HF identifier / source:** HuggingFace FW/fineweb-eduml
* **Deduplication target %:** Already deduped (MinHash)
* **Freshness priority:** High (2013–2024)
* **Training phase:** Pretrain (Base)
* **Token budget contribution (B):** ~1,000B
* **Dedup Method:** MinHash (Aggressive)
* **Duplication %:** < 1%
* **Avg Entropy:** High (Diverse)
* **Avg Perplexity:** Low (Cleaned)
* **Junk %:** ~0% (Edu Filtered)
* **Spam %:** ~0%
* **Freshness Score:** 5-May
* **Benchmarks Checked:** MMLU, ARC, HellaSwag
* **Overlap %:** Low (vs C4)
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 5 (Edu)
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (General)
* **Distillation:** No (Annotated)
* **Freshness:** 2024

---

## 2. DCLM-Baseline

* **Dataset type/domain:** Filtered Web (Compute Optimal)
* **Raw + effective token scale:** ~240T (Raw) / 4T (Clean)
* **License + legal usability:** ODC-BY (High)
* **Benchmark contamination risk:** Low (Model-based Filter)
* **Engineering filters required:** Minimal (Ready to use)
* **Recommended weighting:** 30–40% (Efficiency Base)
* **Exact HF identifier / source:** foundations/dclm-baseline-1.0
* **Deduplication target %:** High (Global Dedup)
* **Freshness priority:** Medium (Pre-2023)
* **Training phase:** Pretrain (Base)
* **Token budget contribution (B):** ~800B
* **Dedup Method:** Bloom Filter (Global)
* **Duplication %:** < 2%
* **Avg Entropy:** High
* **Avg Perplexity:** Low
* **Junk %:** < 1%
* **Spam %:** < 0.5%
* **Freshness Score:** 5-Mar
* **Benchmarks Checked:** MMLU (Core), MMLU-Pro
* **Overlap %:** High (Contains CC)
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 4
* **Benchmark Safety (0–5):** 4
* **Domain Value (0–5):** 5 (Efficiency)
* **Distillation:** No
* **Freshness:** 2023

---

## 3. The Stack v2

* **Dataset type/domain:** Code (600+ Languages)
* **Raw + effective token scale:** 67TB (Raw) / 3TB+ (Dedup)
* **License + legal usability:** Permissive Varies (Apache/MIT)
* **Benchmark contamination risk:** Low (Opt-out compliant)
* **Engineering filters required:** High (License/Language filter)
* **Recommended weighting:** 15–20% (Reasoning Base)
* **Exact HF identifier / source:** bigcode/the-stack-v2
* **Deduplication target %:** High (Near-dedup)
* **Freshness priority:** High (2024)
* **Training phase:** Pretrain / Mid-train
* **Token budget contribution (B):** ~400B
* **Dedup Method:** MinHash + Exact
* **Duplication %:** ~20% (if unfiltered)
* **Avg Entropy:** High (Syntax heavy)
* **Avg Perplexity:** Low
* **Junk %:** Low
* **Spam %:** Low
* **Freshness Score:** 5-May
* **Benchmarks Checked:** HumanEval, MBPP
* **Overlap %:** Medium (vs StarCoder)
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 5 (Code)
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (Reasoning)
* **Distillation:** No
* **Freshness:** 2024

---

## 4. OpenMathInstruct-2

* **Dataset type/domain:** Math Reasoning (SFT)
* **Raw + effective token scale:** 14M Samples / ~10B Tokens
* **License + legal usability:** CC-BY-4.0 (High)
* **Benchmark contamination risk:** Low (Synthetic Generated)
* **Engineering filters required:** Minimal
* **Recommended weighting:** 2–5% (Math Specialist)
* **Exact HF identifier / source:** nvidia/OpenMathInstruct-2
* **Deduplication target %:** N/A (Unique prompts)
* **Freshness priority:** High (2024)
* **Training phase:** SFT / Post-train
* **Token budget contribution (B):** ~10B
* **Dedup Method:** Prompt-based uniq
* **Duplication %:** ~0%
* **Avg Entropy:** Medium (Structured)
* **Avg Perplexity:** Very Low
* **Junk %:** 0%
* **Spam %:** 0%
* **Freshness Score:** 5-May
* **Benchmarks Checked:** MATH, GSM8k
* **Overlap %:** N/A
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 5 (Math)
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (Math/Logic)
* **Distillation:** Yes (Llama-3.1-405B)
* **Freshness:** 2024

---

## 5. Synthetic-2

* **Dataset type/domain:** Verified Reasoning (RL/CoT)
* **Raw + effective token scale:** 4M Traces / ~5B Tokens
* **License + legal usability:** Apache 2.0 (High)
* **Benchmark contamination risk:** Very Low (Verified)
* **Engineering filters required:** Minimal
* **Recommended weighting:** 5–10% (Reasoning Specialist)
* **Exact HF identifier / source:** PrimeIntellect/SYNTHETIC-2
* **Deduplication target %:** N/A (Unique traces)
* **Freshness priority:** Very High (Late 2025)
* **Training phase:** SFT / RLHF / Mid-train
* **Token budget contribution (B):** ~10B
* **Dedup Method:** Ground Truth Verification
* **Duplication %:** ~0%
* **Avg Entropy:** Low (Step-by-step logic)
* **Avg Perplexity:** Very Low
* **Junk %:** 0%
* **Spam %:** 0%
* **Freshness Score:** 5-May
* **Benchmarks Checked:** GPQA, MATH, AIME
* **Overlap %:** N/A
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 5+ (Verified)
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (Complex Logic)
* **Distillation:** Yes (DeepSeek-R1/Qwen)
* **Freshness:** Late 2025

---

## 6. SmolLM Corpus

* **Dataset type/domain:** Pre-mixed Curriculum (Textbook)
* **Raw + effective token scale:** ~300B Tokens
* **License + legal usability:** ODC-By / Apache 2.0
* **Benchmark contamination risk:** Low (Synthetic/Curated)
* **Engineering filters required:** Minimal
* **Recommended weighting:** 10–15% (Definition/Fact)
* **Exact HF identifier / source:** HuggingFace TB/smollm-corpus
* **Deduplication target %:** Already Deduped
* **Freshness priority:** High (2024)
* **Training phase:** Pretrain / Mid-train
* **Token budget contribution (B):** ~300B
* **Dedup Method:** Clustering / MinHash
* **Duplication %:** < 1%
* **Avg Entropy:** Medium (Textbook style)
* **Avg Perplexity:** Very Low
* **Junk %:** ~0%
* **Spam %:** 0%
* **Freshness Score:** 5-Apr
* **Benchmarks Checked:** MMLU, HellaSwag
* **Overlap %:** Medium (Contains FW-Edu)
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 5 (Textbook)
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 4 (Foundational)
* **Distillation:** Yes (Mixtral)
* **Freshness:** 2024

---

## 7. FineWeb

* **Dataset type/domain:** Web (General)
* **Raw + effective token scale:** 15T (Raw) / 15T (Eff)
* **License + legal usability:** ODC-By 1.0 (High)
* **Benchmark contamination risk:** Low (Aggressive decontamination)
* **Engineering filters required:** Minimal (Ready to use)
* **Recommended weighting:** 50–60%
* **Exact HF identifier / source:** HuggingFace FW/fineweb
* **Deduplication target %:** Already deduped (MinHash)
* **Freshness priority:** High (2013–2024)
* **Training phase:** Pretrain (Base)
* **Token budget contribution (B):** ~1,000B
* **Dedup Method:** MinHash (Aggressive)
* **Duplication %:** < 1%
* **Avg Entropy:** High (Diverse)
* **Avg Perplexity:** Low (Cleaned)
* **Junk %:** Very Low
* **Spam %:** < 0.5%
* **Freshness Score:** 5-May
* **Benchmarks Checked:** MMLU, HellaSwag, ARC
* **Overlap %:** Low (vs C4/Dolma)
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 4
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (General)
* **Distillation:** No
* **Freshness:** 2024

---

## 8. Dolma

* **Dataset type/domain:** Web + Academic + Code
* **Raw + effective token scale:** 3T (Raw) / ~2T (Eff)
* **License + legal usability:** ODC-BY (High)
* **Benchmark contamination risk:** Medium (Decontaminated)
* **Engineering filters required:** Moderate (Mix required)
* **Recommended weighting:** 40–50%
* **Exact HF identifier / source:** allenai/dolma
* **Deduplication target %:** ~30-40%
* **Freshness priority:** High (Updated V1.7)
* **Training phase:** Pretrain (Base)
* **Token budget contribution (B):** ~800B
* **Dedup Method:** Bloom Filter
* **Duplication %:** ~10-15%
* **Avg Entropy:** High
* **Avg Perplexity:** Low
* **Junk %:** Low
* **Spam %:** < 1%
* **Freshness Score:** 5-Apr
* **Benchmarks Checked:** MMLU, BBH, GSM8k
* **Overlap %:** High (Contains C4)
* **Leak Risk Level:** Low-Medium
* **Legal (0–5):** 5
* **Signal (0–5):** 4
* **Benchmark Safety (0–5):** 4
* **Domain Value (0–5):** 5 (General)
* **Distillation:** No
* **Freshness:** 2023

---

## 9. C4 (Original)

* **Dataset type/domain:** Web (Snapshot)
* **Raw + effective token scale:** ~800B (Raw) / 156B (Clean)
* **License + legal usability:** ODC-BY (Medium)
* **Benchmark contamination risk:** High (Older, heavily leaked)
* **Engineering filters required:** High (Blocklists needed)
* **Recommended weighting:** 10–20% (if legacy)
* **Exact HF identifier / source:** c4
* **Deduplication target %:** Low (Document level)
* **Freshness priority:** Low (2019 Snapshot)
* **Training phase:** Pretrain (Legacy)
* **Token budget contribution (B):** ~200B
* **Dedup Method:** Url/Line-based
* **Duplication %:** ~10-15%
* **Avg Entropy:** Medium
* **Avg Perplexity:** Medium
* **Junk %:** Medium
* **Spam %:** ~5%
* **Freshness Score:** 5-Jan
* **Benchmarks Checked:** None (Historical)
* **Overlap %:** N/A
* **Leak Risk Level:** High
* **Legal (0–5):** 3
* **Signal (0–5):** 3
* **Benchmark Safety (0–5):** 2
* **Domain Value (0–5):** 3 (Legacy)
* **Distillation:** No
* **Freshness:** 2019

---

## 10. SYNTH (Cosmopedia)

* **Dataset type/domain:** Synthetic / Textbook
* **Raw + effective token scale:** ~25B (High Quality)
* **License + legal usability:** Varies (check gen model)
* **Benchmark contamination risk:** High (Model gen loop)
* **Engineering filters required:** Minimal (Quality check)
* **Recommended weighting:** 10–30% (Curriculum)
* **Exact HF identifier / source:** HuggingFace TB/cosmopedia
* **Deduplication target %:** N/A (Generated)
* **Freshness priority:** N/A
* **Training phase:** Pretrain / SFT
* **Token budget contribution (B):** ~200B
* **Dedup Method:** N/A
* **Duplication %:** < 1%
* **Avg Entropy:** Low (Clean/Predictable)
* **Avg Perplexity:** Very Low
* **Junk %:** ~0%
* **Spam %:** 0%
* **Freshness Score:** N/A
* **Benchmarks Checked:** GSM8k, MMLU
* **Overlap %:** N/A
* **Leak Risk Level:** Medium
* **Legal (0–5):** 4
* **Signal (0–5):** 5
* **Benchmark Safety (0–5):** 3
* **Domain Value (0–5):** 4 (Reasoning)
* **Distillation:** Yes
* **Freshness:** 2024

---

## 11. StarCoder

* **Dataset type/domain:** Code (General)
* **Raw + effective token scale:** ~250B+
* **License + legal usability:** BigCode OpenRAIL-M
* **Benchmark contamination risk:** Low
* **Engineering filters required:** Moderate (PII/Secret removal)
* **Recommended weighting:** 10–15%
* **Exact HF identifier / source:** bigcode/starcoderdata
* **Deduplication target %:** High (File/Repo level)
* **Freshness priority:** Medium
* **Training phase:** Pretrain / Mid-train
* **Token budget contribution (B):** ~200B
* **Dedup Method:** StarCoder Pipeline
* **Duplication %:** ~20-30%
* **Avg Entropy:** High (Syntax heavy)
* **Avg Perplexity:** N/A
* **Junk %:** Low
* **Spam %:** < 1%
* **Freshness Score:** 5-Mar
* **Benchmarks Checked:** HumanEval
* **Overlap %:** Medium (vs Stack)
* **Leak Risk Level:** Low
* **Legal (0–5):** 4
* **Signal (0–5):** 4
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (Coding)
* **Distillation:** No
* **Freshness:** 2023

---

## 12. Stack v2 (Permissive)

* **Dataset type/domain:** Code (Permissive)
* **Raw + effective token scale:** ~3TB (Raw) / 600B+ (Dedup)
* **License + legal usability:** Permissive (Apache/MIT)
* **Benchmark contamination risk:** Low
* **Engineering filters required:** High (License verification)
* **Recommended weighting:** 10–20%
* **Exact HF identifier / source:** bigcode/the-stack-v2
* **Deduplication target %:** High (Near-dedup)
* **Freshness priority:** High
* **Training phase:** Pretrain / Mid-train
* **Token budget contribution (B):** ~300B
* **Dedup Method:** MinHash + Exact
* **Duplication %:** ~40% (before dedup)
* **Avg Entropy:** High
* **Avg Perplexity:** N/A
* **Junk %:** Moderate
* **Spam %:** Low
* **Freshness Score:** 5-May
* **Benchmarks Checked:** HumanEval
* **Overlap %:** High
* **Leak Risk Level:** Low
* **Legal (0–5):** 5
* **Signal (0–5):** 4
* **Benchmark Safety (0–5):** 5
* **Domain Value (0–5):** 5 (Coding)
* **Distillation:** No
* **Freshness:** 2024

---

## 13. StackExchange

* **Dataset type/domain:** Q&A / Human Reasoning
* **Raw + effective token scale:** ~50B
* **License + legal usability:** CC-BY-SA 3.0 (Viral Risk)
* **Benchmark contamination risk:** Medium (GSM8k leaks)
* **Engineering filters required:** Moderate (HTML cleaning)
* **Recommended weighting:** 2–5%
* **Exact HF identifier / source:** flax-sentence-embeddings/stackexchange_xml
* **Deduplication target %:** Low
* **Freshness priority:** Medium
* **Training phase:** Mid-train / Post-train
* **Token budget contribution (B):** ~50B
* **Dedup Method:** Post-level
* **Duplication %:** High (Repetitive answers)
* **Avg Entropy:** High (Dense info)
* **Avg Perplexity:** Low
* **Junk %:** Moderate (Comments)
* **Spam %:** Low
* **Freshness Score:** 5-Mar
* **Benchmarks Checked:** GSM8k, MATH
* **Overlap %:** Low
* **Leak Risk Level:** High
* **Legal (0–5):** 2 (ShareAlike)
* **Signal (0–5):** 5
* **Benchmark Safety (0–5):** 2
* **Domain Value (0–5):** 5 (Reasoning)
* **Distillation:** No
* **Freshness:** 2023
