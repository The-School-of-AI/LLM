📌 Candidate Datasets for LLM Pretraining

This document summarizes shortlisted datasets for training a Large Language Model (LLM) from scratch, focusing on:

- **Scale & coverage**
- **Indic language strength**
- **Licensing clarity**
- **Cleaning + contamination risk**
- **Suitability for foundation vs specialization**

---

## 1. Dolma (AllenAI)

**Best general-purpose large-scale dataset for strong foundation pretraining**

| Attribute | Details |
|----------|---------|
| **Dataset Type** | Web + Academic + Code mixture |
| **Token Scale** | 3T raw / ~2T effective |
| **License** | ODC-BY (High usability) |
| **Primary Use** | General-purpose base pretraining |
| **Release Freshness** | High (v1.7 released Apr 2024) |

### 🔗 Hugging Face Link
➡️ https://huggingface.co/datasets/allenai/dolma


**Data Quality & Processing**

- Moderate cleaning + decontamination

- Bloom filter deduplication

- Strong academic + code coverage for reasoning tasks

**Composition**

- General pretraining share: 40–50%

- Dataset reduction: ~30–40%

**Contamination Risk**

- Post-filter contamination: <1%

**Benchmarks Supported**

- MMLU, BBH, GSM8K

- Contains C4-derived material

**Stakeholder Score (1–5)**
| Category                 | Score       |
| ------------------------ | ----------- |
| **General Coverage**         | ⭐⭐⭐⭐⭐ (5)   |
| **Reasoning + Code Utility** | ⭐⭐⭐⭐⭐ (5)   |
| **Indic Suitability**        | ⭐⭐⭐⭐☆ (4)   |
| **Contamination Risk**       | ⭐⭐⭐⭐☆ (Low) |

---

## 2. Sangraha (AI4Bharat)

**Best choice for Indic-first multilingual training**

| Attribute | Details |
|----------|---------|
| **Dataset Type** | Multilingual Indic corpus (Web + PDFs + Transcribed + Synthetic) |
| **Languages Covered** | 22 Indic languages |
| **Token Scale** | ~251B tokens |
| **License** | CC-BY-4.0 |
| **Primary Use** | Pretraining + Mid-training (Indic specialization) |

### 🔗 Hugging Face Link
➡️ https://huggingface.co/datasets/ai4bharat/sangraha

**Data Quality & Processing**

- Aggressive cleaning + deduplication

- PDF extraction via olmOCR

- Perplexity + heuristic filtering

- Post-sampling contamination: <5%

**Composition & Retention**

- Indic-specific training share: 5–15%

- Raw pool removed: ~35% (high refinement)

**Benchmarks Supported**

- IndicXTREME
- IndicGLUE

**Stakeholder Score (1–5)**

| Category           | Score            |
| ------------------ | ---------------- |
| **Indic Context Fit**  | ⭐⭐⭐⭐⭐ (5)        |
| **Freshness**          | ⭐⭐⭐⭐⭐ (5)        |
| **Diversity**          | ⭐⭐⭐⭐⭐ (5)        |
| **Contamination Risk** | ⭐⭐⭐⭐⭐ (Low risk) |

✅ Includes English subsets via IndicTrans2

⚠ Verified subset coverage is minimal (~0%)

---

## 3. IndicCorp v2 (AI4Bharat)

**Strong baseline Indic dataset, widely used but older and noisier**

| Attribute | Details |
|----------|---------|
| **Dataset Type** | Monolingual Indic web scrape |
| **Languages Covered** | 24 Indic languages |
| **Token Scale** | ~20.9B tokens |
| **License** | CC-0 (Public Domain) |
| **Primary Use** | Legacy/base Indic pretraining |

### 🔗 Hugging Face Link
➡️ https://huggingface.co/datasets/ai4bharat/IndicCorpV2

**Data Quality & Processing**

- Requires HTML + boilerplate cleaning

- Deduplication: sentence-level MinHash

- Noise includes navigational + repetitive news text

**Contamination & Noise**

- Repetitive headers: ~10–15%

- Typical web contamination: ~2–3%

**Benchmarks Supported**

- IndicXTREME
- IndicGLUE

**Stakeholder Score (1–5)**

| Category              | Score       |
| --------------------- | ----------- |
| **Indic Coverage**        | ⭐⭐⭐⭐☆ (4)   |
| **Freshness**            | ⭐⭐⭐☆☆ (3)   |
| **Cleanliness**           | ⭐⭐⭐☆☆ (3–4) |
| **Licensing Flexibility** | ⭐⭐⭐⭐⭐ (5)   |

❌ No English subset included

⚠ Older release compared to Sangraha

---

## 4. NCERT Books Dataset (Educational / India-specific)

**High-quality curriculum-aligned dataset for education + grounded Indian knowledge**

| Attribute | Details |
|----------|---------|
| **Dataset Type** | School textbooks (NCERT, India) |
| **Domain** | Science, Math, Social Studies, Languages |
| **Languages Covered** | Primarily English + Hindi |
| **Token Scale** | Smaller (hundreds of millions–low billions) |
| **Primary Use** | Mid-training + Instruction tuning |

### 🔗 Verified Hugging Face Dataset Link
➡️ https://huggingface.co/datasets/KadamParth/Ncert_dataset

**Composition**

- ~120k+ rows of textbook material

- Multiple grades + subjects

- Clean, structured curriculum content

**Why NCERT is Valuable**

- Extremely low-toxicity, high-quality text

- Strong for:

  - Educational assistants

  - Indian exam preparation (CBSE/UPSC-style)

  - Curriculum-grounded reasoning

**Risks / Considerations**

- Must confirm redistribution + licensing per dataset source

- Not large enough for full foundation pretraining

- Best used as mid-training add-on

**Stakeholder Score (1–5)**
| Category                 | Score     |
| ------------------------ | --------- |
| **Educational Value**       | ⭐⭐⭐⭐⭐ (5) |
| **Cleanliness**              | ⭐⭐⭐⭐⭐ (5) |
| **Scale for Pretraining**    | ⭐⭐☆☆☆ (2) |
| **Indic Cultural Grounding** | ⭐⭐⭐⭐☆ (4) |



---

# 📊 Summary Comparison

| Dataset | Tokens | Indic Strength | Freshness | Cleanliness | Best Role |
|--------|--------|---------------|----------|------------|----------|
| **Dolma** | ~2T eff | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | General foundation pretraining |
| **Sangraha** | ~251B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Indic-specialized pretraining |
| **IndicCorp v2** | ~21B | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | Legacy Indic baseline |
| **NCERT Books** | Small | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | Education mid-training |

**Final Dataset Risk Summary**

| Dataset          | License Risk | Copyright Risk | Contamination Risk | Notes                                     |
| ---------------- | ------------ | -------------- | ------------------ | ----------------------------------------- |
| **Dolma**        | Low–Medium   | Medium         | Low                | Strong general base, attribution required |
| **Sangraha**     | Low          | Low            | Very Low           | Best Indic dataset, clean + recent        |
| **IndicCorp v2** | Very Low     | Medium         | Medium             | Older scrape, noisier text                |
| **NCERT Books**  | Medium       | Medium–High    | Low                | Needs redistribution/license verification |

---

# ✅ Recommendation

- **Foundation Backbone Dataset:** **Dolma**
- **Primary Indic Dataset:** **Sangraha**
- **Supplementary Indic Dataset:** IndicCorp v2
- **Domain-Specific Add-on:** NCERT Books
