📌 Candidate Datasets for LLM Pretraining

This document summarizes shortlisted datasets for training a Large Language Model (LLM) from scratch, focusing on:

- **Scale & coverage**
- **Indic language strength**
- **Licensing clarity**
- **Cleaning + contamination risk**
- **Suitability for foundation vs specialization**

# 📖 LLM Data Glossary: Definitions & Terms

This document serves as the technical reference for the **Data Radar and Acquisition** phase. It defines the metrics used to evaluate dataset quality and the techniques used to refine raw data into "model-ready" tokens.

---


## 📊 Volume & Architectural Metrics
| Term | Definition | Impact on Model |
| :--- | :--- | :--- |
| **Token Scale** | The total volume of atomic text units (tokens) in a dataset. | Defines the knowledge "depth." Foundations require Trillion-scale; specialized sets require Million/Billion-scale. |
| **General Pretraining Share** | The % of the total training "mixture" allocated to a specific dataset. | Controls the "personality" of the model. Usually **40–50%** for general base data to prevent over-specialization. |
| **Dataset Reduction** | The % of data discarded during the cleaning phase. | A high reduction rate (e.g., 30-40%) is often a sign of high-quality, aggressive filtering. |
| **Raw Pool Removed** | The specific volume of the initial raw "crawl" that was discarded before tokenization. | Measures the "noise" level of the original source; helps in auditing the efficiency of the scraper. |
| **Foundation Suitability** | A rating (1-5) of how well a dataset can build core reasoning and logic from scratch. | Based on the inclusion of code, math, and high-quality academic reasoning. |

---

## 🛠️ Data Cleaning & Refinement Techniques
| Technique | How it Works | Purpose |
| :--- | :--- | :--- |
| **MinHash Deduplication** | Uses probabilistic "fingerprints" to find and remove documents that are nearly identical. | Eliminates "near-duplicates" (e.g., the same news article on 50 different websites). |
| **Perplexity Filtering** | Uses a small "judge" model to score text; if the text is gibberish or nonsensical, it receives a high score and is deleted. | Removes "junk" text, SEO spam, and incoherent web scrapes. |
| **Repetitive Header Removal** | Strips recurring boilerplate like navigation menus, "Sign In" buttons, and legal footers. | Prevents the model from learning "website UI" components instead of actual language. |
| **Bloom Filter Filtering** | A space-efficient data structure used to check if a specific piece of data has been seen before. | Used for high-speed deduplication across massive (Trillion-token) datasets. |

---

## 🛡️ Trust, Quality & Risk Metrics
| Metric | Definition | Critical Threshold |
| :--- | :--- | :--- |
| **Post-filter Contamination** | The % of evaluation benchmark questions (test answers) that leaked into the training data. | **Target: <1%**. High contamination results in "fake" benchmark scores (cheating). |
| **Indic Strength** | The depth and accuracy of coverage for the 22+ official Indian languages. | Essential for ensuring the model respects regional linguistic nuances. |
| **Licensing Clarity** | The legal transparency and usage rights of the data source. | **Low Risk:** CC-0/Public Domain. **High Risk:** Copyrighted textbooks or private data. |

---
## ⚖️ Dataset Licensing Guide
The following table categorizes our shortlisted datasets by their legal risk and usage permissions.

| License Type | Examples | Key Requirements | Permissions | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| **ODC-BY / CC-BY-4.0** | **Dolma**, **Sangraha** | Attribution. | Use and modify freely; must credit the original source. | **Low** |
| **CC-BY-SA** | **Wikipedia** | Share-Alike. | Any derivative model or data must use the same license. | **Medium** |
| **CC-BY-NC** | **Certain Research Sets** | Non-Commercial. | Strictly for research; cannot be used in paid products. | **High** |
| **Proprietary / Custom** | **NCERT Books** | Varies by source. | Redistribution is often restricted; requires explicit verification. | **Critical** |


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


## 3. NCERT Books Dataset (Educational / India-specific)

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
| **NCERT Books** | Small | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | Education mid-training |

**Final Dataset Risk Summary**

| Dataset          | License Risk | Copyright Risk | Contamination Risk | Notes                                     |
| ---------------- | ------------ | -------------- | ------------------ | ----------------------------------------- |
| **Dolma**        | Low–Medium   | Medium         | Low                | Strong general base, attribution required |
| **Sangraha**     | Low          | Low            | Very Low           | Best Indic dataset, clean + recent        |
| **NCERT Books**  | Medium       | Medium–High    | Low                | Needs redistribution/license verification |

---

# ✅ Recommendation

- **Foundation Backbone Dataset:** **Dolma**
- **Primary Indic Dataset:** **Sangraha**
- **Domain-Specific Add-on:** NCERT Books

