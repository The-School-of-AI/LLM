# Dataset Distribution & Cleaning

## Definitions: The Difficulty Ladder

| Band | Level | Description | Examples |
|------|-------|-------------|----------|
| B0 | Nursery | Simple grammar, high-frequency facts | Wikipedia, Children's Stories |
| B1 | Primary | Fluent, general knowledge | News, General Web |
| B2 | High School | Structured narrative, intro coding | Books, Tutorials |
| B3 | Undergrad | Reasoning, non-trivial code | Tech Blogs, Manuals |
| B4 | Graduate | Deep technical, algorithms | Papers, Complex Code |
| B5 | PhD | Advanced abstraction, agent logs | Proofs, Traces |

---

## PLAN 1: The "Lazy" Strategy (Source-Based Proxy)

**Philosophy:** Use folder names as a proxy for difficulty. Trust the data provider's cleaning.

**Data Cleaning:** None. (Zero compute spent).

**Data Action:** Delete IndicCorpV2 and Sangraha-Unverified immediately. They are too dirty for this plan.

### Stage 1: The 1B Model (Basics)

- **Goal:** Grammatical stability.
- **Active Bands:** B0 (60%), B1 (40%).
- **Dataset Configuration:**
  - English: Load Dolma & FineWeb-Edu. (Pure B0/B1).
  - Indic: Load Sangraha & FineWeb2.
  - Excluded: Books, Code, Papers.

> **Why:** Wikipedia/Edu data is grammatically perfect. It acts as the "Nursery."

### Stage 2: The 3B Model (Structure)

- **Goal:** Long context and narrative flow.
- **Active Bands:** B0 (30%), B1 (40%), B2 (30%).
- **Dataset Configuration:**
  - Add English: Dolma. (Introduces B2 structure).
  - Add Indic: Sangraha (Translated Textbooks) & Sangraha.
  - Add Code: Dolma (Limit to top 1% popularity via metadata if available).

> **Why:** Books force the model to attend to longer sequences. Synthetic data teaches structure.

### Stage 3: The 8B Model (Reasoning)

- **Goal:** Logic, Math, and Coding proficiency.
- **Active Bands:** B1 (20%), B2 (30%), B3 (40%), B4 (10%).
- **Dataset Configuration:**
  - Add English: Dolma (Scientific Papers) & Dolma.
  - Add Code: Dolma (Full access).
  - Indic: FineWeb2 (Full Indic subset).

> **Why:** Scientific papers and raw code introduce complex logic (B3/B4) now that the model doesn't hallucinate grammar.

### Stage 4: The 70B Model (Complexity)

- **Goal:** Scale and long-tail knowledge.
- **Active Bands:** All Bands (B0-B5).
- **Dataset Configuration:**
  - Load every folder available in Dolma, FineWeb2, and Sangraha Verified.

> **Why:** The model has enough capacity to handle noise and learn from rare examples.

---

## PLAN 2: The "High-ROI" Strategy (Filtered Proxy)

**Philosophy:** Trust sources but verify quality. Use cheap heuristics to "unlock" dirty datasets.

**Data Cleaning Required:** Yes (CPU-based).

### Preprocessing Checklist (Plan 2 Only)

Before training, run these 4 steps:

1. **Global MinHash Dedup:** Remove documents in FineWeb/Dolma that appear in Sangraha. (Prioritize Sangraha).
2. **Gopher Rules:** Filter IndicCorpV2 and Sangraha-Unverified. Remove if symbol ratio > 40%, word count < 50, or stop-words missing.
3. **KenLM Scoring:** Train a tiny language model on Wikipedia. Remove the bottom 10% high-perplexity docs from FineWeb2.
4. **Decontamination:** Remove text matching MMLU/HumanEval test sets.

### Stage 1: The 1B Model (Basics)

- **Goal:** Grammatical stability + Native Diversity.
- **Active Bands:** B0 (50%), B1 (50%).
- **Dataset Configuration:**
  - English: FineWeb-Edu (Deduped).
  - Indic (Enhanced): Sangraha + IndicCorpV2 (Filtered).
  - **Difference:** Plan 2 uses IndicCorpV2. This provides "street-level" native Hindi (B1) that isn't just translated Wikipedia.

> **Cleaning Impact:** The Gopher rules ensure the IndicCorp data doesn't crash the loss curve with garbage.

### Stage 2: The 3B Model (Structure)

- **Goal:** Long context + Diverse Knowledge.
- **Active Bands:** B0 (20%), B1 (30%), B2 (40%), B3 (10%).
- **Dataset Configuration:**
  - English: Dolma (Deduped) + Dolma.
  - Indic (Enhanced): Sangraha + Sangraha.
  - **Difference:** We unlock the massive "Unverified" folder because KenLM filtering proved it is readable. This doubles the data volume for the 3B model.
  - Code: Dolma (Filtered for length/complexity).

### Stage 3: The 8B Model (Reasoning)

- **Goal:** Logic, Math, and Clean Code.
- **Active Bands:** B1 (15%), B2 (25%), B3 (40%), B4 (20%).
- **Dataset Configuration:**
  - English: Dolma (Papers) + FineWeb2 (Perplexity Filtered).
  - Code (Enhanced): Dolma (Aggressively Deduped).
  - **Difference:** Code deduplication prevents the "repetition penalty" issue. The model learns logic, not memorization.
  - Indic: FineWeb2 (Full Indic Filtered).

### Stage 4: The 70B Model (Complexity)

- **Goal:** Scale + Agentic Traces.
- **Active Bands:** All Bands (B0-B5).
- **Dataset Configuration:**
  - All clean data + Curated Agent Traces (if available/scraped).
  - **Filtering Relaxed:** We relax the perplexity filter slightly to allow "creative" or "weird" text (B5) that might have been filtered earlier.

---

## Summary Comparison

| Feature | Plan 1 (Lazy) | Plan 2 (High-ROI) |
|---------|---------------|-------------------|
| **Cleaning** | None. (Folder selection only). | MinHash Dedup, Gopher Rules, KenLM. |
| **Indic Data** | Sangraha Verified + FineWeb2. | Adds IndicCorpV2 & Sangraha Unverified. |
| **Code Quality** | Noisy (Risk of duplicates). | High (Deduped & Filtered). |
| **Hindi Tail** | Weak (Mostly formal/translated). | Strong (Native "street" text included). |
| **Compute Cost** | Zero Pre-processing. | ~2-3 Days of CPU Cluster time. |
| **Recommended** | If you have 0 resources. | If you have some CPU resources. |


---

## References

### Dolma Dataset

- [Discussion on Downloading different segments of data](https://huggingface.co/datasets/allenai/c4/discussions/10)
- [List of different segments](https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_6.txt)

### FineWeb-2 & IndicCorpV2

- [FineWeb-2 Repository](https://github.com/huggingface/fineweb-2/tree/main?tab=readme-ov-file)
- [FineWeb-2 Reference Datasets](https://github.com/huggingface/fineweb-2/tree/main/misc/reference_datasets)
- [Why not FineWeb2 over IndicCorpV2 (Discussion)](https://chatgpt.com/c/697c3084-f31c-832d-9b62-4b995dc68b5c)

### Utility Tools

- [Datatrove](https://github.com/huggingface/datatrove) — Data processing utilities