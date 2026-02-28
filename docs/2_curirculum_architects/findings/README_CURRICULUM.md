# Curriculum Design Strategy

This document outlines the strategies and methodologies for arriving at the final curriculum for multi-stage LLM pretraining (1B → 3B → 8B → 70B).

---

## 1. Document Classification Pipeline

### 1.1 Metadata Extraction

All documents are tagged with standardized metadata from source dataset:
- **source_dataset**: Origin dataset (dolma, fineweb, sangraha)
- **doc_id**: Unique identifier
- **language**: Detected language (en, indic, other)
- **url**: Source URL (when available)
- **source_subsection**: Dataset subsection (e.g., Dolma's stack, arxiv)

### 1.2 Signal Extraction

Documents are analyzed for multiple high-level difficulty and capability signals.  
Signals are derived from combinations of underlying document-level features (listed below in 1.3). These features feed into the band, modality, and domain assignment.

| Signal | Correlation | Detection Method |
|--------|-------------|-----------------|
| Avg sentence length | Higher difficulty | Aggregated sentence statistics |
| Rare token ratio | Higher difficulty | Token frequency |
| Flesch-Kincaid grade | Readability proxy | Syllable counting |
| Code blocks | B3+ | Code-token density + regex |
| Math symbols | B4+ | Math symbol density |
| CoT markers | B3+ reasoning | Reasoning marker aggregation |
| Research paper structure | B4+ | Citation + section patterns |
| Agentic traces | B5 | Agent marker + JSON structure |


### 1.3 Underlying Document-Level Features
 
These features are extracted programmatically for all documents

| Feature | Description                     | Used By Signals |
|--------|---------------------------------|-----------------|
| doc_token_count | Total tokens in document        | Length normalization |
| sentence_len_avg | Mean tokens per sentence        | Avg sentence length |
| sentence_len_std | Variance in sentence length     | Difficulty estimation |
| unique_token_ratio | Unique / total tokens           | Rare token ratio |
| token_entropy | Entropy over token distribution | Difficulty, quality |
| flesch_kincaid_grade | FK grade level                  | Readability proxy |
| code_token_ratio | Fraction of code-like tokens    | Code blocks |
| math_symbol_ratio | Math symbols / tokens           | Math symbols |
| citation_count | Citation-like patterns          | Research structure |
| repetition_score | Repeated n-gram score           | Noise filtering |
| gzip_compression_ratio | Compressed / raw size           | Redundancy detection |

[Sample feature extractor](../../experiments/2_curirculum_architects/feature_extractor.py)  



### 1.4 Multi-Dimensional Classification

Each document receives **three independent labels**:

**Band (B0-B5)**: Difficulty level
- B0 (Nursery): readability < 6, simple structure
- B1 (Primary): readability 6-10
- B2 (High School): readability 10-14
- B3 (Undergraduate): code or readability > 14
- B4 (Graduate): math symbols + complexity
- B5 (PhD): agentic traces or advanced reasoning

**Modality**: Content type
- `general_text` (default)
- `code`
- `math`
- `cot_reasoning`
- `research_papers`
- `agentic_traces`

**Domain**: Subject area (source-based mapping)
- `general_web_clean`
- `encyclopedic` (Wikipedia)
- `code_repos` (GitHub/Stack)
- `math_science` (arXiv)
- `technical_docs`
- `news_nonpolitical`
- `dialogue_chat`
- `planning_reasoning_curated`


### 1.5 Processing Flow and Storage

Each document is processed in a single pass:

1. The document is read once, and all **underlying low-level features** (tokens, sentence statistics, structural cues, lexical features, etc.) are computed.
2. These features are then aggregated to compute the **higher-level modalities**: 
   - Band (B0-B5)  
   - Modality (general_text, code, math, etc.)  
   - Domain (source-based classification)
3. Only the **dataset metadata** and the **higher-level modalities** are stored back at the document level.  
   - Low-level features are used internally for signal computation but are **not saved** to reduce storage and maintain privacy.
---

## 2. Language Filtering

**Supported languages**: English, Indic languages (Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Oriya)

**Detection method**:
1. Use dataset metadata when available (FineWeb language field)
2. Fallback: Unicode script range detection (fast, no external dependencies)
3. Reject all other languages

---

## 3. Stage-Specific Constraints

Constraints enforce curriculum policy and prevent capability degradation:

| Constraint | 1B | 3B | 8B | 70B | Justification |
|------------|----|----|----|----|---------------|
| **Indic languages** | ❌ | ✓ | ✓ | ✓ | Establish English foundation first (BLOOM 2022) |
| **Agentic traces** | ❌ | ❌ | B4-B5 only | B4-B5 only | Requires baseline reasoning (Toolformer 2023) |
| **CoT caps (B3)** | 3% | 4% | 5% | 5% | Prevent formatting overfitting (Phi-3 2024) |
| **CoT caps (B4)** | 6% | 7% | 8% | 8% | |
| **CoT caps (B5)** | 8% | 9% | 10% | 10% | |
| **Global CoT cap** | 6% | 6% | 6% | 6% | Maintain chat quality |
| **Global agentic cap** | 0% | 0% | 3% | 3% | Limited pretraining exposure |

Documents violating constraints are rejected at classification time.

---

## 4. Band Proportion Calculation

### 4.1 Base Distribution from Natural Corpus

First, we compute the **empirical base distribution** by classifying all documents and measuring token counts per band:

```
base_distribution(b) = total_tokens_in_band(b) / total_corpus_tokens
```

This reflects the natural difficulty distribution of the corpus before any curriculum adjustments.

### 4.2 Capacity-Aligned Growth Model

**Step 1: Difficulty Quantile Assignment**

Bands represent corpus difficulty percentiles:

| Band | Percentile | Difficulty Centroid (d_b) |
|------|-----------|---------------------------|
| B0 | 0–15% | 0.10 |
| B1 | 15–30% | 0.225 |
| B2 | 30–50% | 0.40 |
| B3 | 50–70% | 0.60 |
| B4 | 70–85% | 0.775 |
| B5 | 85–100% | 0.925 |

**Step 2: Model Capacity Scaling**

Capacity grows logarithmically with parameters:

```
capacity(stage) = [log(params) - log(1B)] / [log(70B) - log(1B)]
```

| Stage | Params | Capacity |
|-------|--------|----------|
| 1B | 1B | 0.00 |
| 3B | 3B | 0.26 |
| 8B | 8B | 0.49 |
| 70B | 70B | 1.00 |

**Step 3: Alignment Weighting**

For each band at each stage, compute alignment weight:

```
alignment_weight(b, s) = exp(-λ × |d_b - c_s|)
```

Where λ = 3.0 (alignment sharpness parameter)

This peaks when difficulty matches capacity, decays when they diverge.

**Step 4: Raw Weight Computation**

Combine base distribution with alignment:

```
raw_weight(b, s) = base_distribution(b) × alignment_weight(d_b, c_s)
```

**Step 5: Apply Floors and Caps**

Enforce minimum exposure floors to prevent capability gaps:

| Band | Floor |
|------|-------|
| B0 | 10% |
| B1 | 14% |
| B2 | 18% |
| B3 | 14% |
| B4 | 6% |
| B5 | 2% |

**Step 6: Renormalization**

Normalize all band weights to sum to 100% for each stage.

### 4.3 Theoretical Justification

Grounded in:
- **Curriculum learning** (Bengio et al., ICML 2009): gradual difficulty progression improves optimization
- **Competence-based pacing** (Platanios et al., ICML 2019): data selection based on difficulty-capacity alignment
- **Scaling laws** (Kaplan et al. 2020): log-parameter capacity modeling

---

## 5. Guardrails

### Anti-Domain-Spike
- Max 25% of any domain in rolling 2M token window
- EMA smoothing prevents sudden shifts

### Modality Caps
- CoT: 6% global, band-specific caps
- Agentic: 3% global
- Indic: 8% global

### Quality Gates
- Minimum 40 tokens per document
- Language confidence requirements
- Rejection tracking for audits

---

## 6. Rejection Strategy

Documents are rejected (excluded from training) if they fail quality or policy checks:

### 6.1 Language Rejection
- **Rule**: Reject if language is not English or Indic
- **Detection**: Unicode script range analysis + metadata
- **Reason code**: `language_not_en_or_indic`

### 6.2 Stage-Specific Rejections
- **Indic at 1B**: Rejected at 1B stage (allowed from 3B onward)
- **Agentic before B4**: Rejected if agentic modality appears in B0-B3
- **Reason codes**: `indic_not_allowed_at_{stage}`, `agentic_not_allowed_in_{band}_at_{stage}`

### 6.3 Quality Filters
- **Minimum length**: Reject if < 40 tokens (prevents fragments)
- **Early exit**: Documents under token threshold assigned to B0 or rejected

### 6.4 Rejection Tracking
All rejections are logged with:
- Original metadata (source, ID)
- Rejection reason
- Stage (if applicable)

This enables audit trails and dataset quality analysis.

---

## 7. Implementation

**Design Principles**:
- Stateless (map-friendly for distributed processing)
- Single-pass (no multi-stage parsing)
- Cheap-first (structural signals before expensive readability)
- Auditable (all decisions logged with metadata)

**Pipeline**:
```
raw_sample → extract_metadata → detect_language → 
extract_signals → assign_band/modality/domain → 
check_stage_constraints → output classification
```

**Outputs**:
- Band, modality, domain labels
- All signals (for debugging)
- Metadata (source, ID, language)
- Rejection reason (if applicable)

---

**Reference Implementation**: [https://colab.research.google.com/drive/1a5ASgJIDi8VbbUvc43uT93DBPB7OXkCg?usp=sharing](https://colab.research.google.com/drive/1Qhch3c5XAqNyQOJDfOsOyoIZVUJuur_-?usp=sharing)


---

## References

1. **Bengio et al. (2009)** - Curriculum Learning, ICML
2. **Platanios et al. (2019)** - Competence-based Curriculum Learning, ICML
3. **Kaplan et al. (2020)** - Scaling Laws for Neural Language Models
4. **BigScience (2022)** - BLOOM: Multilingual staging strategies
5. **Schick et al. (2023)** - Toolformer: Tool use after base capabilities
6. **Microsoft (2024)** - Phi-3: CoT data mixing recommendations

---
