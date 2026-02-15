# Data Preprocessing Strategy — Questions, Analysis & Decisions

## Context

We have **~4 TB of training data** from four heterogeneous sources, each with different formats, languages, and quality characteristics. Before writing any preprocessing code, we must answer the fundamental questions below. This document rephrases each question clearly, provides analysis with pros and cons, and ends with a recommended decision.

### Our Data Sources

| Source | Description | Format | Approx. Size | Languages | Tokens (est.) |
|--------|-------------|--------|-------------|-----------|---------------|
| **Dolma** | AI2's multi-source web corpus (Common Crawl, C4, peS2o, The Stack, Wikipedia, Gutenberg) | JSONL.gz / Parquet | ~2.5–3 TB | English-dominant | ~1.7 T tokens |
| **Sangraha** | AI4Bharat's Indic language corpus — Verified (web scrape + OCR + transcriptions), Unverified (filtered multilingual), Synthetic (translated Wikimedia) | Parquet | ~500 GB–1 TB | 22 Indian languages | ~251 B tokens |
| **NCERT** | Indian school textbook content — structured educational text, potentially extracted PDF content | CSV / JSONL / raw text | ~5–50 GB | Hindi + English | ~1–5 B tokens |
| **AI4Bharat IndicNLP** | Monolingual text corpus for Indian languages, web-crawled and cleaned | Plain text / Parquet | ~100–500 GB | 22 Indian languages | ~20 B tokens |

**Combined: ~4 TB raw data → ~2 T+ tokens after tokenization**

---

## Question 1: What is the structure of each dataset?

> **Rephrased**: *"What are the schema, fields, and document-level structure of each dataset? What does a single record look like in Dolma, Sangraha, NCERT, and IndicNLP?"*

### Answer

#### Dolma
```jsonl
{
  "id": "cc-en-head-0000-00042",
  "text": "The full document text...",
  "source": "common-crawl",
  "added": "2023-06-01",
  "metadata": {"url": "https://...", "quality_score": 0.87}
}
```
- **Key field for training**: `text`
- **Useful for filtering**: `source`, `metadata.quality_score`

#### Sangraha
```jsonl
{
  "text": "भारत एक विविधताओं से भरा देश है...",
  "source": "verified",
  "language": "hi",
  "url": "https://...",
  "doc_id": "sangraha-hi-00042"
}
```
- **Three subsets**: `verified` (highest quality), `unverified` (perplexity-filtered), `synthetic` (translated)
- **Key field**: `text`, `language`

#### NCERT
```csv
chapter,subject,class,text
"Chapter 1","Science","10","Matter in our surroundings. Everything around us..."
```
- May also be raw text extracted from PDFs
- Structured by subject, class level, chapter
- Small but very high quality, curated educational content

#### IndicNLP Corpus
```
Plain text files, one sentence per line, organized by language:
  indic_nlp/hi/hi_corpus.txt
  indic_nlp/ta/ta_corpus.txt
  ...
```
- Pre-tokenized using IndicNLP tokenizer (can be detokenized)
- Organized by language directory

### Key Takeaway
> Each dataset uses a **different format** (JSONL, Parquet, CSV, plain text). The preprocessing pipeline must handle all four formats and extract a uniform `(text, language, source)` tuple from each.

---

## Question 2: How is the data partitioned within each dataset?

> **Rephrased**: *"How are the files organized within each dataset? Is the data split by language, by source, by size, or randomly? How many files are there and how are they named?"*

### Answer

| Dataset | Partitioning Strategy | Number of Files | File Naming |
|---------|----------------------|-----------------|-------------|
| **Dolma** | By source (common-crawl/, c4/, stack/, etc.), then sharded into many files per source | Thousands | `cc-head-0000.jsonl.gz` |
| **Sangraha** | By quality tier (verified/unverified/synthetic), then by language | Hundreds | `train-00000-of-00XXX.parquet` |
| **NCERT** | By subject and class level | Tens | `class10_science.csv` |
| **IndicNLP** | By language (one directory per language) | 22+ | `{lang}_corpus.txt` |

### Pros and Cons of Source-Separated Partitioning

| Aspect | Pro | Con |
|--------|-----|-----|
| **Filtering** | ✅ Easy to include/exclude specific sources or languages | |
| **Quality control** | ✅ Can apply different quality filters per source | |
| **Training mix** | | ❌ Must explicitly mix sources during preprocessing — skipping this leads to "catastrophic source bias" (model overfits to the last source it sees) |
| **Shard identification** | ✅ If a shard fails, you know which source it came from | |

---

## Question 3: What is the size of individual files across datasets?

> **Rephrased**: *"What is the typical file size of individual data files in each dataset, and how does this affect our staging and processing strategy?"*

### Answer

| Dataset | Per-file Size (compressed) | Per-file Size (uncompressed) | Total Files | Total Size |
|---------|---------------------------|------------------------------|-------------|------------|
| **Dolma** | 200 MB – 1 GB | 1 – 5 GB | ~5,000+ | ~2.5–3 TB |
| **Sangraha** | 500 MB – 2 GB (Parquet) | N/A (columnar) | ~200–500 | ~500 GB–1 TB |
| **NCERT** | 1 MB – 50 MB | 5 – 200 MB | ~20–50 | ~5–50 GB |
| **IndicNLP** | 100 MB – 2 GB | 500 MB – 10 GB | ~22 | ~100–500 GB |

### Implications for Preprocessing

| File Size | Impact | Recommendation |
|-----------|--------|----------------|
| **Very small (<50 MB)** — NCERT | Many small files = high I/O overhead | Merge into larger files before tokenization |
| **Medium (200 MB – 2 GB)** — Dolma, Sangraha | Ideal for parallel processing | Process directly with multiprocessing |
| **Large (>2 GB)** — Some IndicNLP | Single-threaded bottleneck if not split | Stream-process, don't load into memory |

---

## Question 4: Is the data shuffled across files or separated by dataset?

> **Rephrased**: *"Are documents from different sources (Dolma, Sangraha, NCERT, IndicNLP) mixed together within individual files, or does each file contain documents from only one source? Should we mix them?"*

### Answer

**Currently: Each dataset is completely separate. No mixing exists within files.**

- Dolma files contain only Dolma documents
- Sangraha files contain only Sangraha documents (further split by language)
- NCERT files contain only NCERT text
- IndicNLP files contain only IndicNLP text per language

### Should We Mix? — Pros and Cons

| Strategy | Pros | Cons |
|----------|------|------|
| **A. Keep separate, mix during training** (dynamic mixing in DataLoader) | ✅ Flexible — can change mix ratios without re-preprocessing ✅ Easy to experiment with different ratios | ❌ Complex DataLoader logic ❌ Harder to checkpoint/resume deterministically ❌ Non-uniform shard sizes across sources |
| **B. Mix during preprocessing** (interleave documents into unified shards) | ✅ Simple training code — just iterate shard-00000, shard-00001... ✅ Deterministic and reproducible ✅ Uniform shard sizes ✅ Easy checkpoint/resume | ❌ Must re-run preprocessing if mix ratio changes ❌ One-time cost is higher |
| **C. Hybrid** — pre-mix the majority training set, keep special datasets (NCERT) separate for curriculum phases | ✅ Best of both worlds | ❌ Slightly more complex preprocessing |

### ✅ Recommended Decision: **Option B (pre-mix during preprocessing)** for the main training run, with the flexibility to create separate shard sets for curriculum learning phases.

### Recommended Mixing Ratios (starting point)

| Source | Ratio | Reasoning |
|--------|-------|-----------|
| **Dolma** (English web + code + academic) | 55% | Strong English foundation |
| **Sangraha Verified** (high-quality Indic) | 20% | Core Indic language capability |
| **Sangraha Unverified** (filtered Indic) | 10% | Additional Indic coverage |
| **IndicNLP Corpus** | 10% | Broader Indic web content |
| **NCERT** (educational) | 3% | Structured knowledge, upsampled for quality |
| **Sangraha Synthetic** (translated) | 2% | Cross-lingual alignment |

> [!NOTE]
> These ratios are starting points. You should experiment with different values based on evaluation on Indic language benchmarks.

---

## Question 5: Should we tokenize in real-time or pre-tokenize offline?

> **Rephrased**: *"Should tokenization happen during training (on-the-fly as the model consumes data), or should we tokenize the entire dataset offline first and store the pre-tokenized data on S3 for the training pipeline to consume directly?"*

### Answer — Pros and Cons

| Strategy | Pros | Cons |
|----------|------|------|
| **A. Real-time tokenization** (tokenize during training) | ✅ No preprocessing step needed ✅ Can change tokenizer without re-processing ✅ Works well for small datasets | ❌ **CPU bottleneck** — tokenization is slow (~100K tokens/sec/core) ❌ 8× H200 GPUs consume ~4M tokens/sec — CPU can't keep up ❌ Wastes expensive GPU hours ($98/hr for P5en) waiting for CPU ❌ Non-deterministic (if using streaming with shuffling) |
| **B. Offline pre-tokenization** (tokenize once, store on S3) | ✅ **Zero tokenization cost during training** ✅ Training runs at full GPU speed ✅ Deterministic and reproducible ✅ Can memory-map .npy files (zero-copy reads) ✅ How OLMo, LLaMA, GPT all do it at scale | ❌ One-time preprocessing cost (hours on a CPU cluster) ❌ Must re-tokenize if changing the tokenizer ❌ Additional S3 storage cost (~$100/month for 4 TB) |

### Cost Analysis

```
Real-time tokenization waste:
  GPU hourly cost:          $98.32/hr (p5en.48xlarge on-demand)
  GPU utilization loss:     ~30-50% (GPUs wait for CPU tokenization)
  Training time:            ~72 hours for 2T tokens
  Wasted cost:              72 × $98.32 × 0.4 = ~$2,832 wasted

Offline pre-tokenization:
  CPU cost (96-core instance): ~$4/hr × 10 hours = $40
  S3 storage:                  ~$92/month (4 TB at $0.023/GB)

  Net savings:                 ~$2,700+ per training run
```

### ✅ Recommended Decision: **Option B — Offline pre-tokenization. Always.**

This is not a close call. Every serious LLM training pipeline (OLMo, LLaMA, GPT, Mixtral) pre-tokenizes offline.

---

## Question 6: Can we shard the data upfront before training?

> **Rephrased**: *"Should we partition the data into fixed-size shards before feeding it to the DataLoader, so that each shard is an identifiable, self-contained unit? This would allow us to track exactly which shard failed during training and resume from that specific shard on restart."*

### Answer — Pros and Cons

| Strategy | Pros | Cons |
|----------|------|------|
| **A. No pre-sharding** (one huge file or many unstructured files) | ✅ Simpler preprocessing | ❌ Cannot identify failure point ❌ Cannot resume precisely ❌ Single file = single point of failure ❌ Cannot memory-map efficiently |
| **B. Pre-shard into fixed-size .npy files** | ✅ **Each shard is an identifiable unit** ✅ On failure: know exactly which shard was active ✅ On restart: resume from (shard_idx, seq_offset) ✅ **Same data ordering guaranteed across restarts** ✅ Can memory-map individual shards (constant RAM) ✅ Can evict consumed shards to free disk space ✅ Parallel downloads from S3 | ❌ Must decide shard size upfront ❌ Slight data loss at shard boundaries (partial sequences skipped) |

### ✅ Recommended Decision: **Option B — Pre-shard into fixed-size .npy files.**

### Recommended Shard Size

| Shard Size | Tokens per Shard | File Size (.npy, uint32) | Num Shards (for 2T tokens) | Pros | Cons |
|------------|-----------------|--------------------------|----------------------------|------|------|
| 100M tokens | 100,000,000 | ~400 MB | ~20,000 | Fine-grained resume, fast download | Too many files, high S3 listing overhead |
| **500M tokens** | **500,000,000** | **~2 GB** | **~4,000** | **Good balance** | **Recommended** |
| 1B tokens | 1,000,000,000 | ~4 GB | ~2,000 | Fewer files | Coarser resume granularity |

**Recommendation: 500M tokens per shard (~2 GB .npy file, ~4,000 shards total)**

---

## Question 7: Should we shard before or after tokenization?

> **Rephrased**: *"At which stage of the preprocessing pipeline should sharding occur — should we split the raw text into shards first and then tokenize each shard, or should we tokenize first and then partition the resulting token stream into fixed-size shards?"*

### Answer — Pros and Cons

| Strategy | Pros | Cons |
|----------|------|------|
| **A. Shard first, tokenize after** (split raw text → tokenize each shard) | ✅ Easier to parallelize (each shard is an independent tokenization job) | ❌ Inconsistent shard sizes (text compresses differently by source/language) ❌ Each shard needs a tokenizer at training time if not pre-tokenized ❌ Hindi text vs English text → wildly different token counts for same byte size ❌ Document boundaries may split awkwardly |
| **B. Tokenize first, shard after** (tokenize all → split token stream into fixed-size shards) | ✅ **Uniform shard sizes** (exactly N tokens per shard) ✅ Training code is trivial — each shard is a flat array of integers ✅ Memory-map reads are instant (no parsing needed) ✅ Deterministic: shard-00042 always contains the same tokens | ❌ Tokenization step processes all data before sharding begins ❌ Intermediate storage needed for the token stream |

### ✅ Recommended Decision: **Option B — Tokenize first, then shard.**

### The Correct Processing Order

```
Step 1: INGEST       Raw sources → Unified format (text, lang, source)
         ↓
Step 2: CLEAN        Quality filtering, dedup, language detection
         ↓
Step 3: MIX          Interleave documents from all 4 sources at desired ratios
         ↓
Step 4: TOKENIZE     Text → token IDs (using your chosen tokenizer)
         ↓                  Insert EOS tokens between documents
         ↓                  No padding tokens
         ↓
Step 5: SHARD        Split the flat token stream into fixed-size .npy shards
         ↓                  Each shard = 500M tokens = ~2 GB
         ↓                  shard-00000.npy, shard-00001.npy, ...
         ↓
Step 6: UPLOAD       Upload shards to S3 in sorted order
         ↓
Step 7: TRAIN        S3Stager → NVMe → StreamingTokenDataset → PrefetchDataLoader → GPU
```

---

## Complete Preprocessing Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES (4 TB)                         │
├──────────┬──────────┬──────────┬────────────────────────────────────┤
│  Dolma   │ Sangraha │  NCERT   │  IndicNLP                         │
│ JSONL.gz │ Parquet  │ CSV/TXT  │  Plain Text                       │
│  ~3 TB   │  ~700GB  │  ~30GB   │  ~300GB                           │
└────┬─────┴────┬─────┴────┬─────┴────┬───────────────────────────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: SOURCE READERS (format-specific)                           │
│                                                                     │
│  read_dolma(path)    → yields (text, "en", "dolma")                │
│  read_sangraha(path) → yields (text, lang, "sangraha-{tier}")      │
│  read_ncert(path)    → yields (text, lang, "ncert")                │
│  read_indicnlp(path) → yields (text, lang, "indicnlp")            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: QUALITY FILTER + DEDUP                                     │
│                                                                     │
│  • Min document length (skip very short docs)                      │
│  • Language detection verification                                  │
│  • Near-duplicate removal (MinHash / SimHash)                      │
│  • Perplexity filtering for IndicNLP/Sangraha Unverified           │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: MIX (interleave at specified ratios)                       │
│                                                                     │
│  Stream from all sources simultaneously:                            │
│    55% Dolma | 20% Sangraha-V | 10% Sangraha-U | 10% IndicNLP     │
│    3% NCERT | 2% Sangraha-S                                        │
│                                                                     │
│  Shuffle order within a buffer (e.g., 100K document buffer)        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: TOKENIZE                                                   │
│                                                                     │
│  for doc in mixed_stream:                                           │
│      tokens = tokenizer.encode(doc.text)                           │
│      token_buffer.extend(tokens)                                   │
│      token_buffer.append(EOS_TOKEN_ID)    ← Document boundary      │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: SHARD (fixed-size .npy files)                              │
│                                                                     │
│  while token_buffer has data:                                       │
│      shard = token_buffer[:500_000_000]    ← 500M tokens           │
│      np.save(f"shard-{idx:05d}.npy", shard.astype(np.uint32))     │
│      idx += 1                                                       │
│                                                                     │
│  Output: shard-00000.npy (2 GB)                                    │
│          shard-00001.npy (2 GB)                                    │
│          ...                                                        │
│          shard-03999.npy (2 GB)                                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: UPLOAD TO S3                                               │
│                                                                     │
│  s3://my-bucket/training-data/shard-00000.npy                      │
│  s3://my-bucket/training-data/shard-00001.npy                      │
│  ...                                                                │
│  s3://my-bucket/training-data/shard-03999.npy                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Decision Summary

| # | Question | Decision | Confidence |
|---|----------|----------|------------|
| 1 | Data structure? | Multi-format — need source-specific readers that output uniform (text, lang, source) tuples | Factual |
| 2 | How partitioned? | By source + language — we must explicitly mix them during preprocessing | Factual |
| 3 | File sizes? | 1 MB to 5 GB — need to handle both tiny (NCERT) and large (Dolma) files | Factual |
| 4 | Shuffled or separate? | **Pre-mix during preprocessing** into unified shards | ⭐ Strong recommendation |
| 5 | Tokenize when? | **Offline pre-tokenization, always** — saves ~$2,700+ per training run | ⭐ Non-negotiable |
| 6 | Pre-shard? | **Yes — 500M tokens per shard (~2 GB .npy files)** — enables shard-level fault tolerance | ⭐ Strong recommendation |
| 7 | Shard before or after tokenization? | **After tokenization** — gives uniform shard sizes and instant mmap reads | ⭐ Strong recommendation |

---

## Next Step

The next piece of code to build is the **offline preprocessing script** that implements Steps 1–6 above:

```
scripts/preprocess_data.py
    ├── Source readers (Dolma, Sangraha, NCERT, IndicNLP)
    ├── Quality filter + dedup
    ├── Multi-source mixer (with configurable ratios)
    ├── Tokenizer (BPE, with EOS insertion)
    ├── Fixed-size shard writer (.npy)
    └── S3 uploader
```

This script runs **once** on a large CPU machine (e.g., c7i.48xlarge, ~$8/hr) and produces the ready-to-train shards on S3.
