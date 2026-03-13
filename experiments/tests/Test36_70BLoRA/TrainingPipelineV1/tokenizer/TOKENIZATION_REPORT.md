# Tokenization Pipeline Report

**Date**: 2026-03-09
**Tokenizer**: 131K Kronecker-aware multilingual BPE (vocab=131,072)
**Instance**: Storage-optimized CPU instance (128 vCPUs, 1 TB RAM, NVMe local storage)
**S3 Bucket**: `s3://t1-dataacquisition-datasets-2/shards/`

---

## Pipeline Summary

| Metric | Value |
|--------|-------|
| **Total tokens** | **~900B** (896.7B across all batches) |
| **Total shards** | **33,382** (128 MB each, ~4.2 TB) |
| **Total docs processed** | **~2.79 billion** |
| **Docs cleaned/kept** | **~2.28 billion** (81.6%) |
| **Docs dropped** | **~509M** (mostly empty/minified) |
| **Throughput** | 52–56M chars/s, 12–16M tok/s |
| **Bands** | 7 (B0–B6) |
| **Shard format** | `tokens.bin` (uint32) + `tokens.idx` (uint64 offsets) + `metadata.json` |
| **Block size** | 4,096 tokens |
| **Blocks per shard** | 8,192 (= 33.5M tokens/shard) |

---

## Per-Band Breakdown

| Band | Purpose | Shards | ~Tokens | Key Sources |
|------|---------|--------|---------|-------------|
| **B0** | Low-quality web / short Indic | 4,894 | **~164B** | cc_head (259M docs), reddit (317M docs), erav4_lang_* (6 Indic langs), samvaad_hi |
| **B1** | General web clean | 20,158 | **~675B** | cc_tail (483M), cc_middle (378M), refinedweb (487M), C4 (164M), ai-bharath (210M), sangraha_* (40M), cc_news, megawika |
| **B2** | High-quality curated | 934 | **~31B** | flan (41M docs), stackexchange (29M), books (56K), ncert (120K), sarvamai_mmlu |
| **B3** | Code | 5,869 | **~197B** | Starcoder (200M docs), pes2o (4M docs) |
| **B4** | Math/Science | 1,146 | **~38B** | open_web_math (2.7M), redpajama-arxiv (1.5M) |
| **B5** | Advanced math | 318 | **~11B** | proof_pile_2-algebraic_stack (2.7M docs) |
| **B6** | Curated synthetic | 63 | **~2.1B** | finephrase (3M docs) |

---

## Per-Batch Processing

| Batch | Sources | Files | Time | Shards | Tokens | Notes |
|-------|---------|-------|------|--------|--------|-------|
| 0 — FineWeb head | cc_head | 3,182 | 2.5h | 3,177 | 106.3B | 259M docs, B0 |
| 1 — CC tail | cc_tail | 3,696 | 3.8h | 4,952 | 165.7B | 483M docs, B1 |
| 2 — CC middle | cc_middle | 4,101 | 4.2h | 5,014 | 167.9B | 378M docs, B1 |
| 3 — RefinedWeb | refinedweb | ~4,000 | 6.0h | 7,430 | 248.8B | 487M docs, B1 |
| 4 — Mixed | C4, Starcoder, reddit, pes2o | 3,137 | 5.9h | 9,169 | 306.9B | 4 sources, B0/B1/B3 |
| 5 — Small sources | 38 sources (Indic, math, books, flan, etc.) | 7,844 | 2.1h | 3,577 | 119.6B | Multi-band |
| 6 — Finephrase | finephrase | 77 | 6min | 63 | ~2.1B | B6, curated synthetic |

---

## Per-Source Detail (Batch 4 — Mixed)

| Source | Band | Docs | Median Chars | Entropy | Instruction% | Code Score | Math Score |
|--------|------|------|-------------|---------|-------------|------------|------------|
| C4 | B1 | 163,763,421 | 928 | 7.37 | 0.1% | 0.000 | 0.0002 |
| Starcoder | B3 | 200,361,619 | 1,278 | 7.59 | 54.1% | 0.086 | 0.0027 |
| pes2o | B3 | 4,109,556 | 641 | 7.39 | 0.5% | 0.000 | 0.0001 |
| reddit | B0 | 317,281,969 | 659 | 7.30 | 0.2% | 0.000 | 0.0002 |

## Per-Source Detail (Batch 5 — Small Sources)

| Source | Band | Docs | Median Chars | Entropy | Instruction% | Code Score | Math Score |
|--------|------|------|-------------|---------|-------------|------------|------------|
| ai-bharath-comparable | B1 | 4,352,263 | 218 | 7.03 | 0.0% | 0.000 | 0.0001 |
| ai-bharath-daily | B0 | 137,668 | 91 | 6.27 | 0.0% | 0.000 | 0.0000 |
| ai-bharath-ilci | B1 | 1,342,259 | 186 | 6.96 | 0.0% | 0.000 | 0.0000 |
| ai-bharath-massive | B0 | 114,250 | 79 | 6.02 | 0.0% | 0.000 | 0.0000 |
| ai-bharath-nllb_filtered | B1 | 82,268,073 | 100 | 6.43 | 0.0% | 0.000 | 0.0001 |
| ai-bharath-samanantar | B1 | 121,591,752 | 93 | 6.36 | 0.0% | 0.000 | 0.0000 |
| ai-bharath-wiki | B1 | 644,008 | 218 | 7.15 | 0.0% | 0.000 | 0.0001 |
| books | B2 | 55,606 | 264,114 | 8.07 | 95.2% | 0.001 | 0.0001 |
| cc_news | B1 | 8,709,830 | 1,719 | 7.63 | 0.0% | 0.000 | 0.0009 |
| erav4_lang_as | B0 | 168,712 | 45 | 5.35 | 0.0% | 0.000 | 0.0000 |
| erav4_lang_hi | B0 | 177,921 | 46 | 5.26 | 0.0% | 0.000 | 0.0000 |
| erav4_lang_kn | B0 | 228,222 | 49 | 5.34 | 0.0% | 0.000 | 0.0000 |
| erav4_lang_mr | B0 | 201,587 | 47 | 5.32 | 0.0% | 0.000 | 0.0000 |
| erav4_lang_pa | B0 | 213,657 | 56 | 5.45 | 0.0% | 0.000 | 0.0000 |
| erav4_lang_te | B0 | 59,635 | 50 | 5.41 | 0.0% | 0.000 | 0.0000 |
| erav4_math | B1 | 361,482 | 42 | 5.14 | 0.0% | 0.000 | 0.0000 |
| erav4_pattern | B1 | 83,080 | 53 | 5.29 | 0.0% | 0.000 | 0.0000 |
| flan | B2 | 41,440,083 | 542 | 7.25 | 0.8% | 0.000 | 0.0001 |
| megawika | B1 | 5,975,788 | 1,040 | 7.44 | 0.3% | 0.000 | 0.0001 |
| ncert | B2 | 120,376 | 965 | 7.45 | 0.1% | 0.000 | 0.0001 |
| proof_pile_2-algebraic_stack | B5 | 2,692,346 | 3,135 | 7.81 | 48.9% | 0.030 | 0.0090 |
| proof_pile_2-open_web_math | B4 | 2,728,257 | 3,816 | 7.99 | 48.5% | 0.043 | 0.0122 |
| redpajama-arxiv | B4 | 1,487,615 | 44,651 | 8.53 | 83.4% | 0.001 | 0.0276 |
| samvaad_hi | B0 | 87,190 | 2,012 | 7.97 | 93.9% | 0.000 | 0.0000 |
| sangraha_as | B1 | 219,046 | 989 | 7.75 | 0.1% | 0.000 | 0.0000 |
| sangraha_bn | B1 | 7,006,860 | 1,177 | 7.74 | 0.0% | 0.000 | 0.0000 |
| sangraha_gu | B1 | 2,170,230 | 1,219 | 7.74 | 0.0% | 0.000 | 0.0000 |
| sangraha_hi | B1 | 9,958,079 | 1,264 | 7.66 | 0.0% | 0.000 | 0.0000 |
| sangraha_kn | B1 | 2,568,121 | 1,157 | 7.86 | 0.0% | 0.000 | 0.0000 |
| sangraha_ml | B1 | 4,152,043 | 1,023 | 7.66 | 0.0% | 0.000 | 0.0000 |
| sangraha_mr | B1 | 2,772,896 | 1,232 | 7.77 | 0.0% | 0.000 | 0.0000 |
| sangraha_or | B1 | 1,296,088 | 913 | 7.68 | 0.0% | 0.000 | 0.0000 |
| sangraha_pa | B1 | 945,877 | 1,159 | 7.74 | 0.0% | 0.000 | 0.0000 |
| sangraha_ta | B1 | 5,400,145 | 1,208 | 7.43 | 0.0% | 0.000 | 0.0000 |
| sangraha_te | B1 | 3,830,615 | 1,114 | 7.85 | 0.0% | 0.000 | 0.0000 |
| sarvamai_mmlu | B2 | 294,130 | 521 | 7.47 | 0.1% | 0.000 | 0.0005 |
| stackexchange | B2 | 28,686,983 | 1,607 | 7.78 | 41.4% | 0.030 | 0.0047 |

## Per-Source Detail (Batch 6 — Finephrase)

| Source | Band | Docs | Median Chars | Entropy | Instruction% | Code Score | Math Score |
|--------|------|------|-------------|---------|-------------|------------|------------|
| finephrase | B6 | 2,975,299 | 2,207 | 7.63 | 0.0% | 0.000 | 0.0001 |

---

## Cleaning Statistics (All Batches Combined)

| Cleaning Operation | Count |
|-------------------|-------|
| **Total docs processed** | 2,790,244,493 |
| **Docs dropped — empty** | 618,877,468 |
| **Docs dropped — minified** | 400,534,792 |
| **Docs dropped — autogenerated** | 2,692,608 |
| **Docs dropped — mojibake** | 278,693 |
| **Docs dropped — short** | 4,750,032 |
| **Docs dropped — lockfile** | 960 |
| **Citation markers stripped** | 795,271,129 |
| **Ghost tags removed** | 27,870,750 |
| **License headers stripped** | 2,137,718 |
| **Autogen warnings stripped** | 339,563 |
| **C0/C1 control chars removed** | 25,981,035 |
| **Zero-width/BiDi chars removed** | 45,745,068 |
| **PUA chars removed** | 4,910,533 |
| **Reference sections stripped** | 78,083 |

---

## Band Design Rationale

- **B0 (Low quality / short)**: Web crawl head (lower quality CC pages), short Indic sentence pairs, reddit. Used for early pretraining warm-up.
- **B1 (General web clean)**: Bulk of the corpus. CC tail/middle (higher quality pages), RefinedWeb, C4, ai-bharath parallel corpora, sangraha Indic web, news. Core pretraining data.
- **B2 (High-quality curated)**: Instruction-following (FLAN), Q&A (StackExchange), books, educational (NCERT), evaluation (MMLU). Upsampled in later pretraining phases.
- **B3 (Code)**: Starcoder code corpus, scientific papers (pes2o). Code-specific pretraining phase.
- **B4 (Math/Science)**: OpenWebMath, arXiv papers. High math density for quantitative reasoning.
- **B5 (Advanced math)**: Proof Pile 2 algebraic stack. Formal mathematics and proofs.
- **B6 (Curated synthetic)**: Finephrase — high-quality curated synthetic text for final pretraining refinement.

---

## Technical Details

### Tokenizer
- **Vocab size**: 131,072
- **EOS ID**: 130717, **PAD ID**: 130718
- **Type**: Kronecker-aware multilingual BPE
- **Script coverage**: Latin (107,756), Devanagari (4,103), Bengali (2,222), Gujarati (1,803), Malayalam (1,835), Kannada (1,475), Telugu (1,459) + others
- **Merge depth**: max=15, mean=3.2

### Shard Format
Each shard directory contains:
- `tokens.bin` — packed uint32 token IDs (~128 MB)
- `tokens.idx` — uint64 byte offsets for document boundaries
- `metadata.json` — source, band, doc count, token count, stats

### Data Pipeline
1. **Input**: Parquet files from various sources on NVMe
2. **Cleaning**: Mojibake detection, minified code removal, autogenerated content filtering, control char stripping, citation/reference removal, ghost tag cleanup
3. **Tokenization**: 131K BPE with 127 parallel workers
4. **Packing**: Documents packed into fixed-size blocks (4096 tokens), blocks grouped into shards (8192 blocks/shard)
5. **Upload**: Shards uploaded to S3 with band-prefixed directory structure

### S3 Structure
```
s3://t1-dataacquisition-datasets-2/shards/
  band_B0/shard_000000/ {tokens.bin, tokens.idx, metadata.json}
  band_B0/shard_000001/ ...
  ...
  band_B6/shard_033381/ ...
```

### Backup Bucket
`s3://t1-dataacquisition-dataset-shards/` — synced every 15 minutes via cron.
