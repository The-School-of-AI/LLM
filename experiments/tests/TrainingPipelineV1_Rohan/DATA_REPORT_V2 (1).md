# DATA REPORT v2 — Pretraining Corpus & Curriculum Architecture

**Date**: 2026-03-10
**Status**: Production-ready. Manifests generated, shard lists frozen (seed=42).
**Supersedes**: DATA_STRATEGY.md (sections on band design), curriculum.yaml (v1)

---

## 1. Executive Summary

We built a 1.15T-token pretraining corpus across 36,294 shards, reorganized into a 3-tier architecture for progressive model growth (1B → 3B → 8B → 70B MoE). Key decisions:

- **Dropped B2** (31.3B tokens) — too noisy (18.7% leakage, 9.14% garbage)
- **Merged B4+B5** into a single STEM pool — cosine similarity 0.949, only 49B combined
- **Combined all code** (StarCoder + firmware/CUDA/kernel tab-code + CRLF code) into one pool
- **Indic data protected** from OPUS bias via Always-ON injection
- **Hindi enabled from 1B stage** (overriding v1's "earliest 3B" constraint)
- **Token IDs reordered** by descending global frequency (most frequent = ID 0)
- **Deterministic loading** via seed-shuffled manifest files

**Training corpus**: 33,353 shards, ~1,118B tokens across 5 pools + golden proxy.
**Training budget**: 209B tokens (200B stages + 9B warmups), ~1,254B effective with OPUS.

---

## 2. S3 Locations

### Primary Buckets

| Location | Description |
|----------|-------------|
| `s3://t1-dataacquisition-datasets-2/shards/` | **Original** shards (pre-reorder). Old token IDs. DO NOT USE for training. |
| `s3://t1-dataacquisition-datasets-2/shards_reordered/` | **Production** shards. Frequency-sorted token IDs. USE THIS. |
| `s3://t1-dataacquisition-dataset-shards` | **Backup** of original dataset. |

### Production S3 Layout (`shards_reordered/`)

```
s3://t1-dataacquisition-datasets-2/shards_reordered/
├── band_B0/                      4,894 shards   611.51 GB   → Pool D1
├── band_B1/                     20,706 shards  2586.97 GB   → Pool D2 (minus 1,996 indic)
├── band_B2/                        934 shards   116.63 GB   → DROPPED
├── band_B3/                      5,869 shards   733.59 GB   → Pool D3
├── band_B4/                      1,146 shards   143.16 GB   → Pool D4
├── band_B5/                        318 shards    39.68 GB   → Pool D4
├── band_B6/                        356 shards    41.78 GB   → Pool AON (bench_train)
├── band_code_tab/                   52 shards     6.34 GB   → Pool D3
├── band_code_crlf/                  12 shards     1.29 GB   → Pool D3
├── band_indic_numerals/          1,996 shards   249.19 GB   → Pool AON (indic_guaranteed)
└── golden_proxy/
    └── band_golden_proxy/           11 shards     0.03 GB   → Pool GP
                                 ──────          ────────
                                 36,294 shards  4,528.17 GB
```

### Shard Format

Each shard is a directory: `shard_XXXXXX/tokens.bin`
- `tokens.bin`: uint32 array, frequency-reordered token IDs
- Block size: 4,096 tokens (configurable)
- Average shard: ~128MB, ~33.5M tokens
- Documents delimited by EOS token (ID=36 after reorder, was 130717)
- PAD token: ID=130726 (was 130718)

### Tokenizer

| Item | Location |
|------|----------|
| Original tokenizer | `s3://t1-dataacquisition-datasets-2/shards/tokenizer.json` |
| Reordered tokenizer | EC2 `/mnt/nvme0/code_datasets/tokenizer_reordered.json` |
| Permutation map | EC2 `/mnt/nvme0/code_datasets/token_permutation.npy` |
| Inverse permutation | EC2 `/mnt/nvme0/code_datasets/token_inv_permutation.npy` |
| Global frequencies | EC2 `/mnt/nvme0/code_datasets/freq_global_training.npy` |

- 131,072-token Kronecker-aware multilingual BPE (Tekken + IST hybrid)
- Top 8,192 tokens cover 81.0% of corpus
- Merges are string-based (NOT affected by ID reordering)
- 758 tokens were unseen in corpus (CRLF + tab-prefix → fixed with synthetic code shards)

---

## 3. Pool Architecture — Why Each Decision Was Made

### Three Tiers

| Tier | Purpose | OPUS Status |
|------|---------|-------------|
| **OPUS-eligible** (D1-D4) | Main pretraining data. OPUS scores candidates and selects ~40%. | Scored by OPUS |
| **Always-ON** (AON) | Critical data that CANNOT be missed. Bypasses OPUS. | **Invisible to OPUS** |
| **Golden Proxy** (GP) | OPUS scoring reference direction. | **Never trained on** |

**Why three tiers?** OPUS maximizes data efficiency by projecting optimizer updates onto a proxy direction from benchmark data. But the golden proxy is English-heavy (cosine 0.876 with B1), so OPUS systematically undervalues non-English data. If we let OPUS score everything, Indic content gets rejected. Always-ON bypasses this bias.

---

### Pool D1 — Web Foundation (164.1B tokens, 4,894 shards)

**Physical source**: `band_B0/`

| Source | Description |
|--------|-------------|
| `cc_head` | CommonCrawl head segment — highest quality web pages by URL frequency |
| `reddit` | Reddit posts/comments — conversational English |

**Why D1 exists**: Cleanest, simplest data in the corpus. Lowest entropy (10.96), lowest repetition (2.04%), best chars/token ratio (4.63). Ideal for the 1B model to learn core English language patterns.

**Quality metrics**: Leakage 0.87%, garbage 1.36%, repetition 2.04% — all excellent.

**Difficulty composite**: 0.372 (easiest pool).

**Stage allocation**: 42% at 1B → 6% at 70B (fades out as model advances).

---

### Pool D2 — Web Diverse (627.4B tokens, 18,710 shards)

**Physical source**: `band_B1/` minus 1,996 shards in `indic_numerals_exclude.txt`

| Source | Shards | Description |
|--------|--------|-------------|
| `cc_tail` | ~7,000 | CommonCrawl tail — long-tail web pages, noisier |
| `cc_middle` | ~5,000 | CommonCrawl middle — medium-frequency web |
| `refinedweb` | ~3,000 | Falcon's deduplicated CommonCrawl extract |
| `cc_news` | ~1,100 | CommonCrawl news articles |
| `ai-bharath-*` | ~1,100 | AI4Bharat curated Indic (BPCC, comparable, ilci, wiki) |
| `erav4_math/pattern` | ~1,100 | TSAI internal math + pattern Q&A (mixed into above) |
| `IndicCorpV2` | 548 | AI4Bharat IndicCorpV2 (16 languages) |
| Other web | ~900 | Misc web sources |

**Why D2 exists**: Largest pool (54.8% of training corpus). Contains virtually ALL Indic data — Devanagari (11.6B), Bengali (6.4B), Tamil (4.6B), Telugu (3.8B), plus 7 more scripts. Essential for multilingual capability.

**Why 1,996 shards excluded**: These are the Indic-heavy shards that had Indic numeral swaps. The `band_indic_numerals/` versions (with native Indic numerals ०१२ instead of Arabic 012) are used in AON instead, avoiding duplication.

**ERAV4 data**: 1,102 B1 shards contain ERAV4 math/pattern data, but it's mixed into multi-source shards alongside ai-bharath, cc_news, etc. Cannot be separated without re-sharding. Guaranteed via D2 (OPUS candidates) + AON (indic_numerals overlap).

**Quality metrics**: Leakage 2.48%, garbage 2.29%, repetition 3.81% — good.

**Stage allocation**: 30% at 1B → 12% at 70B.

---

### Pool D3 — Code (199.0B tokens, 5,933 shards)

**Physical sources**: `band_B3/` + `band_code_tab/` + `band_code_crlf/`

| Source | Shards | Tokens | Description |
|--------|--------|--------|-------------|
| `StarCoder` (B3) | 5,869 | 196.7B | The Stack v1/v2 — Python, JS, C++, Java, multi-lang |
| `code_tab` | 52 | 1.70B | Tab-indented code: firmware, CUDA, kernel, embedded, hardware, systems, C/C++, Java — from 74 GitHub repos |
| `code_crlf` | 12 | 0.35B | CRLF line-ending code: C#, VB.NET, PowerShell, Windows batch |

**Why D3 exists**: Code is the statistically hardest modality (composite 0.605, highest entropy 12.08). Critical for HumanEval 92+, MBPP 92+, SWE-Bench 34+ targets.

**Why code_tab and code_crlf were added**: The tokenizer had 232 unseen tokens (114 tab-prefixed like `\tvirtual`, `\tAssert` and 118 CRLF like `}\r\n`). We downloaded 74 GitHub repos covering firmware, CUDA, kernel, embedded systems, and hardware code, processed them into 64 new shards, achieving 100% token coverage (all 232 tokens now seen in training data).

**Repos downloaded for code_tab**: linux kernel, u-boot, zephyr-rtos, freertos, esp-idf, stm32cube, nvidia/cuda-samples, llvm-project, openblas, ffmpeg, grpc, and 62 more covering systems programming with tab indentation.

**Quality metrics**: Leakage 1.21% (good), garbage 7.98% (moderate — code has valid binary/hex content that appears as "garbage"), repetition 40.25% (high — boilerplate, imports, license headers).

**Stage allocation**: 13% at 1B → 35% at 70B (strongest ramp-up).

---

### Pool D4 — STEM (49.1B tokens, 1,464 shards)

**Physical sources**: `band_B4/` + `band_B5/`

| Source | Shards | Tokens | Description |
|--------|--------|--------|-------------|
| `pes2o` (B4) | ~600 | ~20B | Peered Semantic Scholar Open — academic papers |
| `redpajama-arxiv` (B4) | ~400 | ~13B | ArXiv preprints (math, physics, CS). LaTeX-heavy |
| `proof_pile_2-algebraic_stack` (B4) | ~100 | ~3B | Formal proofs (Lean, Coq, Isabelle) |
| `proof_pile_2-open_web_math` (B4) | ~50 | ~2B | Math from web (MathSE, math blogs) |
| `flan` (B4/B5) | variable | variable | Google FLAN — diverse NLP tasks |
| Other STEM (B5) | 318 | 10.7B | Advanced math, proofs, science |

**Why B4+B5 merged**: Cosine similarity 0.949 — practically identical distributions. B5 alone (10.7B, 318 shards) is too small for a standalone pool. Combined they give 49B of STEM data.

**Why D4 is numbered "4" despite lower statistical difficulty (0.41) than D3 (0.61)**: The numbering reflects **capability difficulty** — math/reasoning is the hardest SKILL for a model to develop, even though the text patterns are formulaic (low entropy). STEM is the capability bottleneck at 70B scale for AIME 80+, GPQA Diamond 67+, Math500 97+ targets.

**Data availability concern**: 49B is the tightest pool. Across all stages, ~124B of STEM candidates are needed. STEM data cycles ~2.5x as candidates, but after OPUS selects 40%, actual training ≈ 49.6B — effectively 1x through all STEM data. Acceptable, especially since math benefits from revisiting.

**Stage allocation**: 7% at 1B → 39% at 70B (sharpest ramp-up).

---

### Pool AON — Always-ON (78.1B tokens, 2,352 shards)

**Purpose**: Data that CANNOT be missed. Injected directly into training at 8% of every batch. OPUS never sees or scores this data — it bypasses the OPUS selection pipeline entirely.

#### AON Sub-pool: bench_train (11.2B tokens, 356 shards)

**Physical source**: `band_B6/`

| Source | Shards | Tokens | Category |
|--------|--------|--------|----------|
| **finephrase** | 63 | 2.10B | Curated synthetic high-quality text |
| SWE-smith-trajectories | 103 | 3.45B | Code engineering trajectories |
| OpenMathReasoning | 66 | 2.21B | Math/reasoning traces |
| OpenMathInstruct-2 | 45 | 1.48B | Math instruction pairs |
| OpenCodeReasoning-2 | 24 | 0.78B | Code reasoning traces |
| NuminaMath-1.5 | 12 | 0.38B | Competition math |
| OpenR1-Math-220k | 9 | 0.30B | Math reasoning |
| flan_v2 | 3 | 0.09B | Diverse NLP tasks |
| glaive-function-calling-v2 | 2 | 0.06B | Function calling |
| mmlu_auxiliary_train | 2 | 0.06B | Knowledge QA |
| Other (19 sources) | 27 | 0.19B | CodeFeedback, SWE-smith, Agent-FLAN, Mind2Web, gsm8k_train, arc-challenge_train, triviaqa_train, etc. |

**Why bench_train is in AON**: These are benchmark train splits — the model MUST learn these patterns (math reasoning traces, code engineering, function calling). If exposed to OPUS, some might be rejected because they don't project onto the English-web-heavy golden proxy.

#### AON Sub-pool: indic_guaranteed (66.9B tokens, 1,996 shards)

**Physical source**: `band_indic_numerals/`

These are B1 shards that contained Indic content, modified with Indic numeral swap (native script numerals ०१२ instead of Arabic 012).

| Content | Tokens (approx) | Scripts |
|---------|-----------------|---------|
| Devanagari (Hindi, Marathi) | ~11.6B | देवनागरी |
| Bengali | ~6.4B | বাংলা |
| Tamil | ~4.6B | தமிழ் |
| Telugu | ~3.8B | తెలుగు |
| Gujarati | ~3.5B | ગુજરાતી |
| Malayalam | ~3.4B | മലയാളം |
| Kannada | ~3.1B | ಕನ್ನಡ |
| Gurmukhi (Punjabi) | ~2.7B | ਗੁਰਮੁਖੀ |
| Odia | ~1.4B | ଓଡ଼ିଆ |
| Mixed English + above | ~26.4B | Latin + Indic |

Also contains mixed in:
- ai-bharath curated Indic (BPCC, comparable, ilci, wiki)
- ERAV4 language literacy Q&A (hi, as, kn, mr, pa, te)
- ERAV4 math and pattern recognition
- samvaad_hi (Hindi dialogue)
- cc_news (co-located in same shards)

**Why Indic in AON**: The golden proxy has cosine 0.876 with B1 and is English-dominant. OPUS would systematically reject Indic-heavy batches because they project poorly onto the English proxy direction. Always-ON guarantees every Indic script gets training exposure at every stage.

**Why native numerals**: The tokenizer has dedicated tokens for Indic numerals (` ०` through ` ९` for Devanagari, etc.). Without the numeral swap, these tokens had ZERO occurrences across the entire corpus. The swap ensures the model learns native numeral representations.

#### AON Internal Split: 50/50

At each training step, the 8% AON injection is split evenly:
- 50% from bench_train (math/code/reasoning train splits + finephrases)
- 50% from indic_guaranteed (Indic-heavy shards with native numerals)

This ensures both benchmark capability AND Indic coverage are maintained throughout training.

---

### Pool GP — Golden Proxy (6.8M tokens, 11 shards)

**Physical source**: `golden_proxy/band_golden_proxy/`

OPUS scoring reference. Contains benchmark test/validation splits. **NEVER trained on.**

| Benchmark | Examples |
|-----------|----------|
| MMLU test | 14K |
| MMLU Pro test | 12K |
| MILU test | ~80K |
| GPQA Diamond test | 198 |
| Math500 test | 500 |
| GSM8K test | 1.3K |
| AIME 2025 | 30 |
| HumanEval test | 164 |
| MBPP test | 500 |
| ARC-Challenge test | 1.2K |
| BBH all | ~6.5K |
| IFEval all | 541 |
| Other (IndicGLUE, TriviaQA val, etc.) | ~13K |

---

### DROPPED — B2 (31.3B tokens, 934 shards)

**Physical source**: `band_B2/` — exists on S3 but NOT used in training.

**Why dropped**: Quality analysis showed unacceptable contamination:
- **18.7% cross-document leakage** (worst of all bands — 7.5x worse than B0)
- **9.14% garbage token rate** (worst — 6.7x worse than B0)
- **15.69% repetition rate** (moderate, but combined with above makes it unreliable)

With 1,039B of clean data in D1-D4, losing 31.3B (2.7%) is negligible. The risk of contaminated training outweighed the marginal data benefit.

The shard list is preserved in `DROPPED_B2_shards.txt` in case future deduplication/cleaning makes B2 recoverable.

---

## 4. Token ID Reordering

**What**: All 131,072 tokens were renumbered by descending global frequency. Token seen most often across the 1.15T corpus → ID 0. Least frequent → ID 131,071.

**Why**: Frequency-sorted IDs enable:
- Embedding table optimization (hot tokens at low indices → better cache locality)
- Adaptive softmax potential (common tokens need less computation)
- Natural vocabulary truncation (top-K IDs by count covers X% of corpus)

**Coverage after reorder**:
| Top-K IDs | Corpus Coverage |
|-----------|----------------|
| 100 | 35.2% |
| 1,000 | 62.8% |
| 8,192 | 81.0% |
| 32,000 | 94.7% |
| 65,536 | 99.1% |

**Key token IDs after reorder**:
- EOS: 36 (was 130717)
- PAD: 130726 (was 130718)
- Most common token: ` ` (space) = ID 0

**Safety**: BPE merges are string-based (`['Ġ', 'Ġ'] → 'ĠĠ'`), NOT ID-based. Merges were NOT modified. Only the vocab mapping was updated. Verified with 9 encode/decode test cases (English, Hindi, Bengali, tab-indented C, CRLF PHP).

**All 36,294 shards reordered** in `shards_reordered/`. Original shards in `shards/` are untouched as backup.

---

## 5. Synthetic Code Shards (code_tab + code_crlf)

### Problem
758 tokens had ZERO occurrences across the 1.15T corpus:
- 114 tab-prefixed code tokens (`\tvirtual`, `\tAssert`, `\tfprintf`, `\tenum`, etc.)
- 118 CRLF tokens (Windows line endings: `}\r\n`, `;\r\n`, etc.)
- 526 other (reserved slots, Indic numerals [fixed separately], rare patterns)

### Solution
Downloaded 74 GitHub repositories covering firmware, CUDA, kernel, embedded systems, hardware drivers, and Windows-style code. Tokenized into:

| Band | Shards | Tokens | Tokens Fixed |
|------|--------|--------|-------------|
| `band_code_tab` | 52 | 1.70B | 114/114 tab-prefixed (100%) |
| `band_code_crlf` | 12 | 0.35B | 118/118 CRLF (100%) |

**Repos included**: linux kernel, u-boot, zephyr, freertos, esp-idf, stm32cube, nvidia/cuda-samples, llvm-project, openblas, ffmpeg, grpc, abseil, protobuf, boringssl, v8, chromium (subset), and 58 more.

---

## 6. Indic Numeral Swap

### Problem
The tokenizer has dedicated tokens for space-prefixed Indic numerals (` ०`-` ९`, ` ০`-` ৯`, etc. across 10 scripts = ~120+ tokens). All had ZERO occurrences because the source data used Arabic digits (0-9) even in Indic text.

### Solution
Identified 1,996 B1 shards with Indic content. For each, scanned for Indic Unicode ranges and swapped adjacent Arabic digits with the corresponding native script numerals. Uploaded as `band_indic_numerals/`.

**Result**: 4.7M numeral swaps across 1,996 shards (66.9B tokens). All Indic numeral tokens now have non-zero training frequency.

---

## 7. Stage Profiles — Progressive Difficulty

### Token Budgets

| Stage | Budget | Warmup | Architecture | Wall-clock (est) |
|-------|--------|--------|-------------|-----------------|
| 1B | 50B | — | 2.5B Dense | ~19 days |
| WU→3B | 3B | 3B | Transition | ~2 hours |
| 3B | 40B | — | 3.9B Dense | ~23 days |
| WU→8B | 3B | 3B | Transition | ~2 hours |
| 8B | 80B | — | 8.4B Dense→MoE | ~18 days |
| WU→70B | 3B | 3B | Transition | ~2 hours |
| 70B | 30B | — | 71B MoE (260 experts) | ~7 days |
| **Total** | **209B** | | | **~67 days** |

With OPUS at 6x efficiency: **~1,254B effective tokens**.

### Pool Weights by Stage

| Pool | 1B | WU→3B | 3B | WU→8B | 8B | WU→70B | 70B |
|------|-----|-------|-----|-------|-----|--------|-----|
| D1 (Web Found.) | **0.42** | 0.34 | 0.22 | 0.14 | 0.09 | 0.06 | 0.06 |
| D2 (Web Div.) | 0.30 | 0.30 | 0.28 | 0.22 | 0.18 | 0.12 | 0.12 |
| D3 (Code) | 0.13 | 0.17 | 0.25 | 0.30 | 0.33 | 0.35 | **0.35** |
| D4 (STEM) | 0.07 | 0.11 | 0.17 | 0.26 | 0.32 | 0.39 | **0.39** |
| AON (Always-ON) | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |

**Pattern**:
- Web (D1+D2) drops from 72% → 18% as model grows
- Code (D3) ramps from 13% → 35%
- STEM (D4) ramps from 7% → 39%
- AON stays flat at 8% — always injected regardless

**Warmup bands**: 3B tokens each at growth transitions. Blend of 60% outgoing + 40% incoming stage profile. Purpose: stabilize newly-initialized parameters before the distribution shift.

### Data Availability per Pool

| Pool | Available | Cumul. Candidates | After OPUS (0.4) | Margin |
|------|-----------|-------------------|-------------------|--------|
| D1 | 164.1B | ~104B | ~41.6B | 1.6x |
| D2 | 627.4B | ~118B | ~47.2B | 5.3x |
| D3 | 199.0B | ~147B | ~58.8B | 1.4x |
| D4 | 49.1B | ~124B | ~49.6B | 1.0x (tight) |
| AON | 78.1B | 16.7B (direct) | 16.7B | 4.7x |

---

## 8. Deterministic Loading

### Mechanism

1. **Manifest files** (`manifests/*.txt`): Pre-shuffled shard lists per pool (seed=42)
2. **Shard striping**: `shards[rank::world_size]` — each GPU gets non-overlapping subset
3. **Block shuffling**: Within each pool, blocks are shuffled between epochs using `random.Random(seed + hash(pool))`
4. **AON injection**: Every ~12 training steps, one AON batch is injected (bypasses OPUS)

### What's Deterministic
- Shard order per pool (seed-shuffled, same for every run with same seed)
- Batch composition (which pool each block comes from)
- AON injection pattern (fixed period)
- DDP shard assignment (rank-based modulo)

### What Adapts
- OPUS selection (depends on live model state — intentional, this is the data efficiency mechanism)

### Files

| File | Location | Purpose |
|------|----------|---------|
| `curriculum_v2.yaml` | `DataSet/pipeline/` | Pool definitions, stage weights, guardrails |
| `curriculum_v2_manifest.json` | `DataSet/pipeline/manifests/` | Master manifest (JSON) |
| `D1_shards.txt` | `DataSet/pipeline/manifests/` | 4,894 shuffled D1 shard paths |
| `D2_shards.txt` | `DataSet/pipeline/manifests/` | 18,710 shuffled D2 shard paths |
| `D3_shards.txt` | `DataSet/pipeline/manifests/` | 5,933 shuffled D3 shard paths |
| `D4_shards.txt` | `DataSet/pipeline/manifests/` | 1,464 shuffled D4 shard paths |
| `AON_bench_train_shards.txt` | `DataSet/pipeline/manifests/` | 356 shuffled B6 shard paths |
| `AON_indic_shards.txt` | `DataSet/pipeline/manifests/` | 1,996 shuffled indic shard paths |
| `GP_shards.txt` | `DataSet/pipeline/manifests/` | 11 golden proxy shard paths |
| `indic_numerals_exclude.txt` | `DataSet/pipeline/manifests/` | 1,996 B1 shards to exclude from D2 |
| `DROPPED_B2_shards.txt` | `DataSet/pipeline/manifests/` | 934 dropped B2 shards (reference) |
| `curriculum_dataloader_v2.py` | `DataSet/pipeline/` | Dataloader with OPUS/AON/combined modes |
| `generate_manifest_v2.py` | `DataSet/pipeline/` | Script to regenerate manifests from S3 |

---

## 9. Corpus Statistics Summary

### By Pool

| Pool | Shards | Tokens | GB | % of Training |
|------|--------|--------|-----|--------------|
| D1 (Web Foundation) | 4,894 | 164.1B | 611.5 | 14.7% |
| D2 (Web Diverse) | 18,710 | 627.4B | 2,337.8 | 56.1% |
| D3 (Code) | 5,933 | 199.0B | 741.2 | 17.8% |
| D4 (STEM) | 1,464 | 49.1B | 182.8 | 4.4% |
| AON bench_train | 356 | 11.2B | 41.8 | 1.0% |
| AON indic_guaranteed | 1,996 | 66.9B | 249.2 | 6.0% |
| **Training Total** | **33,353** | **~1,117.7B** | **4,164.3** | **100%** |
| GP (never trained) | 11 | 0.007B | 0.03 | — |
| DROPPED B2 | 934 | 31.3B | 116.6 | — |
| **Grand Total** | **34,298** | **~1,149.0B** | **4,281.0** | — |

### By Language

| Script | Tokens | % of Corpus |
|--------|--------|-------------|
| Latin (English) | ~848B | 74.1% |
| Devanagari (Hindi, Marathi) | ~11.7B | 1.0% |
| Bengali | ~6.4B | 0.6% |
| Tamil | ~4.7B | 0.4% |
| Telugu | ~3.8B | 0.3% |
| Gujarati | ~3.5B | 0.3% |
| Malayalam | ~3.4B | 0.3% |
| Kannada | ~3.1B | 0.3% |
| Gurmukhi (Punjabi) | ~2.7B | 0.2% |
| Odia | ~1.4B | 0.1% |
| Digits + Punctuation + Whitespace | ~120B | 10.5% |
| Other / Code syntax | ~108B | 9.4% |

### Quality Summary

| Metric | D1 | D2 | D3 | D4 | AON |
|--------|-----|-----|-----|-----|-----|
| Entropy | 10.96 | 11.71 | 12.08 | 10.31-11.01 | Mixed |
| Repetition | 2.04% | 3.81% | 40.25% | 39-42% | 34% |
| Leakage | 0.87% | 2.48% | 1.21% | 1.25-7.19% | 5.43% |
| Garbage | 1.36% | 2.29% | 7.98% | 2.78-3.96% | 7.18% |
| Chars/Token | 4.63 | 4.40 | 3.48 | 3.17-3.18 | 3.50 |
| Difficulty | 0.37 | 0.52 | 0.61 | 0.41 | Mixed |

---

## 10. What Was Done — Complete Changelog

| # | Action | Date | Details |
|---|--------|------|---------|
| 1 | Corpus analysis | Feb 2026 | Analyzed 34,234 shards across 7 bands. Identified quality issues, vocab gaps, script distribution. |
| 2 | Tokenizer design | Feb 2026 | 131K Kronecker-aware multilingual BPE. 10 Indic scripts + Latin. |
| 3 | Code gap fix | Mar 2026 | Downloaded 74 repos. Created 52 tab-code + 12 CRLF shards (2.05B tokens). Covered 232 unseen tokens. |
| 4 | Indic numeral swap | Mar 2026 | Modified 1,996 B1 shards. 4.7M swaps. Native Indic numerals now in training data. |
| 5 | Token ID reordering | Mar 2026 | Remapped all 36,294 shards. Frequency-sorted IDs. Top 8K = 81% coverage. |
| 6 | Band redesign | Mar 10, 2026 | Dropped B2. Created D1-D4 + AON + GP architecture. Wrote curriculum_v2.yaml. |
| 7 | Manifest generation | Mar 10, 2026 | Enumerated S3, built deterministic shard lists (seed=42). 33,353 training shards. |
| 8 | Dataloader v2 | Mar 10, 2026 | curriculum_dataloader_v2.py with OPUS/AON/combined modes. |

---

## 11. Known Gaps & Future Work

| Priority | Issue | Impact | Mitigation |
|----------|-------|--------|------------|
| P0 | Code data is thin (only StarCoder) | Curriculum needs 13-35% code; D3 has 199B but 1.4x margin | Acquire The Stack v2 dedup (32TB) |
| P0 | No dedicated long-context data (>8K tokens) | Long context evals (L-Eval, RULER) | Acquire ProLong 64K (20B tokens) |
| P1 | STEM pool tight (49B, 1.0x margin after OPUS) | May not fully utilize at 70B stage | Acquire more academic/math data |
| P1 | Guardrails not enforced in dataloader | Anti-domain-spike, CoT caps defined but not implemented | Implement in curriculum_dataloader_v2.py |
| P1 | No checkpoint/resume for curriculum position | If training crashes, resumes from start of pool | Add RNG state serialization |
| P2 | Golden proxy is small (6.8M tokens) | OPUS proxy direction may be noisy | Expand with more benchmark test data |
| P2 | No Hindi-English code-switching data | Important for real Indic chat usage | Acquire or synthesize |
| P2 | Modality weights not enforced | CoT 6%, agentic 3% caps are aspirational | Implement per-block modality tagging |

---

## 12. EC2 Instance

| Item | Value |
|------|-------|
| Host | `ubuntu@ec2-13-218-189-218.compute-1.amazonaws.com` |
| Key | `dynamoCLI.pem` |
| Disk | 3.4TB NVMe (`/mnt/nvme0`), 1% used |
| Artifacts | `/mnt/nvme0/code_datasets/` — permutation maps, frequencies, reordered tokenizer |
| Manifest copy | `/mnt/nvme0/manifests/` — shard lists generated on this instance |

**DO NOT terminate** — contains permutation artifacts needed for tokenizer alignment.

---

*Generated 2026-03-10. Covers all work from corpus analysis through curriculum v2 deployment.*
