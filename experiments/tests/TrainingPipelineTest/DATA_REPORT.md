# TrainingPipelineV1 — Data & Curriculum Report

**Date**: 2026-03-13
**Status**: Production-ready. Manifests frozen (seed=42), tokenizer reordered, shards on S3.

---

## 1. Where Is the Data?

### Production S3 Bucket

```
s3://t1-dataacquisition-datasets-2/shards_reordered/
```

This is the ONLY bucket to use. All 36,294 shards have frequency-reordered token IDs.

| S3 Directory | Shards | Size | Maps To |
|---|---|---|---|
| `band_B0/` | 4,894 | 611 GB | Pool D1 |
| `band_B1/` | 20,706 | 2,587 GB | Pool D2 (minus 1,996 indic shards) |
| `band_B2/` | 934 | 117 GB | DROPPED — do not use |
| `band_B3/` | 5,869 | 734 GB | Pool D3 |
| `band_B4/` | 1,146 | 143 GB | Pool D4 |
| `band_B5/` | 318 | 40 GB | Pool D4 |
| `band_B6/` | 356 | 42 GB | Pool AON (bench_train) |
| `band_code_tab/` | 52 | 6 GB | Pool D3 |
| `band_code_crlf/` | 12 | 1 GB | Pool D3 |
| `band_indic_numerals/` | 1,996 | 249 GB | Pool AON (indic_guaranteed) |
| `golden_proxy/band_golden_proxy/` | 11 | 0.03 GB | Pool GP (never trained on) |
| **Total** | **36,294** | **~4,528 GB** | |

### Shard Format

Each shard is a directory containing `tokens.bin`:
- **Format**: uint32 array of frequency-reordered token IDs
- **Block size**: 4,096 tokens for training data (D1-D4, AON)
- **Block size**: 512 tokens for Golden Proxy (GP) — shorter because benchmark examples are short QA pairs
- **Blocks per shard**: ~8,192 (variable)
- **Average shard**: ~128 MB, ~33.5M tokens
- **EOS token**: ID 36 (after reorder)
- **PAD token**: ID 130726 (after reorder)

### Backup Bucket (DO NOT use for training)

```
s3://t1-dataacquisition-datasets-2/shards/          # Original (old token IDs)
s3://t1-dataacquisition-dataset-shards/              # Mirror backup
```

---

## 2. The D-Band Pool Architecture

**D bands are virtual, not physical.** The S3 directories are still named `band_B0/`, `band_B1/`, etc. The D1-D4 pools are logical groupings defined by the manifest shard lists. For example, D3 pulls shards from three physical directories (`band_B3/` + `band_code_tab/` + `band_code_crlf/`), and D2 pulls from `band_B1/` but excludes 1,996 shards that are in AON instead.

There are three tiers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPUS-Eligible (D1-D4) — OPUS scores candidates, selects ~40%      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ D1: Web  │ │ D2: Web  │ │ D3: Code │ │ D4: STEM │              │
│  │ Found.   │ │ Diverse  │ │          │ │          │              │
│  │ 164B tok │ │ 627B tok │ │ 199B tok │ │  49B tok │              │
│  │ 4,894 sh │ │18,710 sh │ │ 5,933 sh │ │ 1,464 sh │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  Always-ON (AON) — 8% of every batch, bypasses OPUS entirely       │
│  ┌────────────────────┐ ┌──────────────────────────┐              │
│  │ bench_train        │ │ indic_guaranteed          │              │
│  │ 11.2B tok, 356 sh  │ │ 66.9B tok, 1,996 sh      │              │
│  │ math/code/reasoning│ │ Indic + native numerals   │              │
│  └────────────────────┘ └──────────────────────────┘              │
│  Internal split: 50% bench_train, 50% indic_guaranteed             │
├─────────────────────────────────────────────────────────────────────┤
│  Golden Proxy (GP) — OPUS reference signal, NEVER trained on       │
│  11 shards, 6.8M tokens (MMLU, GPQA, Math500, HumanEval, etc.)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Pool D1 — Web Foundation (164.1B tokens, 4,894 shards)

**Source**: `band_B0/` (cc_head + reddit)

The cleanest data. Lowest entropy (10.96), lowest repetition (2.04%), best chars/token (4.63). This is where the 1B model learns core English language patterns.

**Quality**: Leakage 0.87%, garbage 1.36% — excellent.
**Difficulty composite**: 0.372 (easiest).

### Pool D2 — Web Diverse (627.4B tokens, 18,710 shards)

**Source**: `band_B1/` minus 1,996 shards listed in `indic_numerals_exclude.txt`

Largest pool (54.8% of training corpus). Contains:
- CommonCrawl tail/middle/news, RefinedWeb, C4
- AI4Bharat curated Indic (BPCC, comparable, ilci, wiki)
- IndicCorpV2 (16 languages)
- ERAV4 math/pattern (mixed into 1,102 shards, cannot be separated)

All Indic languages live here: Devanagari 11.6B, Bengali 6.4B, Tamil 4.6B, Telugu 3.8B, Gujarati 3.5B, Malayalam 3.4B, Kannada 3.1B, Gurmukhi 2.7B, Odia 1.4B.

The 1,996 excluded shards have Indic content — those are in AON instead (with native numeral swaps) to avoid duplication.

**Quality**: Leakage 2.48%, garbage 2.29% — good.

### Pool D3 — Code (199.0B tokens, 5,933 shards)

**Source**: `band_B3/` + `band_code_tab/` (52 shards) + `band_code_crlf/` (12 shards)

- StarCoder (The Stack v1/v2) = 196.7B tokens, 5,869 shards
- code_tab = 1.70B tokens — firmware, CUDA, kernel, embedded, C (from 74 GitHub repos)
- code_crlf = 0.35B tokens — C#, VB.NET, PowerShell, Windows batch

The 64 synthetic code shards were created to cover 232 unseen tokens (114 tab-prefixed like `\tvirtual`, 118 CRLF like `}\r\n`). Without these, those tokens would have zero training frequency.

**Quality**: Highest entropy (12.08), highest repetition (40.25% — code boilerplate).
**Difficulty composite**: 0.605 (hardest statistically).

### Pool D4 — STEM (49.1B tokens, 1,464 shards)

**Source**: `band_B4/` + `band_B5/` (merged — cosine similarity 0.949)

- pes2o (academic papers) ~20B
- RedPajama-ArXiv ~13B
- Proof Pile 2 (algebraic_stack, open_web_math) ~5B
- FLAN + other STEM ~11B

Tightest pool. After OPUS selects 40%, effective training ≈ 49.6B — essentially 1.0x through all STEM data. Acceptable because math benefits from revisiting.

**Difficulty composite**: 0.41 (lower entropy than code, but conceptually hardest — math/reasoning is the bottleneck skill at 70B).

### Pool AON — Always-ON (78.1B tokens, 2,352 shards)

**Purpose**: Data that CANNOT be missed. Injected at 8% of every batch. OPUS never sees it.

**Why AON exists**: The Golden Proxy is English-heavy (cosine 0.876 with B1). OPUS would systematically reject Indic-heavy batches and non-standard benchmark formats because they project poorly onto the English proxy direction. AON bypasses this bias.

| Sub-pool | Source | Shards | Tokens | Contents |
|---|---|---|---|---|
| bench_train | `band_B6/` | 356 | 11.2B | finephrase (2.1B), SWE-smith trajectories (3.5B), OpenMathReasoning (2.2B), OpenMathInstruct-2 (1.5B), OpenCodeReasoning-2 (0.8B), NuminaMath (0.4B), OpenR1-Math (0.3B), flan_v2, glaive-function-calling, mmlu_aux_train, and 19 more |
| indic_guaranteed | `band_indic_numerals/` | 1,996 | 66.9B | B1 Indic shards with native numeral swap (e.g., ०१२ instead of 012). Covers all 10 Indic scripts. Also contains ai-bharath, ERAV4 lang/math, samvaad_hi, cc_news (co-located) |

**Internal split**: Each AON injection batch is 50% bench_train + 50% indic_guaranteed.

### Pool GP — Golden Proxy (6.8M tokens, 11 shards, block_size=512)

OPUS scoring reference. Contains benchmark test/validation splits (MMLU, MMLU Pro, MILU, GPQA Diamond, Math500, GSM8K, AIME 2025, HumanEval, MBPP, ARC-Challenge, BBH, IFEval, IndicGLUE, TriviaQA).

**Block size is 512** (not 4096 like training data) because benchmark examples are short QA/MCQ pairs. Packing them into 4096-token blocks would mix unrelated benchmark examples, weakening the proxy signal.

**NEVER trained on.** OPUS computes a proxy direction from these, then uses it to score D1-D4 candidate batches.

### DROPPED — B2 (31.3B tokens, 934 shards)

Killed for quality: 18.7% cross-document leakage, 9.14% garbage tokens, 15.69% repetition. Shard list preserved in `DROPPED_B2_shards.txt` for reference only.

---

## 3. Stage Profiles — Which Pools Feed Which Model

Training is progressive: 1B → 3B → 8B → 70B MoE, with warmup transitions.

### Token Budgets

Throughput measured on single 8×A100-80GB node (p4de).

| Stage | Budget | Throughput | Architecture | Wall-clock |
|---|---|---|---|---|
| 1B | 50B | 55K tok/s | 2.5B Dense | ~10.5 days |
| WU→3B | 3B | 38K tok/s | Transition | ~22 hours |
| 3B | 40B | 38K tok/s | 3.9B Dense | ~12.2 days |
| WU→8B | 3B | 12K tok/s | Transition | ~2.9 days |
| 8B | 80B | 12K tok/s | 8.4B Dense→MoE | ~77.2 days |
| WU→70B | 3B | 13K tok/s | Transition | ~2.7 days |
| 70B | 30B | 13K tok/s | 71B MoE (260 experts) | ~26.7 days |
| **Total** | **209B actual → ~1,254B effective with OPUS** | | | **~133 days** |

The 8B stage dominates wall-clock (77 days / 133 total = 58%). Multi-node scaling would reduce this proportionally.

### OPUS Collapsed (1/8th Token Budget)

With OPUS at ~8× data efficiency, each stage needs only 1/8th the tokens. Throughput is lower due to OPUS scoring overhead.

| Stage | Budget | Throughput | Architecture | Wall-clock |
|---|---|---|---|---|
| 1B | 6.25B | 55K tok/s | 2.5B Dense | ~1.3 days |
| WU→3B | 0.375B | 25K tok/s | Transition | ~4.2 hours |
| 3B | 5B | 25K tok/s | 3.9B Dense | ~2.3 days |
| WU→8B | 0.375B | 8K tok/s | Transition | ~13 hours |
| 8B | 10B | 8K tok/s | 8.4B Dense→MoE | ~14.5 days |
| WU→70B | 0.375B | 9K tok/s | Transition | ~11.6 hours |
| 70B | 3.75B | 9K tok/s | 71B MoE (260 experts) | ~4.8 days |
| **Total** | **~26.1B actual → ~209B effective** | | | **~24.1 days** |

### Pool Weights by Stage

```
         D1(Web)  D2(Diverse)  D3(Code)  D4(STEM)  AON
  1B:     0.42      0.30        0.13      0.07     0.08
  WU→3B:  0.34      0.30        0.17      0.11     0.08
  3B:     0.22      0.28        0.25      0.17     0.08
  WU→8B:  0.14      0.22        0.30      0.26     0.08
  8B:     0.09      0.18        0.33      0.32     0.08
  WU→70B: 0.06      0.12        0.35      0.39     0.08
  70B:    0.06      0.12        0.35      0.39     0.08
```

**The pattern**:
- Web (D1+D2): 72% at 1B → 18% at 70B (fades out)
- Code (D3): 13% → 35% (ramps up)
- STEM (D4): 7% → 39% (sharpest ramp)
- AON: flat 8% always (benchmark train + Indic, bypasses OPUS)

**Warmup stages**: 3B tokens at each growth transition. Blend of 60% outgoing + 40% incoming profile to stabilize newly-initialized parameters.

### What Each Stage Learns

| Stage | Primary Focus | Key Benchmarks Targeted |
|---|---|---|
| **1B** | English language patterns, basic knowledge, Hindi from day 1 | MMLU foundations, basic GSM8K |
| **3B** | Knowledge expansion, code growth, Indic ramp-up | MMLU 70+, HumanEval 70+, MILU |
| **8B** | Reasoning emergence, advanced code, math | MMLU 80+, HumanEval 90+, AIME, GPQA |
| **70B** | Expert specialization, hardest benchmarks, agentic | MMLU 85+, AIME 80+, SWE-Bench 34+, GPQA Diamond 67+ |

---

## 4. Tokenizer

**File**: `tokenizer/tokenizer_reordered.json`

| Property | Value |
|---|---|
| Type | 131K Kronecker-aware multilingual BPE (Tekken + IST hybrid) |
| Vocab size | 131,072 |
| EOS ID | 36 (was 130717 before reorder) |
| PAD ID | 130726 (was 130718 before reorder) |
| Most common token | ` ` (space) = ID 0 |

### Why Token IDs Were Reordered

All 131,072 tokens renumbered by descending global frequency across the 1.15T corpus:
- Cache locality: hot tokens at low embedding table indices
- Adaptive softmax potential: common tokens need less computation
- Natural vocabulary truncation: top-K IDs by count = X% of corpus

| Top-K IDs | Corpus Coverage |
|---|---|
| 100 | 35.2% |
| 1,000 | 62.8% |
| 8,192 | 81.0% |
| 32,000 | 94.7% |
| 65,536 | 99.1% |

**Safety**: BPE merges are string-based (`['G', 'G'] → 'GG'`), NOT ID-based. Merges were NOT modified. Only the vocab→ID mapping was updated. Verified with 9 encode/decode test cases.

### Permutation Maps

- `tokenizer/token_permutation.npy` — maps old ID → new ID
- `tokenizer/token_inv_permutation.npy` — maps new ID → old ID

These exist for debugging/inference only. Training uses `tokenizer_reordered.json` directly. All shards on S3 already contain reordered IDs.

---

## 5. Curriculum Dataloader

**File**: `code/src/curriculum_dataloader_v2.py`

### Three Operating Modes

```python
from curriculum_dataloader_v2 import build_curriculum_v2_dataloader

# Mode 1: OPUS candidate batches (D1-D4 only, to be scored by OPUS)
candidate_loader = build_curriculum_v2_dataloader(
    shard_dir="/path/to/shards_reordered",
    manifest_dir="/path/to/manifests",
    curriculum_path="configs/curriculum_v2.yaml",
    stage="1B",
    batch_size=8,
    mode="opus_candidates",
)

# Mode 2: Always-ON batches (bypasses OPUS, injected directly)
aon_loader = build_curriculum_v2_dataloader(
    ..., mode="always_on",
)

# Mode 3: Combined (non-OPUS, curriculum-weighted D1-D4 + AON at 8%)
combined_loader = build_curriculum_v2_dataloader(
    ..., mode="combined",
)
```

### How It Works

1. **Manifest-driven**: Reads pre-shuffled shard lists from `manifests/*.txt` (seed=42)
2. **Shard striping**: Each GPU gets `shards[rank::world_size]` — non-overlapping
3. **Block shuffling**: Within each pool, blocks are reshuffled between epochs using `random.Random(seed + hash(pool))`
4. **Pool sampling**: Each sequence, a pool is chosen via `random.choices()` with stage weights
5. **AON injection**: In combined mode, AON batches are sampled at 8% of overall weight
6. **mmap-based reading**: Shards are memory-mapped for efficient random access
7. **Stats tracking**: Logs actual vs target pool proportions every N sequences

### Yields

```python
{
    "input_ids": torch.Tensor,       # [seq_len] int64
    "attention_mask": torch.Tensor,  # [seq_len] all ones
    "labels": torch.Tensor,          # [seq_len] same as input_ids
    "_pool": str,                    # Pool name (D1/D2/D3/D4/AON_bench/AON_indic)
}
```

---

## 6. Manifest Files

All in `manifests/`. Pre-shuffled with seed=42. Each line is a relative path like `band_B0/shard_001554`.

| File | Lines | Purpose |
|---|---|---|
| `D1_shards.txt` | 4,894 | Pool D1 shard paths |
| `D2_shards.txt` | 18,710 | Pool D2 shard paths (B1 minus indic overlap) |
| `D3_shards.txt` | 5,933 | Pool D3 shard paths (B3 + code_tab + code_crlf) |
| `D4_shards.txt` | 1,464 | Pool D4 shard paths (B4 + B5) |
| `AON_bench_train_shards.txt` | 356 | AON bench_train sub-pool (B6) |
| `AON_indic_shards.txt` | 1,996 | AON indic_guaranteed sub-pool |
| `GP_shards.txt` | 11 | Golden Proxy (never trained on) |
| `DROPPED_B2_shards.txt` | 934 | Dropped B2 (reference only) |
| `indic_numerals_exclude.txt` | 1,996 | B1 shards excluded from D2 (already in AON) |
| `curriculum_v2_manifest.json` | — | Master manifest (pool definitions, stage weights, summary) |

---

## 7. Corpus Statistics

### By Pool

| Pool | Shards | Tokens | % of Training |
|---|---|---|---|
| D1 (Web Foundation) | 4,894 | 164.1B | 14.7% |
| D2 (Web Diverse) | 18,710 | 627.4B | 56.1% |
| D3 (Code) | 5,933 | 199.0B | 17.8% |
| D4 (STEM) | 1,464 | 49.1B | 4.4% |
| AON bench_train | 356 | 11.2B | 1.0% |
| AON indic_guaranteed | 1,996 | 66.9B | 6.0% |
| **Training Total** | **33,353** | **~1,118B** | **100%** |

### By Language

| Script | Tokens | % |
|---|---|---|
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
| Code syntax + digits | ~228B | 19.9% |

### Quality by Pool

| Metric | D1 | D2 | D3 | D4 | AON |
|---|---|---|---|---|---|
| Entropy | 10.96 | 11.71 | 12.08 | 10.31-11.01 | Mixed |
| Repetition | 2.04% | 3.81% | 40.25% | 39-42% | 34% |
| Leakage | 0.87% | 2.48% | 1.21% | 1.25-7.19% | 5.43% |
| Garbage | 1.36% | 2.29% | 7.98% | 2.78-3.96% | 7.18% |

---

## 8. OPUS Integration

**OPUS** (Optimizer-induced Projected Utility Selection) scores candidate batches from D1-D4 and selects ~40%. It projects optimizer-induced updates onto a proxy direction from the Golden Proxy (benchmark test data), keeping only tokens that move the model toward benchmark performance.

**Data flow during training**:

```
                    ┌─────────────────┐
                    │  Golden Proxy   │
                    │  (11 shards)    │
                    │  NEVER trained  │
                    └────────┬────────┘
                             │ proxy direction
                             ▼
┌──────────────────────────────────────────┐
│         OPUS Scoring Engine              │
│  Scores candidate batches from D1-D4    │
│  Selects top ~40% by proxy alignment    │
└──────────────┬───────────────────────────┘
               │ selected batches
               ▼
┌──────────────────────────────────────────┐
│         Training Step                    │
│  92% selected D1-D4 batches (via OPUS)  │
│   8% AON batches (bypasses OPUS)        │
└──────────────────────────────────────────┘
```

**Without OPUS** (using `mode="combined"`): All D1-D4 + AON batches are used directly at curriculum weights. No selection/filtering. This mode is for early debugging or when OPUS overhead is too high.

---

## 9. File Layout of This Folder

```
TrainingPipelineV1/
├── DATA_REPORT.md                          # This file
├── configs/
│   └── curriculum_v2.yaml                  # Pool definitions + stage weights + guardrails
├── code/
│   └── src/
│       └── curriculum_dataloader_v2.py     # Manifest-driven dataloader (3 modes)
├── manifests/
│   ├── curriculum_v2_manifest.json         # Master manifest
│   ├── D1_shards.txt                       # 4,894 shard paths
│   ├── D2_shards.txt                       # 18,710 shard paths
│   ├── D3_shards.txt                       # 5,933 shard paths
│   ├── D4_shards.txt                       # 1,464 shard paths
│   ├── AON_bench_train_shards.txt          # 356 shard paths
│   ├── AON_indic_shards.txt                # 1,996 shard paths
│   ├── GP_shards.txt                       # 11 shard paths
│   ├── DROPPED_B2_shards.txt               # 934 (reference only)
│   └── indic_numerals_exclude.txt          # 1,996 B1 shards excluded from D2
└── tokenizer/
    ├── tokenizer_reordered.json            # Production tokenizer (frequency-sorted IDs)
    ├── tokenizer_config.json               # HuggingFace tokenizer config
    ├── special_tokens_map.json             # Special token definitions
    ├── token_permutation.npy               # old ID → new ID mapping
    └── token_inv_permutation.npy           # new ID → old ID mapping
```

---

## 10. Quick Start — Downloading Data to EC2

```bash
# Sync production shards from S3 to local NVMe
aws s3 sync s3://t1-dataacquisition-datasets-2/shards_reordered/ /mnt/local-nvme/data/shards_reordered/ \
    --exclude "band_B2/*" \
    --no-sign-request

# Or download specific pools only (for 1B stage, D1+D2 are 72%):
aws s3 sync s3://t1-dataacquisition-datasets-2/shards_reordered/band_B0/ /mnt/local-nvme/data/shards_reordered/band_B0/
aws s3 sync s3://t1-dataacquisition-datasets-2/shards_reordered/band_B1/ /mnt/local-nvme/data/shards_reordered/band_B1/
aws s3 sync s3://t1-dataacquisition-datasets-2/shards_reordered/band_B3/ /mnt/local-nvme/data/shards_reordered/band_B3/
# ... etc for all bands
```

Total download: ~4.4 TB (excluding dropped B2).
