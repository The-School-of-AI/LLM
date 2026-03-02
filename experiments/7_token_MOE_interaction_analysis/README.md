# Team 7 — Token–MoE Interaction Analysis

> **Null-Expert Routing + Pluggable Mitigation**
>
> Turning null-expert routing into a measured, governed compute-allocation policy rather than an architectural hope.

---

## Overview

This project analyses the interaction between tokenization and Mixture-of-Experts (MoE) routing, with a specific focus on **null experts** — zero-compute experts that should absorb low-information tokens (whitespace, punctuation, boilerplate, template scaffolding) while leaving real experts free to process high-signal content.

The core goal is to ensure that:

- **Junk goes to null** — low-information tokens are routed to null experts, saving compute.
- **Signal goes to real experts** — code, reasoning, and structured content retain access to real expert compute.
- **Routing stays stable** — no collapse, polarization, or starvation pathologies emerge during training.

---

## Repository Structure

```
.
├── pyproject.toml
├── scripts/
│   ├── generate_token_distribution.py   # Build per-modality token frequency distributions from a parquet dataset
│   ├── jsongz_to_parquet.py             # Download Dolma-format .json.gz files and convert them to parquet
│   └── plot_token_dist.py               # Plot the top-K token distributions for each modality
└── src/
    └── moeint/
        ├── expert_analysis.py           # ModalityDistribution dataclass — stores, saves, and loads per-modality token distributions
        └── routing_health_metrics.py    # RouterHealthAnalyzer + RouterStats — full routing telemetry suite
```

---

## Core Concepts

### Null Experts

A null expert is a zero-compute expert: when a token is routed to one, no feed-forward computation is performed for that token. The number of null experts `M` is determined from the number of real experts `N` and the target data sparsity `ρ`:

```
M = N · (1 - ρ) / ρ
```

For example, with `N = 4` real experts and `ρ = 0.5`, there are `M = 4` null expert slots.

### Token Groups

Tokens are partitioned into two groups used throughout the telemetry:

| Group | Value | Description |
|-------|-------|-------------|
| `junk` | `0` | Low-information tokens: whitespace, separators, boilerplate |
| `content` | `1` | High-signal tokens: words, code, structured data |

The `token_id_group_mapping` argument to `RouterHealthAnalyzer` is a `dict[token_id, group_id]` that determines which group every token ID belongs to. Tokens not listed in the mapping default to `content`.

---

## `moeint` Library

### `RouterHealthAnalyzer` (`routing_health_metrics.py`)

The central telemetry engine. Instantiate once per training run, then call `analyze_logits` after every forward pass to obtain a `RouterStats` record per MoE layer.

```python
from moeint.routing_health_metrics import RouterHealthAnalyzer, TokenGroups

analyzer = RouterHealthAnalyzer(
    vocab_size=50257,
    token_id_group_mapping={tid: TokenGroups.junk for tid in junk_ids},
    num_experts=4,          # real experts only
    data_sparsity=0.5,      # determines number of null experts
    topk=2,
    starvation_threshold=0.01,
    polarization_threshold=(0.1, 0.9),
    device="cuda",
)

# During training, after model forward pass:
stats = analyzer.analyze_logits(input_ids, routing_logits)
# stats is a list[RouterStats], one entry per MoE layer
```

**`analyze_logits` inputs:**

| Argument | Type | Shape | Description |
|---|---|---|---|
| `input_ids` | `Tensor` | `(batch, seq_len)` | Token IDs for the current batch |
| `routing_logits` | `list[Tensor]` | each `(batch, seq_len, num_real + num_null)` | Raw router logits for each MoE layer |

---

### `RouterStats` — Telemetry Contract

Each call to `analyze_logits` returns one `RouterStats` `NamedTuple` per MoE layer. All scalar fields are Python `float` or `int`.

| Field | Type | Description |
|---|---|---|
| `entropy_mean` | `float` | Mean routing entropy across all tokens. Low → router is confident; high → router is uncertain. Null experts are collapsed into a single option before computing entropy. |
| `entropy_std` | `float` | Standard deviation of per-token entropy — spread of router confidence. |
| `entropy_per_group` | `list[float]` | Mean entropy indexed by `TokenGroups` value. `[entropy_junk, entropy_content]`. |
| `n_eff` | `float` | Effective number of experts: `exp(entropy_mean)`. Values near `num_real_experts + 1` indicate near-uniform routing; values near `1` indicate strong concentration. |
| `polarization` | `float` | Fraction of tokens at the extremes of compute intensity (either all-real or mostly-null). `0` = gradual distribution; `1` = fully polarized. |
| `tokens_to_real_rate` | `float` | Fraction of tokens with at least one top-K selection pointing to a real expert. Range `[0, 1]`. |
| `tokens_to_null_rate` | `float` | Fraction of tokens with at least one top-K selection pointing to a null expert. Range `[0, 1]`. |
| `null_junk_rate` | `float` | Of tokens routed to null, what fraction are junk? Measures null-expert purity. `NaN` if no tokens were routed to null. |
| `junk_to_null_rate` | `float` | Of junk tokens in the batch, what fraction went to null? Measures junk-routing success. `NaN` if no junk tokens were present. |
| `imbalance_ratio` | `float` | `max_tokens_per_real_expert / min_tokens_per_real_expert`. Lower is more balanced. Null experts excluded. |
| `gini` | `float` | Gini coefficient of real-expert token load. `0` = perfectly balanced; `1` = fully concentrated. |
| `cv` | `float` | Coefficient of variation of real-expert load. Standard deviation divided by mean. |
| `starvation_count` | `int` | Number of real experts receiving fewer than `starvation_threshold × average_tokens_per_expert` tokens. |
| `router_logit_scale` | `float` | Standard deviation of the raw router logits — a proxy for router confidence drift. |

**Healthy signal examples:**
- `junk_to_null_rate` trending upward → null experts are absorbing junk as intended.
- `null_junk_rate` near `1.0` → null experts are staying pure (not absorbing content tokens).
- `starvation_count == 0` and low `gini` → real experts are sharing load evenly.
- `polarization` staying low → no collapse toward all-null or all-real extremes.

---

### `ModalityDistribution` (`expert_analysis.py`)

A lightweight dataclass that stores a normalised token frequency distribution broken down by data modality (e.g. `code`, `web`, `math`).

```python
from moeint.expert_analysis import ModalityDistribution

dist = ModalityDistribution.load("_data/token_dist.pt")
# dist.distribution  — shape (num_modalities, vocab_size), rows sum to 1
# dist.index_to_modality  — {0: "code", 1: "web", ...}
# dist.source_files  — list of parquet files that produced this distribution

dist.save("_data/token_dist_v2.pt")
```

---

## Scripts

### `jsongz_to_parquet.py` — Download & Convert Dataset

Downloads Dolma-format `.json.gz` files from a list of URLs and converts them to Parquet for downstream processing. Uses Ray for parallel downloads.

```bash
python scripts/jsongz_to_parquet.py \
    --urls   urls.txt          \
    --output _data/parquet     \
    --workers 8
```

| Flag | Default | Description |
|---|---|---|
| `--urls` / `-u` | required | Path to a text file containing one `.json.gz` URL per line |
| `--output` / `-o` | required | Output directory for the resulting `.parquet` files |
| `--workers` / `-w` | `4` | Number of parallel Ray download workers |

Files that already exist at the output path are skipped automatically.

---

### `generate_token_distribution.py` — Build Token Distributions

Processes a directory of Parquet files in parallel (via Ray), tokenizes each record, tags it with a modality label, and accumulates per-modality token frequency arrays. The result is saved as a `ModalityDistribution` `.pt` file.

```bash
python scripts/generate_token_distribution.py \
    --input     _data/parquet      \
    --output    _data/token_dist.pt \
    --tokenizer gpt2               \
    --workers   8
```

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Directory containing `.parquet` files (searched recursively) |
| `--output` / `-o` | required | Output path for the `.pt` file |
| `--tokenizer` / `-t` | required | HuggingFace tokenizer name or local path |
| `--workers` / `-w` | `4` | Number of parallel Ray workers |

> **Note:** This script depends on `curriculum_tags` (from Team 2) for modality classification. Ensure it is installed or available on `PYTHONPATH` before running.

---

### `plot_token_dist.py` — Visualise Token Distributions

Reads a saved `ModalityDistribution` `.pt` file and plots the top-K most probable tokens for each modality as bar charts. The value of K is chosen dynamically based on the modality's entropy, so high-entropy (flat) distributions show more tokens than low-entropy (peaked) ones.

```bash
python scripts/plot_token_dist.py \
    _data/token_dist.pt \
    --k-min 20          \
    --k-max 500         \
    -o token_dist.png
```

| Argument | Default | Description |
|---|---|---|
| `dist_file` (positional) | required | Path to a `ModalityDistribution` `.pt` file |
| `--k-min` | `20` | Minimum number of top tokens to display |
| `--k-max` | `500` | Maximum number of top tokens to display |
| `-o` | required | Output image path (e.g. `.png`, `.pdf`) |

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) and is packaged as a standard Python project.

```bash
# Install dependencies and the moeint package in editable mode
uv sync

# Or with pip
pip install -e .
```

**Requirements:** Python ≥ 3.12

**Dependencies** (from `pyproject.toml`):

| Package | Purpose |
|---|---|
| `torch >= 2.10` | Tensor operations |
| `transformers >= 5.0` | Tokenizers, RoPE, RMSNorm |
| `datasets >= 4.5` | Streaming dataset loading |
| `ray >= 2.53` | Parallel data processing in pipeline scripts |
| `pyarrow >= 23.0` | Parquet I/O |
| `matplotlib >= 3.10` | Token distribution visualisation |

---

## Metrics Quick Reference

| Metric | Healthy Direction | Alert Condition |
|---|---|---|
| `junk_to_null_rate` | ↑ trending upward | Stays flat or falls — junk not reaching null |
| `null_junk_rate` | Close to `1.0` | Drops significantly — content leaking into null |
| `tokens_to_null_rate` | Moderate and stable | Approaches `1.0` (routing collapse) or `0.0` (null experts unused) |
| `polarization` | Low (`< 0.2`) | High values — bimodal compute: some tokens always skip, some never do |
| `n_eff` | Stable, well above `1` | Collapses toward `1` — all tokens routing to same expert |
| `starvation_count` | `0` | Any sustained positive count — expert starvation |
| `gini` | Low (`< 0.3`) | High values — severe load imbalance among real experts |
| `router_logit_scale` | Stable | Sudden spikes or monotonic growth — router confidence drifting |

---

## Team Context

This is the output of **Team 7** of the ERA4 capstone project.

**Depends on:**
- **Team 6** — Tokenizer artifact, token ID mapping, special token registry
- **Team 8** — MoE architecture configuration, null expert specification

**Advises / Gates:**
- **Team 8** — Advisory gate: Team 7 can flag routing collapse, null-expert cannibalization of high-signal groups, or unsafe sparsity before architecture decisions are finalized.

## Unaddressed Rquirements

### Expert - Token Affinity

The original plan included tracking which experts specialize in which modalities by comparing token distributions per modality (precomputed from the dataset) against the token distributions seen by each expert during training — since expert routing distributions shift over time, this comparison would reveal emergent expert–modality affinity.
The precomputation side was implemented: token distributions per modality are calculated from the dataset ahead of training. However, the runtime comparison against expert distributions was not integrated. Additionally, the precomputation itself has a performance problem that must be resolved before it is viable at scale — on a 16 GB subset of Dolma v1.6, calculating token distributions per modality takes approximately 30 minutes, making it impractical for the full dataset.

### Junk Token Labeling

Several null router metrics depend on knowing which tokens are classified as "junk" (e.g., padding, special tokens, or other tokens the router should learn to ignore). The metrics logic itself was implemented and verified, but no authoritative source or specification was identified that defines which tokens qualify as junk. Testing was carried out using a placeholder classification, so the metrics cannot be considered production-ready until a proper junk token definition is established and integrated.
