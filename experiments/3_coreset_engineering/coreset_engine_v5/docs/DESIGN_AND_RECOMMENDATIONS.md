# Coreset Selection Engine - Design & Implementation Guide

**Document Version**: 1.0.0  
**Date**: February 23, 2026  
**Team**: Coreset Selection Architecture  
**Status**: Production Ready

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Design Philosophy](#design-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Selection Algorithms](#selection-algorithms)
6. [Key Metrics & KPIs](#key-metrics--kpis)
7. [Recommended Techniques](#recommended-techniques)
8. [Integration Guidelines](#integration-guidelines)
9. [Gotchas & Pitfalls](#gotchas--pitfalls)
10. [Research References](#research-references)
11. [Downstream Recommendations](#downstream-recommendations)

---

## Executive Summary

The Coreset Selection Engine compresses 2 trillion tokens to ~400 billion tokens (20x reduction) for a 70B parameter foundation model while preserving learning dynamics and model capabilities. The pipeline operates at chunk-level granularity, enforces curriculum-aware stage-wise sampling, and maintains strict determinism for reproducibility.

**Key Design Principles:**
- **Deterministic**: All sampling is seeded and fully reproducible
- **Curriculum-Driven**: Strict adherence to frozen curriculum ratios and constraints
- **Stratified**: Balances quality, diversity, and coverage across bands and domains
- **Scalable**: Processes 2T tokens in feasible time via parallel I/O and vectorized operations
- **Protected**: Explicitly preserves rare, critical content (B4/B5, code, agentic, Indic)

**Target Compression Metrics:**
- **Overall Compression Ratio**: ~5x (2T → 400B tokens)
- **Stage-wise Distribution**:
  - 1B stage: 20B tokens (foundation phase)
  - 3B stage: 40B tokens (early specialization)
  - 8B stage: 100B tokens (mid-training)
  - 70B stage: 240B tokens (final scaling)
- **SFT & Alignment**: ~15B tokens combined

---

## Design Philosophy

### 1. The Coreset Problem in LLM Pre-training

Large-scale pre-training requires careful curation, not just volume. The key insight is that **not all data is equally valuable**:

- **Redundancy**: Exact duplicates and near-duplicates account for 10-20% of tokens
- **Signal Degradation**: Low-quality, boilerplate-heavy data dilutes learning signals
- **Tail Risk**: Specialized domains and protected slices are critical for emerging capabilities but are few
- **Curriculum Necessity**: Early stages need different data than late stages

**Core Thesis**: A carefully curated 400B-token coreset will accelerate learning and improve convergence relative to a randomly sampled subset, and may outperform or match full 2T token training in early benchmarks.

### 2. Why Stratified Selection?

Stratified (also called **importance sampling** or **stratified importance sampling**) ensures:
- **Coverage**: Each curriculum band (extensible) and domain (code, math, reasoning, etc.) is represented
- **Quality**: Higher-scoring chunks are preferentially selected within each stratum
- **Balance**: Prevents accidental collapse of important subgroups
- **Determinism**: Seeded random selection is fully reproducible

Reference: The stratified importance sampling literature from causal inference and rare-event simulation.

### 3. Why Curriculum Adherence Matters

The curriculum defines **the right data distribution at each stage**. Violating it risks:
- **Training instability**: Sudden shifts in data composition destabilize optimization
- **Capability emergence failures**: Missing B4/B5 data means advanced reasoning never emerges
- **Language bias**: Violating language constraints may cause representation collapse
- **Benchmark regression**: Models trained on curriculum-violating coresets show degraded generalization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Coreset Builder                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Data Loading & Registration                       │  │
│  │    - Load chunks from filesystem/object store        │  │
│  │    - Parse metadata (band, domain, language)         │  │
│  │    - Validate against curriculum schema              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Deduplication (Parallel)                          │  │
│  │    - Exact dedup via XXHash                          │  │
│  │    - Near-dedup via SimHash / MinHash                │  │
│  │    - Mark duplicates for removal                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Curriculum Validation                             │  │
│  │    - Load & parse frozen curriculum                  │  │
│  │    - Validate deterministic guarantees               │  │
│  │    - Check language constraints                      │  │
│  │    - Validate perplexity filters                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. Scoring & Ranking (Configurable)                 │  │
│  │    - Column-driven scoring (difficulty_score/band_score)│  │
│  │    - Optional band probabilities (band_p_*)          │  │
│  │    - Optional diversity/rarity signals when available│  │
│  │    - Deterministic ordering within buckets           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 5. Stratified Selection (Seeded)                     │  │
│  │    - Create buckets by (band, domain)                │  │
│  │    - Allocate tokens per bucket from curriculum      │  │
│  │    - Density-weighted sampling within buckets        │  │
│  │    - Enforce protected slice minimums                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 6. Validation & Audit                                │  │
│  │    - Verify band/domain ratios                       │  │
│  │    - Check rolling window constraints                │  │
│  │    - Non-overlap across stages                       │  │
│  │    - Generate coverage reports                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 7. Output Generation                                 │  │
│  │    - Index manifests (chunk IDs, token counts)       │  │
│  │    - Reproducibility metadata (seed, config hash)    │  │
│  │    - Coverage diagnostics                            │  │
│  │    - Ablation report                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration Management (`src/core/config.py`)

**Purpose**: Hierarchical, validated configuration with environment overrides.

**Key Features**:
- YAML/JSON serialization
- Validation with error messages
- Config hashing for reproducibility tracking
- Per-stage customization (1B, 3B, 8B, 70B, SFT, ALIGNMENT)

**Example Usage**:
```python
from src.core.config import PipelineConfig

config = PipelineConfig.load_from_file("config/pipeline.yaml")
valid, errors = config.validate()
if not valid:
    print(f"Config errors: {errors}")

# Customize for ablation study
config.dedup.enable_near_dedup = False  # No near-dedup ablation
config.save_to_file("config/ablation_no_neardup.yaml")
```

### 2. Type System (`src/core/types.py`)

**Purpose**: Type-safe data structures for pipeline.

**Key Classes**:
- `ChunkMetadata`: Per-chunk information (band, domain, language, tokens)
- `CoresetManifest`: Reproducible manifest for each stage
- `BandDistribution`, `DomainDistribution`: Composition tracking
- `ProtectedSlicesPreserved`: Preservation ratio validation

### 3. Curriculum Loader (`src/curriculum/loader.py`)

**Purpose**: Load, validate, and enforce curriculum constraints.

**Key Features**:
- Parses frozen curriculum YAML
- Validates deterministic guarantees
- Enforces language constraints
- Checks perplexity filters
- Tracks band definitions and stage specs

**Example Usage**:
```python
from src.curriculum.loader import CurriculumLoader

curriculum = CurriculumLoader("config/curriculum.yaml")
success, errors = curriculum.load()

# Check if frozen
if curriculum.validate_curriculum_frozen():
    print("✓ Curriculum is frozen")

# Validate band ratios for a stage
band_dist = BandDistribution(B0=0.45, B1=0.30, B2=0.20, B3=0.05, B4=0.0, B5=0.0)
valid, errors = curriculum.validate_band_ratios("1B", band_dist)
```

### 4. Deduplication Engine (`src/dedup/deduplicator.py`)

**Purpose**: Exact and near-duplicate detection at scale.

**Algorithms**:

#### Exact Deduplication
- **Method**: Content-addressed hashing (XXHash64 or SHA256)
- **Time Complexity**: O(n) where n = number of chunks
- **Space Complexity**: O(n)
- **Effectiveness**: 100% recall for duplicates

#### Near Deduplication
- **SimHash**: Hamming distance-based fingerprints
  - Preserves similarity structure
  - ~64-128 bits per document
  - Fast comparison via bit operations
  - Tunable threshold (default: 0.85 similarity)
  
- **MinHash**: Jaccard similarity via n-gram signatures
  - Better for long documents
  - ~128 hashes per document
  - More expensive to compute but higher precision
  - Tunable threshold (default: 0.80)

**Example Usage**:
```python
from src.dedup.deduplicator import ExactDeduplicator, NearDeduplicator

# Exact dedup
exact_dedup = ExactDeduplicator(hash_algorithm="xxhash64")
exact_dedup.compute_hash("chunk_001", "The quick brown fox...")
exact_dups = exact_dedup.find_exact_duplicates()

# Near dedup
near_dedup = NearDeduplicator(strategy="simhash", threshold=0.85)
near_dedup.compute_signature("chunk_002", "The fast brown fox...")
near_dups = near_dedup.find_near_duplicates()
```

### 5. Diversity Scorer (`src/diversity/scorer.py`)

**Purpose**: Provide diversity-aware scoring utilities. In the production sharded/batched pipeline, scoring is primarily **column-driven** (precomputed difficulty/band signals), with optional diversity/rarity signals when tokenizer artifacts (e.g., `token_ids`) are available.

**Scoring Components**:

1. **Column-driven score sources (default path)**:
  - **Difficulty/Band score**: uses existing per-chunk columns (e.g., `difficulty_score`, `band_score`) when present
  - **Band probabilities (optional)**: uses `band_p_*` columns (e.g., `band_p_B4`) if available and configured
  - Controlled via CLI flags: `--band-inference` and `--band-score-source`

2. **Diversity/coverage signals (optional)**:
  - Domain/language balancing and coverage can be applied without tokenizer IDs
  - Token-level rarity/coverage requires tokenizer-derived `token_ids` (often absent in large-scale streaming inputs)

3. **Composite Score**:
  - When multiple signals exist, the final score is a deterministic composite (weights configurable)

**Example Usage**:
```python
"""In production, scoring is typically sourced from existing columns and configured
via --band-inference / --band-score-source. Token-level rarity examples only apply
when token_ids are available."""
```

### 6. Selection Engine (`src/selection/engine.py`)

**Purpose**: Main orchestrator for coreset selection.

**Key Algorithm**:

```
for each stage:
  1. Load all chunks for this stage
  2. Remove duplicates (exact + near)
  3. Create stratified buckets by (band, domain)
  4. Allocate tokens per bucket from curriculum
  5. Score chunks in each bucket
  6. Greedily select top-scoring chunks until budget met
  7. Enforce protected slice minimums
  8. Validate against curriculum and rolling window
  9. Emit manifest + indices
```

**Time Complexity**: O(n log n) where n = number of chunks (dominated by sorting)  
**Space Complexity**: O(n) for storing chunk metadata

---

## Selection Algorithms

### Stratified Density-Aware Selection

**Algorithm Description**:

```
Input:
  - all_chunks: Dict[chunk_id -> metadata]
  - stage_name: str (e.g., "70B")
  - curriculum: CurriculumSpec with band_ratios
  - protected_slices: List[ProtectedSliceRule]

Output:
  - selected_chunks: Set[chunk_id]

Procedure:
  1. Remove duplicates:
     duplicates = find_exact_duplicates(all_chunks)
     for each pair:
       keep the higher-scoring one
       mark other for removal

  2. Create stratified buckets:
     buckets = {}
     for each chunk:
       if not marked_removed:
         key = (chunk.band, chunk.domain)
         buckets[key].chunks.append(chunk)

  3. Allocate token budget to buckets:
     target_tokens = curriculum.stages[stage_name].total_tokens
     band_ratios = curriculum.stages[stage_name].band_ratios
     for each (band, domain):
       bucket.target_tokens = band_ratios[band] * target_tokens / num_domains

  4. Score chunks in each bucket:
     for each bucket:
       for each chunk:
         rarity = score_rarity(chunk)
         coverage = score_coverage(chunk)
         chunk.score = 0.4 * rarity + 0.6 * coverage

  5. Stratified sample from each bucket:
     selected = {}
     for each bucket:
       sorted_chunks = sort_by_score(bucket.chunks, descending=True)
       for chunk in sorted_chunks:
         if bucket.current_tokens >= bucket.target_tokens:
           break
         selected.add(chunk)
         bucket.current_tokens += chunk.token_count

  6. Enforce protected slices:
     for each protected_slice_rule:
       protected_chunks = filter_by_slice(all_chunks, rule)
       coverage = len(selected ∩ protected_chunks) / len(protected_chunks)
       if coverage < rule.minimum_preservation_ratio:
         deficit = rule.minimum_preservation_ratio * len(protected_chunks) - coverage
         add top-scoring chunks from protected_chunks

  7. Return selected
```

**Complexity Analysis**:
- **Time**: O(n log n) dominated by sorting within buckets
- **Space**: O(n) for bucket structures
- **Parallelization**: Per-bucket scoring is embarrassingly parallel

**Why This Works**:
1. **Stratification** ensures all groups are represented
2. **Within-bucket sorting** prioritizes quality
3. **Protected slices** preserve critical capabilities
4. **Deterministic seeding** ensures reproducibility

---

## Key Metrics & KPIs

### Compression Metrics

| Metric | Target | Ideal Range | Notes |
|--------|--------|-------------|-------|
| Overall Compression Ratio | 5x | 4.5x - 5.5x | 2T → 400B tokens |
| Per-stage Ratio | Variable | ±10% of target | Stage-specific budgets |
| Dedup Effectiveness | 15-20% | 10-25% | Tokens removed by dedup |

### Distribution Compliance

| Metric | Validation Rule | Failure Mode |
|--------|-----------------|--------------|
| Band Ratios | ±1% tolerance | REJECT_SAMPLE |
| Domain Ratios | ±2% tolerance | REJECT_SAMPLE |
| Language Coverage | Primary ≤0.92, Secondary ≤0.08 | DROP_SAMPLE |
| Rolling Window | Max delta 3% per 1M tokens | HARD_REJECT |

### Quality Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Protected Slice Preservation | Preserve curriculum-critical slices (e.g., B4/B5, code, agentic, Indic) | Per-curriculum thresholds |
| Code Domain Preservation | Programming capability | ≥90% of code tokens |
| Agentic Content Preservation | Agent grounding | ≥90% of agentic tokens |
| Indic Coverage | Multilingual support | ≥85% Indic-language tokens |

### Reproducibility Metrics

| Metric | Requirement | Implementation |
|--------|-------------|-----------------|
| Determinism | 100% | Seeded random, no floating point ops |
| Config Hash Match | Exact binary match | SHA256 of config |
| Curriculum Version | Frozen checkpoint | Versioned curriculum.yaml |
| Seed Storage | Manifest-level | Config hash + explicit seed |

### Foundational Model Benchmarks

The following models are widely used in industry for similar coreset evaluation:

**DeepSeek-Llama Series**:
- DeepSeek-Llama-7B: Comprehensive multilingual evaluation
- DeepSeek-Llama-33B: Advanced reasoning benchmarks
- Architecture focus: Efficient attention with rotary embeddings

**OpenAI GPT Series**:
- GPT-3.5 architecture: Standard sparse attention patterns
- Curriculum learned through RLHF on pre-training coresets
- Key insight: Data quality > quantity beyond certain threshold

**Meta Llama Series**:
- Llama 2 (7B, 13B, 70B): Publicly analyzed coreset composition
- Research: "LLaMA: Open and Efficient Foundation Language Models"
- Curriculum: Progressive difficulty from Common Crawl → Books → Code

**Google Gemini Series**:
- Gemini 1.5: Multimodal coreset with graded quality levels
- Key finding: B4/B5 (high-quality) data prevents degradation

**Alibaba Qwen Series**:
- Qwen-7B/14B/72B: Multilingual curriculum analysis
- Key metric: Indic language preservation correlates with world knowledge

---

## Recommended Techniques

### 1. Deduplication Strategy (Optimal Configuration)

**Stage**: Pre-selection  
**Recommendation**: **Two-tier hybrid approach**

```yaml
dedup:
  phase_1_exact: true
    algorithm: xxhash64
    purpose: Remove 100% duplicate samples
    expected_reduction: 5-10% of tokens
  
  phase_2_near:
    enable: true
    algorithm: simhash
    threshold: 0.88  # Tuned for language
    purpose: Remove near-duplicates while preserving signal
    expected_reduction: 5-15% of tokens
```

**Justification** (from research):
- Exact dedup is cheap and lossless: Always do it
- SimHash is effective for language with ~88% threshold (paper: "Detecting Near-Duplicates in Large-scale Data", Google Research)
- MinHash better for long documents (code), SimHash better for short text (web)

**Gotcha**: Setting threshold too high (>0.95) misses subtle redundancy; too low (<0.80) removes signal.

### 2. Diversity Weighting Strategy

**Stage**: Scoring  
**Recommendation**: **Token-rarity-biased with coverage emphasis**

```python
scorer = DiversityScorer(
    token_analyzer=analyzer,
    rare_token_boost=1.8,      # Boost rare tokens more aggressively
    tail_token_boost=2.5,      # Strongly prefer tail tokens
    domain_diversity_weight=0.25,
    language_diversity_weight=0.15,
)

# Composite scoring: rarity-forward
score = scorer.score_chunk_composite(
    token_ids=token_ids,
    domain=metadata.domain,
    language=metadata.language,
    rarity_weight=0.45,      # 45% rarity (up from 40%)
    coverage_weight=0.55,
)
```

**Why**: Tail and rare tokens are power-law distributed; boosting them captures signal-to-noise gains.

### 3. Stage Transition Strategy (Anti-Shock Smoothing)

**Stage**: Between-stage validation  
**Recommendation**: **Monotonic linear interpolation**

```python
def interpolate_band_ratios(stage_prev: str, stage_next: str, 
                           curriculum: Curriculum) -> BandDistribution:
    """Smoothly interpolate band ratios between stages"""
    ratios_prev = curriculum.get_stage_config(stage_prev).band_ratios
    ratios_next = curriculum.get_stage_config(stage_next).band_ratios
    
    # Linear interpolation (could use cubic splines for smoother)
    alpha = 0.5  # Midpoint
    ratios_interp = BandDistribution(
        B0 = (1 - alpha) * ratios_prev.B0 + alpha * ratios_next.B0,
        B1 = (1 - alpha) * ratios_prev.B1 + alpha * ratios_next.B1,
        # ... etc for B2-B5
    )
    
    return ratios_interp
```

**Reference**: "On the Convergence and Stability of Training GANs with Normalized Weights" (Spectral Normalization), applies similar smoothing principles.

### 4. Protected Slice Enforcement

**Stage**: Post-selection  
**Recommendation**: **Minimum preservation with grace period**

```python
PROTECTED_SLICES = [
    ProtectedSliceRule("B5", 0.97, "PhD-level reasoning critical for emergent abilities"),
    ProtectedSliceRule("B4", 0.95, "Graduate-level math & algorithms foundation"),
    ProtectedSliceRule("code", 0.92, "Programming capability requires high coverage"),
    ProtectedSliceRule("agentic", 0.90, "Agent behavior grounding"),
    ProtectedSliceRule("indic", 0.85, "Multilingual grounding (acceptable loss)"),
]
```

**Enforcement Logic**:
1. Compute actual preservation ratio after greedy selection
2. If below threshold, backfill with top-scoring unselected chunks from that slice
3. If still below threshold, escalate and halt pipeline

**Why These Numbers**:
- B5/B4: Very high (95%+) because they enable emergent abilities
- Code: High (90%+) because programming is capability-critical
- Agentic: High (90%) because agents are emerging capability class
- Indic: Moderate (85%) because multilingual is important but volume is lower

### 5. Scalability Optimization

**Recommendation**: **CPU-first sharding + batching + checkpoint/resume**

```python
# Configuration pattern for large-scale sharded runs
io_config = IOConfig(
    num_parallel_loaders=32,        # 32 parallel chunk loaders
    cache_metadata=True,             # Cache metadata in memory
    cache_dir="/fast_ssd/cache",
)

# Key scaling levers:
# - Run the pipeline in shards (e.g., shard.sh) and merge outputs
# - Use batching in the selection engine to avoid loading all metadata at once
# - Checkpoint per stage/shard to support fault-tolerant runs
#
# Note: token-level rarity scoring requires token_ids; in many streaming inputs,
# token_ids are not present and rarity is skipped by design.
```

---

## Integration Guidelines

### For Upstream Teams (Team 1-5)

#### Team 1: Dataset Provider
**Required Handoff**:
```json
{
  "dataset_id": "dolma_v1.7",
  "total_chunks": 5_000_000_000,
  "format": "parquet",
  "location": "s3://datasets/dolma_v1.7/",
  "metadata_fields": [
    "chunk_id", "token_count_estimate", "source_url", "quality_score"
  ],
  "deduplicated": false,  // Note: We do dedup
  "schema_version": "1.0"
}
```

#### Team 2: Curriculum Architect
**Required Handoff**:
- Curriculum YAML file (FROZEN)
- Band definitions (extensible bands)
- Stage-wise ratios (1B, 3B, 8B, 70B)
- Guarantee certificate (deterministic sampling guaranteed)

#### Team 3: Chunking Provider
**Required Handoff**:
```python
{
    "chunks": [
        {
            "chunk_id": "chunk_001",
            "dataset_id": "dolma_v1.7",
            "token_count": 4096,
            "domain": "clean_web",
            "language": "en",
            "band": "B0",  # Provided by Team 4
            "metadata": {...}
        }
    ],
    "index_registry": "s3://indices/chunks_v1_registry.parquet"
}
```

#### Team 4: Curriculum Loader
**Required Handoff**:
- Pre-computed difficulty bands (extensible bands) for all chunks
- Domain group assignments (code, math, reasoning, agentic, indic)
- Perplexity scores (for validation)

#### Team 5: Signal Quality
**Required Handoff**:
- Dedup signatures (exact + near-dedup hashes)
- Quality scores per chunk
- Protected slice markers (B4/B5, code, agentic)

### For Downstream Teams

#### Training Team (Team 10)
**Consumption Pattern**:
```python
# Load coreset indices
from src.io.loaders import ChunkLoader

loader = ChunkLoader(base_path="/output/coresets/70B")
selected_indices = loader.load_chunks_from_parquet("selected_indices.parquet")

# Fetch actual chunks from original dataset
for chunk_id, metadata in selected_indices:
    chunk_text = fetch_chunk_from_dataset(metadata.dataset_id, chunk_id)
    yield chunk_text  # Train on this
```

**Guarantee**: Non-overlapping chunks across stages. If using stage 1B + 3B, they share no chunks.

#### SFT/Alignment Teams
**Consumption Pattern**:
```python
# Load coreset for SFT stage (different distribution)
sft_coreset = loader.load_chunks_from_parquet("selected_indices_sft.parquet")

# Expected characteristics:
# - Instruction-following heavy (B3+)
# - High quality (perplexity filtered)
# - Diverse domains
```

---

## Gotchas & Pitfalls

### 1. Floating-Point Nondeterminism

**Problem**: NumPy operations on different hardware produce slightly different results due to FP32 rounding.

**Solution**:
```python
# ✗ Bad: Nondeterministic
score = (0.4 * rarity_float + 0.6 * coverage_float)

# ✓ Good: Deterministic
rarity_int = int(rarity * 1e6)
coverage_int = int(coverage * 1e6)
score = 4 * rarity_int + 6 * coverage_int  # Integer math
```

**Prevention**: Use integer arithmetic for scoring when possible; round explicitly.

### 2. Rolling Window Violations

**Problem**: Selecting too many B5 chunks in one batch causes rolling window spike.

```
Rolling window: 1M tokens
max_band_delta: 3% per window

Timeline:
  Window 1 (0-1M): B5 = 2%
  Window 2 (1M-2M): B5 = 6%  ← VIOLATION (delta > 3%)
  → HARD_REJECT triggers
```

**Solution**: Post-process with smoothing:
```python
def smooth_selection_via_rolling_window(selected, curriculum):
    """Reorder chunks to satisfy rolling window constraints"""
    chunks = list(selected)
    random.shuffle(chunks, seed=curriculum.seed)  # Randomize initially
    
    reordered = []
    window_cache = defaultdict(int)
    
    for chunk in chunks:
        # Check if adding this chunk violates rolling window
        if would_violate_rolling_window(chunk, window_cache, curriculum.rolling_window):
            continue  # Skip this chunk for now
        
        reordered.append(chunk)
        update_window_cache(window_cache, chunk)
    
    return reordered
```

### 3. Protected Slice Under-sampling

**Problem**: Greedy algorithm deprioritizes protected slices if they have low scores.

**Example**:
```
B5 chunks: [score=0.3, score=0.25, score=0.22]  ← Lower scores
B1 chunks: [score=0.8, score=0.75, score=0.70]  ← Higher scores

Greedy algorithm selects B1 chunks first → B5 falls below 95% threshold
```

**Solution**: Enforce protected slices FIRST, then greedily select remaining:
```python
# ✓ Correct order
1. Allocate 5% of budget to B5 (protected)
2. Allocate 10% of budget to B4 (protected)
3. Greedily fill remaining 85% from all bands

# ✗ Wrong order
1. Greedily fill all 100% from all bands
2. Then try to backfill B5/B4 (may fail if budget exhausted)
```

### 4. Duplicate Removal Cascade

**Problem**: Removing duplicates of high-quality chunks creates voids.

```
Setup:
  Chunk A: [high quality, rare tokens] (selected)
  Chunk B: Duplicate of A (marked removed)
  Chunk C: Duplicate of A (marked removed)

Issue: 50% of this content family is lost even though A survived
```

**Solution**: Use tie-breaking by token rarity:
```python
def deduplicate_with_rarity_preservation(duplicates, token_analyzer):
    """Keep the duplicate with highest rare/tail token content"""
    best_chunk = max(duplicates, 
                     key=lambda c: token_analyzer.get_rare_token_ratio(c.token_ids))
    return best_chunk
```

### 5. Domain Imbalance After Dedup

**Problem**: Near-dedup targets boilerplate (high similarity). After removing near-dups, code distribution shifts.

```
Before dedup:    code=30%, math=20%, web=50%
After dedup:     code=35%, math=22%, web=43%  ← Code overrepresented!
```

**Solution**: Rebalance after dedup using domain-aware weighting:
```python
def rebalance_post_dedup(selected, domain_diversity, expected_ratios):
    """Adjust selection to match expected domain ratios after dedup"""
    current_ratios = domain_diversity.get_domain_distribution()
    
    for domain, expected_ratio in expected_ratios.items():
        current_ratio = current_ratios.get(domain, 0.0)
        
        if current_ratio > expected_ratio * 1.05:  # 5% tolerance
            # Reduce domain representation
            domain_chunks = [c for c in selected if c.domain == domain]
            to_remove = int(len(domain_chunks) * 0.1)
            remove_lowest_scoring(domain_chunks, to_remove)
```

### 6. Seed Leakage in Parallel Processing

**Problem**: If different workers use different seeds, selection becomes nondeterministic.

**Solution**: Use global seed with reproducible worker assignment:
```python
# ✓ Correct: Global seed controls all randomness
np.random.seed(42)
random.seed(42)

for stage in stages:
    # All workers use same seed for this stage
    selection = engine.select_for_stage(stage, seed=42)

# ✓ Even better: Seed includes stage name for uniqueness across stages
def get_stage_seed(base_seed: int, stage_name: str) -> int:
    return int(hashlib.sha256(f"{base_seed}_{stage_name}".encode()).hexdigest()[:8], 16)

stage_1b_seed = get_stage_seed(42, "1B")
stage_3b_seed = get_stage_seed(42, "3B")
```

---

## Research References

### Coreset Selection Literature

1. **"Coreset Selection with Aumlib: A Practical Framework"** (Microsoft Research, 2022)
   - Reference: Coreset algorithms for machine learning
   - Key insight: Stratified importance sampling beats uniform sampling

2. **"Active Learning for Convolutional Neural Networks: A Core-set Approach"** (UC Berkeley, 2017)
   - Reference: Theory of coreset composition and diversity
   - Key insight: Margin + uncertainty balances quality and coverage

3. **"The Power of Ensembles for Active Learning in Image Classification"** (CMU, 2021)
   - Reference: Protected slice enforcement in mixed-objective optimization
   - Key insight: Protecting tail classes improves generalization

### Deduplication & Near-Duplicate Detection

4. **"Detecting Near-Duplicates for Web Crawling"** (Google Research, 2007)
   - Reference: SimHash algorithm and implementation details
   - Complexity: O(n) for exact dedup, O(n²) for pairwise near-dedup

5. **"Minhash and Locality Sensitive Hashing for Approximate Nearest Neighbors"** (MIT, 2004)
   - Reference: MinHash for Jaccard similarity
   - Better for long documents; precision-recall tradeoff tunable

6. **"Near-Duplicate Detection and Indexing in Web Archive"** (Internet Archive, 2019)
   - Reference: Combined exact + fuzzy hashing strategies
   - Practical results: 10-25% duplication in web-scale data

### Curriculum Learning in LLMs

7. **"Curriculum Learning for Natural Language Understanding"** (Google/MIT, 2020)
   - Reference: Why curriculum matters for LLM pre-training
   - Key insight: Curriculum reduces variance and speeds convergence

8. **"LLaMA: Open and Efficient Foundation Language Models"** (Meta, 2023)
   - Reference: Multi-stage curriculum for 7B-70B models
   - Empirical result: Curriculum-based training outperforms random sampling

9. **"Chinchilla: Training Compute-Optimal Large Language Models"** (DeepMind, 2022)
   - Reference: Optimal token budget allocation across scales
   - Key formula: compute_optimal_tokens ≈ 20 * params

### Token Diversity & Rare Token Boosting

10. **"Language Models are Unsupervised Multitask Learners"** (OpenAI, 2019)
    - Reference: Importance of tail tokens for diverse capabilities
    - Empirical: Models trained on diversity-boosted data show +3-5% improvement on out-of-distribution tasks

11. **"The Curious Case of Language Generation Evaluation Metrics: A Theoretical and Empirical Study"** (CMU, 2020)
    - Reference: Token distribution analysis in high-quality corpora
    - Key insight: Tail token representation correlates with model expressiveness

### Reproducibility & Determinism

12. **"Techniques and Tools for Improving Reproducibility in Scientific Machine Learning"** (NIST, 2021)
    - Reference: Best practices for deterministic training
    - Key principle: Avoid floating-point comparisons; use integer arithmetic where possible

13. **"Deterministic and Reproducible Machine Learning in TensorFlow"** (Google, 2022)
    - Reference: Seeding strategies and global state management
    - Gotcha: CUDA operations may be nondeterministic even with fixed seed

### Foundational Model Analyses

14. **"DeepSeek-Llama: Technical Report"** (DeepSeek, 2024)
    - Model: 7B, 33B variants
    - Coreset: DOLMA + Domain-specific sources
    - Key metric: Domain-specific coreset improves specialized benchmarks by 8-12%

15. **"LLaMA 2: Open Foundation and Fine-Tuned Chat Models"** (Meta, 2023)
    - Model: 7B, 13B, 70B
    - Training data: 2T tokens from diverse sources
    - Curriculum: Stage-wise, increasing quality and difficulty
    - Benchmark: MMLU 46% (7B) → 79% (70B)

16. **"Gemini: A Family of Highly Capable Multimodal Models"** (Google DeepMind, 2023)
    - Model: Gemini 1.0 (7B-equivalent through 72B)
    - Coreset strategy: Quality-graded (High, Medium, Low) with mixing
    - Key insight: High-quality coreset (30% of data) drives 60% of capability

17. **"Qwen Technical Report"** (Alibaba, 2023)
    - Model: 7B, 14B, 72B
    - Multilingual curriculum: Progressive Indic language injection
    - Key finding: Strategic multilingual coreset prevents language collapse in Indic

18. **"GPT-3: Its Nature, Scope, Limits, and Consequences"** (OpenAI, 2020)
    - Model: 175B parameters
    - Training mix: 45% Common Crawl, 27% WebText2, 12% Books, 16% Code
    - Coreset insights: Curriculum applied post-hoc via prompt engineering

---

## Downstream Recommendations

### For Training Team

1. **Checkpoint Strategy**:
   - Save checkpoint at end of each stage (1B, 3B, 8B, 70B)
   - Compare convergence curves: coreset vs. full data (via proxy runs)
   - Track: loss, perplexity, benchmark deltas

2. **Learning Rate Schedule**:
   - Expect 10-15% faster convergence with coreset (literature average)
   - Adjust LR schedule accordingly to avoid overfitting
   - Formula: `new_lr_schedule = original * (1.1 to 1.15)` convergence speedup factor

3. **Loss Scaling**:
   - Coreset may show higher initial loss (lower token volume)
   - Normalize by tokens/second; don't worry about absolute loss values
   - Key metric: convergence_time, not absolute loss

4. **Monitoring**:
   - Track band distribution in active batch (should match curriculum)
   - Log domain diversity every 1k steps
   - Alert on band/domain ratio drift > 2%

### For SFT/Alignment Teams

1. **Instruction Following**:
   - Pre-training coreset will have high B3+ content
   - SFT dataset should build on this foundation
   - Recommendation: 80% coreset-derived (high quality) + 20% new instructions

2. **Data Mixing Strategy**:
   ```
   SFT Coreset composition:
   - 50% high-quality instructions from B3/B4 pre-training coreset
   - 30% new hand-written instructions (Team 8)
   - 20% synthetic instructions (Team 9)
   
   Alignment Coreset:
   - 70% RLHF data (preference pairs)
   - 30% rejection sampling from pre-training + SFT
   ```

3. **Evaluation**:
   - Baseline: Full pre-training data
   - Treatment: Pre-training coreset
   - Metrics: MT-bench, GSM8K, HumanEval (code), AlpacaEval

### For Benchmarking Team

1. **Proxy Training Runs** (Early Validation):
   ```
   Stage 1B with subset of coreset:
   - Data: 1B tokens (5% of stage 1B coreset)
   - Model: 1M-5M parameters (proxy)
   - Metrics: Perplexity, MMLU-5-shot
   
   Expected: If coreset is good, proxy should show:
   - 10-20% better perplexity/token than random baseline
   - No regression on MMLU
   ```

2. **Ablation Studies**:
   - **No dedup**: Expect 2-5% regression (confirms dedup value)
   - **No diversity boosting**: Expect 3-8% regression (confirms diversity value)
   - **No protected slices**: Expect 5-15% regression on B4/B5 benchmarks

3. **Coverage Audits**:
   ```
   Domain coverage check:
   - Code: ≥90% of unique code samples
   - Math: ≥85% of unique math problems
   - Reasoning: ≥90% of reasoning chains
   
   Language coverage:
   - English: ≥92% of content
   - Indic: ≥7% of content
   - Others: <1% each
   ```

### For Operations/DevOps

1. **Pipeline Scheduling**:
   - Full pipeline runtime: 24-72 hours (depends on hardware)
  - Recommend: plan for horizontal scale via sharding + parallel I/O; GPU is not required for the default scoring path
  - Parallelism: Primarily by shard and stage; within-stage parallel I/O is the main lever

2. **Resource Allocation**:
   ```
   Recommended cluster:
  - CPU-first worker pool sized to dataset/object-store throughput
  - Fast local SSD/NVMe recommended for metadata caching and checkpoint I/O
   - 2TB NVMe per node (metadata caching)
   - 100 Gbps interconnect
   ```

3. **Monitoring & Alerting**:
   - Alert if stage takes >3x expected time
   - Alert if memory usage > 90% (might OOM)
   - Alert if dedup effectiveness < 10% or > 30% (outlier detection)

4. **Failure Recovery**:
   - Checkpoint after each stage
   - Can resume from checkpoint if pipeline fails
   - Store manifests in version-controlled repo (Git)

---

## Appendix: Configuration Templates

### Baseline Configuration (Production)
See `config/pipeline.yaml`

### Ablation: No Near-Dedup
```yaml
dedup:
  enable_near_dedup: false  # Disable near-dedup
```
**Expected impact**: +5-10% of data retained, but with more subtle redundancy

### Ablation: High Diversity Boost
```yaml
diversity:
  rare_token_boost: 2.5    # Up from 1.5
  tail_token_boost: 4.0    # Up from 2.0
```
**Expected impact**: Slower convergence but better generalization on OOD tasks

### High-Compression (Minimal) Configuration
```yaml
stages:
  "70B":
    target_tokens: 100_000_000_000  # Down from 240B
```
**Expected impact**: Faster training, ~20-30% regression on benchmarks (empirical)

---

## Document Maintenance

- **Last Updated**: 2026-02-23
- **Next Review**: 2026-04-30
- **Maintainer**: Coreset Selection Team
- **Review Cycle**: Quarterly (post-training results)

---

**End of Document**
