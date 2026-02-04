# Architecture

## Overview

The coreset engineering pipeline is designed to reduce a 2T token corpus to ~400B tokens across 4 training stages while preserving curriculum integrity and learning dynamics.

## Components

### 1. Deduplication (`src/deduplication/`)

- **Exact Deduplication**: Hash-based exact match removal
- **Near Deduplication**: MinHash LSH for similarity-based removal

### 2. Selection (`src/selection/`)

- **Stratified Sampling**: Curriculum-aware proportional sampling
- **Protected Slices**: Ensures critical data (B4/B5, agentic, etc.) is preserved

### 3. Validation (`src/validation/`)

- **Curriculum Validator**: Checks ratio adherence and smooth transitions
- **Coverage Metrics**: Ensures representational completeness

### 4. Pipeline (`src/coreset_builder/`)

- **Main Pipeline**: Orchestrates dedup → selection → validation
- **Manifest Generation**: Creates reproducible audit trails

## Data Flow

```
Raw Corpus (2T tokens)
    ↓
Chunk-level processing
    ↓
Exact Deduplication
    ↓
Near Deduplication
    ↓
Stratified Sampling (curriculum-aware)
    ↓
Validation (ratios, smoothness, protected slices)
    ↓
Stage Coresets (20B / 40B / 100B / 240B)
```

## Stage Specifications

| Stage | Model Size | Target Tokens | Focus |
|-------|-----------|---------------|-------|
| 1B    | 1B params | 20B tokens    | Foundation, high B0/B1 |
| 3B    | 3B params | 40B tokens    | Intermediate |
| 8B    | 8B params | 100B tokens   | Advanced |
| MoE   | 16B active| 240B tokens   | Specialized, high B4/B5 |

## Reproducibility

All operations are:
- Deterministic (seed-controlled)
- Versioned (config hashes in manifests)
- Auditable (full index and metadata tracking)
