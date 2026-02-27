# Coreset Selection Engine (Skeleton Implementation)

> [!IMPORTANT]
> This is a **skeleton implementation** using **synthetic (dummy) data**. It is designed to demonstrate the architectural concepts of a coreset selection pipeline and how it can be implemented to satisfy complex data curation requirements.

This project implements a deterministic, multi-stage **Coreset Selection Engine** designed to produce high-quality training datasets from massive raw data pools. The engine optimizes for curriculum learning, semantic diversity, and strict budget adherence.

## 🚀 Key Concepts

### 1. Stratified Selection
Data is first grouped into **Strata** based on metadata (Difficulty Band, Domain, Language). This allows the engine to sample precisely from specific sub-populations to meet complex distribution targets.
- **File**: `selection/stratifier.py`

### 2. Selection Policy (Scoring)
Every chunk is assigned a priority score. The engine uses a greedy approach, selecting the highest-scoring chunks within each stratum first.
- **File**: `selection/selection_policy.py`

### 3. Density-Based Pruning
To prevent the model from over-training on redundant data, the `DensitySampler` enforces a token cap per semantic cluster (`cluster_id`). This ensures a diverse representation of concepts even within a single domain.
- **File**: `selection/dense_sampler.py`

### 4. Hard Invariants & Validation
The engine enforces "Physics of Data" through strict invariant checks:
- **Exclusion Index**: Ensures no data point is ever reused across stages.
- **Rolling Window Smoothness**: Prevents distribution "spikes" that could destabilize training.
- **File**: `output/invariants.py`
- **Rolling Window Smoothness**: A unique invariant that validates local distribution ratios within a sliding token window, ensuring a stable training signals.

### 5. Curriculum Shuffling
The engine shuffles selected chunks before final manifest generation. This preserves global metadata targets while ensuring that the training stream is locally balanced (preventing "bursts" of a single difficulty or domain).
- **File**: `selection/selection_loop.py`

### 6. Summary Diagnostics
Automatically prints a token-weighted summary of the selection profile at the end of each stage.
- **File**: `viz/diagnostics.py` (via `print_stage_summary`)

---

## 📋 Charter Mapping

### 🎯 Distribution Preservation
- **Curriculum Difficulty**: Supports targeted ratios across **B0–B5** bands via `StageSpec`.
- **Domain Coverage**: Preserves ratios across `web`, `code`, `math`, `reasoning`, and `Indic` datasets.
- **Protected Slices**: Implements a priority-sort in the `selection_loop` to ensure rare/critical data (B4/B5) is never dropped due to density caps.

### 🏗️ Stage Integrity
- **Distinct Coresets**: The `ExclusionIndex` persists state to disk, guaranteeing that Stage 2 never selects Stage 1 data.
- **Exact Budgets**: The `BudgetTracker` enforces strict token limits (20B, 40B, 100B, 240B) down to the chunk level.

### 🎢 Curriculum Smoothness
- **Anti-Spike Constraints**: The `rolling_window_smoothness` invariant validates that no band/domain exceeds its allowed share within any contiguous token window.
- **Transitions**: Controlled via `StageSpec` configurations to ensure monotonic increase in difficulty.

### 🧪 Reproducibility
- **Deterministic**: Every selection is controlled by a `seed` in the `StageSpec`.
- **Manifests**: Every run emits a JSON manifest containing chunk IDs, composition stats, and configuration hashes for full auditability.

---

## 🛠️ Project Structure

```text
├── accounting/         # Token budgeting and exclusion tracking
├── data/               # Chunk definitions and toy dataset generation
├── output/             # Manifest writers and invariant checks
├── selection/          # Core algorithms (Scoring, Pruning, Looping)
├── stage/              # Lifecycle management and Stage specifications
├── viz/                # Diagnostics and distribution visualization
└── dry_run.py          # Integration script demonstrating the full pipeline
```

## 🏁 Getting Started

To simulate a multi-stage selection run and view the diagnostics:

```bash
python dry_run.py
```

**Outputs:**
- `out/stage_evolution.png`: Visualization of band distribution shifts.
- `out/*_manifest.json`: Stage-specific index files.
- `out/all_chunks.json`: The source dataset used for the dry run.

---

## 📈 Evolution to Production

To transition this skeleton into the complete pipeline described in the project charter, the following modules must be replaced with production-grade implementations:

### 1. Data Ingestion (Team 1 Integration)
- **Current**: Randomly generated chunks in `data/toy_dataset.py`.
- **Production**: Real-time streaming or batch loading of clean, approved datasets from Team 1 (e.g., loading from Parquet or JSONL files).

### 2. Semantic Clustering (Density Pruning)
- **Current**: Random `cluster_id` strings.
- **Production**: Integration of embedding-based clustering (e.g., using FAISS or KMeans on Llama/BERT embeddings) to detect true semantic redundancy.

### 3. Sophisticated Scoring
- **Current**: Static `quality_score`.
- **Production**: Dynamic scoring based on curriculum definitions from Team 2, including toxicity filters, educational value classifiers, and "loss-based" active learning signals.

### 4. Scalability & Speed
- **Current**: Sequential loop on a single CPU thread.
- **Production**: Multi-threaded or distributed selection (Ray/Spark) to process the ~400B token target across massive clusters efficiently.

### 5. Automated Validation
- **Current**: Basic invariant assertions.
- **Production**: Automated proxy training comparisons (training a small model on the coreset vs the full set) to provide empirical evidence of selection quality.

---
## 📦 Required Submissions (Project Status)

| Requirement | Status | Implementation in Skeleton |
| :--- | :--- | :--- |
| **coreset_builder.py** | **Active** | Core selection orchestration and prioritized looping logic. Configurable pipeline for multi-stage generation. |
| **Stage manifests** | **Active** | JSON output with IDs, composition, and seed controls. |
| **Validation Report** | **Active** | Rolling window smoothness and overlap invariants implemented. |
| **Total Tokens** | **Target** | Scalable logic ready for ~400B token target across 4 stages. |