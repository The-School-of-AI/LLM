# Team 2: Curriculum Architects — Capstone Progress Update

> **Team Focus**: Building the "educational pathway" for training Large Language Models — deciding what data the model learns, when it learns it, and how to progressively increase difficulty.

---

## Overview

Our team is responsible for **Curriculum Learning** — the idea that, just like students in school, AI models learn better when they start with simpler content and gradually progress to more complex material.

We've developed a comprehensive data curation pipeline:

| Component | Purpose |
|-----------|---------|
| **curriculum_extractor** | Read-only extraction of metrics from source data |
| **curriculum_reader** | Reading metadata layer for batch creation |
| **Lightning Dataset Sampler** | Browser-based parquet viewer for exploration |
| **Curriculum Constitution** | The "rulebook" defining difficulty levels |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRICULUM PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐    │
│  │   Source    │──────│  curriculum_     │──────│    Metadata      │    │
│  │   Parquet   │ READ │  extractor       │WRITE │    Layer         │    │
│  │   Files     │ ONLY │                  │      │    (.parquet)    │    │
│  └─────────────┘      └──────────────────┘      └──────────────────┘    │
│                                │                         │               │
│                                │                         ▼               │
│                       ┌────────┴────────┐      ┌──────────────────┐     │
│                       │                 │      │  assign_bands.py │     │
│                       ▼                 ▼      │  (post-process)  │     │
│                  ┌─────────┐      ┌─────────┐  └──────────────────┘     │
│                  │ Level 0 │      │ Level N │           │               │
│                  │ Metrics │ ···  │ Metrics │           ▼               │
│                  └─────────┘      └─────────┘  ┌──────────────────┐     │
│                                                │  curriculum_     │     │
│                                                │  reader          │     │
│                                                │  (batch creation)│     │
│                                                └──────────────────┘     │
│                                                         │               │
│                                                         ▼               │
│                                                ┌──────────────────┐     │
│                                                │  Training Loop   │     │
│                                                │  (deterministic) │     │
│                                                └──────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Read-Only Records** | Source records are NEVER modified. `ReadOnlyRecord` wrapper ensures immutability |
| **Early Rejection** | Processing stops at first rejection, saving compute |
| **Level-Based Metrics** | Metrics grouped by level; same level can run in parallel |
| **Metadata Separation** | Metrics stored separately from source data |
| **Post-Processing Bands** | Band assignment happens after extraction, not during |
| **Deterministic Batching** | Training batches are reproducible with same seed |

---

## 1. Curriculum Extractor

### What It Does

The **curriculum_extractor** package extracts metrics from source data while keeping source records completely immutable.

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/curriculum_extractor/` |
| **Purpose** | Read-only extraction of curriculum metrics |
| **Input Format** | JSONL records in Parquet files |
| **Output** | Separate metadata parquet files (source unchanged) |

### Key Components

| Component | Purpose |
|-----------|---------|
| `ReadOnlyRecord` | Immutable wrapper that prevents any modifications |
| `ExtractionResult` | Frozen dataclass with metrics and rejection info |
| `MetricPlugin` | Base class with level support for ordered execution |
| `RecordExtractor` | Main extraction engine with early rejection |

### Current Metrics (Plugin-Based)

| Metric | Level | What It Measures | Can Reject |
|--------|-------|------------------|------------|
| **LengthFilter** | 0 | Token/char length bounds | Yes |
| **LanguageFilter** | 0 | Language detection | Yes |
| **ContentFilter** | 0 | Blocklist/patterns | Yes |
| **PerplexityScorer** | 1 | Language model perplexity | No |
| **QualityClassifier** | 1 | ML quality score | No |
| **DomainClassifier** | 2 | Content domain | No |
| **ReadabilityScorer** | 1 | Flesch-Kincaid grade level | No |
| **DiversityScorer** | 1 | Lexical richness (MTLD) | No |
| **EntropyScorer** | 1 | Information density | No |

### How Levels Work

```
Record → Level 0 (fast filters) → Level 1 (quality) → Level 2 (content)
              ↓                        ↓                    ↓
         If rejected              If rejected          If rejected
              ↓                        ↓                    ↓
         STOP (skip L1, L2)       STOP (skip L2)      Return result
```

### Quick Start

```python
from curriculum_extractor.core.extractor import RecordExtractor

# Initialize extractor
extractor = RecordExtractor("curriculum.yaml")

# Process record (read-only!)
record = {"id": "doc1", "text": "Sample document..."}
result = extractor.extract_record(record)

if result.rejected:
    print(f"Rejected: {result.rejection_reason}")
else:
    print(f"Metrics: {result.metrics}")
```

### Setup/Dependency

```bash
cd experiments/2_curirculum_architects
uv pip install -e .
```

### Run Commands

```bash
# Basic usage
uv run python examples/01_basic_extraction.py

# Process Parquet files
uv run python examples/02_parquet_processing.py

# Run benchmark
uv run python examples/07_benchmarking.py

# Run tests
uv run pytest tests/ -v
```

---

## 2. Curriculum Reader

### What It Does

The **curriculum_reader** package reads metadata from the extraction layer and creates training batches.

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/curriculum_reader/` |
| **Purpose** | Read metadata layer and create deterministic training batches |
| **Input** | Metadata parquet files from curriculum_extractor |
| **Output** | Batches of records with source data joined |

### Key Components

| Component | Purpose |
|-----------|---------|
| `MetadataReader` | Read and query metadata layer |
| `BatchCreator` | Create deterministic batches with stratification |
| `MetadataAnalyzer` | Statistics and distributions |
| `RejectionReader` | Analyze rejection patterns |

### Quick Start

```python
from curriculum_reader.batch_creator import BatchCreator

creator = BatchCreator(
    metadata_dir="./metadata",
    source_dir="./source",
    batch_size=1000,
    seed=42,  # Deterministic
)

# Generate batches with band proportions
for batch in creator.create_batches(
    band_proportions={"high": 0.5, "medium": 0.3, "low": 0.2}
):
    for record in batch:
        train(record)
```

---

## 3. Band Assignment (Post-Processing)

### What It Does

Band assignment happens **after** extraction by reading the metadata layer. This separates concerns and allows re-assignment without re-extraction.

```bash
# Assign bands to metadata
python -m curriculum_extractor.scripts.assign_bands \
    --curriculum curriculum.yaml \
    --metadata ./metadata \
    --output ./metadata_with_bands
```

### Quick Start

```python
from curriculum_extractor.scripts.assign_bands import BandAssigner

# Configure bands
bands = [
    {"name": "B0", "min_score": 0.0, "max_score": 0.20},
    {"name": "B1", "min_score": 0.20, "max_score": 0.40},
    {"name": "B2", "min_score": 0.40, "max_score": 0.60},
    {"name": "B3", "min_score": 0.60, "max_score": 0.75},
    {"name": "B4", "min_score": 0.75, "max_score": 0.90},
    {"name": "B5", "min_score": 0.90, "max_score": 1.0},
]

# Assign bands to metadata
assigner = BandAssigner("curriculum.yaml", bands)
stats = assigner.process_metadata_directory("./metadata")
print(f"Band distribution: {stats}")
```

---

## 4. Lightning Dataset Sampler

### What It Does

A **browser-based tool** for exploring parquet files. Runs on GitHub Pages with no backend required.

| Feature | Description |
|---------|-------------|
| 🔗 S3 URL | Load from signed S3 URLs |
| 🌐 HTTP URL | Load from any HTTP endpoint |
| 📁 Local File | Drag & drop or file picker |
| 👁️ Views | Toggle raw text / JSON / metadata |
| 🎲 Random | Jump to random record |
| 🔍 Search | Find by record ID or content |
| ⏱️ Timing | Console logs for performance |

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/lightning_dataset_sampler/` |
| **Tech Stack** | Vite + Apache Arrow (browser WASM) |
| **Deploy** | GitHub Pages via workflow |

### Run Locally

```bash
cd lightning_dataset_sampler
npm install
npm run dev
```

### Deploy to GitHub Pages

```bash
npm run build
# Or push to trigger .github/workflows/deploy-sampler.yml
```

---

## 5. Curriculum YAML Structure (The "Constitution")

### What It Does (Non-Technical Summary)

We think of the curriculum as a **court system, not a police force**. It doesn't enforce rules in real-time — instead, it defines the *principles* and *boundaries* that all other systems must follow.

This YAML file is the **single source of truth** for training policy decisions:
- What difficulty levels exist (like grades K-12 → PhD)
- What types of content are allowed at each level
- When to introduce new languages or advanced reasoning

### The Band System (Think: Education Levels)

Our curriculum uses **six difficulty bands** (B0-B5) inspired by educational progression. Each band has specific constraints on readability, difficulty scores, entropy, and diversity metrics.

#### Band Overview

| Band | Name | Intent | Allowed Modalities |
|------|------|--------|-------------------|
| **B0** | Nursery | Surface language acquisition | General text |
| **B1** | Primary | Fluent everyday language | General text, clean exposition |
| **B2** | High School | Structured knowledge | General text, structured knowledge |
| **B3** | Undergraduate | Reasoning emergence | Structured knowledge, technical text, code |
| **B4** | Graduate | Explicit abstraction | Technical text, math, code, planning/reasoning |
| **B5** | PhD | Planning & system reasoning | Hard reasoning, math, advanced code, planning |

#### How Band Assignment Works

The `BandAssignmentMetric` is the **final decision maker** that aggregates signals from all other metrics. It follows a strict hierarchical logic:

**1. Modality Overrides (Highest Priority)**

Certain content types force specific bands regardless of text complexity:

| Signal | Target Band | Reason |
|--------|-------------|---------|
| Agentic Traces | **B5** | Agentic planning restricted to PhD level |
| Research Papers | **B4/B5** | B5 if highly complex (FK Grade > 16), otherwise B4 |
| Code/Math | **B2-B5** | B5 if Difficulty > 0.8, B4 if > 0.6, B3 if > 0.4, else B2 |

**2. Quality Floors (Safety Nets)**

- **COT Floor**: If Chain-of-Thought traces detected → minimum **B3** (prevents reasoning content from being misclassified as simple)

**3. Constraint-Based Classification**

For general text, we use multi-constraint matching. A sample qualifies for a band if it meets ALL criteria:

| Band | Difficulty Levels | Readability (FK) | Difficulty Score | Entropy | Diversity |
|------|------------------|------------------|------------------|---------|-----------|
| **B0** | L0, L1 | 0.0 - 6.0 | 0.0 - 0.30 | 0.0 - 4.5 | 0.00 - 0.15 |
| **B1** | L1, L2, L3 | 4.0 - 10.0 | 0.20 - 0.50 | 3.5 - 5.5 | 0.10 - 0.25 |
| **B2** | L2, L3, L4 | 8.0 - 14.0 | 0.40 - 0.70 | 4.0 - 6.0 | 0.15 - 0.35 |
| **B3** | L3, L4 | 12.0 - ∞ | 0.60 - 0.85 | 4.5 - ∞ | 0.20 - ∞ |
| **B4** | L4, L5 | 14.0 - ∞ | 0.75 - ∞ | 5.0 - ∞ | 0.25 - ∞ |
| **B5** | L5 | 16.0 - ∞ | 0.85 - ∞ | 5.5 - ∞ | 0.30 - ∞ |

**Overlap Policy**: If a sample qualifies for multiple bands, we assign the **highest** band by default (configurable).

> [!NOTE]
> See [band_assignment.yaml](./band_assignment.yaml) and [BAND_ASSIGNMENT.md](./curriculum_tags/metrics/BAND_ASSIGNMENT.md) for complete implementation details.

### Key Policies Defined

| Policy Area | Description |
|-------------|-------------|
| **Language Policy** | Primary: English (92%), Secondary: Hindi (8%, introduced at 3B stage) |
| **Context Length** | Minimum 4096 tokens for pre-training |
| **Reasoning Rules** | Chain-of-thought forbidden until B3; Agentic content only in B4-B5 |
| **Growth Schedule** | Defines data mix for 1B → 3B → 8B → 70B model stages |
| **Safety Defaults** | Always downgrade on uncertainty, never upgrade |

### Stage-Based Training

As the model grows, the curriculum shifts toward harder content:

```
1B Stage:  30% B0, 28% B1, 20% B2, 14% B3, 6% B4, 2% B5  (focus on basics)
    ↓
70B Stage: 10% B0, 14% B1, 20% B2, 22% B3, 20% B4, 14% B5 (focus on reasoning)
```

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/curriculum.yaml` |
| **Purpose** | Canonical policy document defining bands, modalities, language rules, and growth schedule |
| **Status** | DRAFT (Frozen in two stages: Structure first, then Values) |

**Freeze Policy:**
1. **Structure**: Frozen after Sign-off by Rohan
2. **Value**: Frozen after full insight into data statistics

---

## 3. Band Proportion Calculation
### How We Calculate Stage Weights
The `calculate_proportions.py` script determines the target data mix for a specific model size (e.g., 1B, 3B, 8B, 70B). It uses a mathematical approach to align model capacity with content difficulty.

#### 1. Model Capacity Score ($C$)
We normalize model size on a logarithmic scale (from 1B to 70B parameters) to a 0–1 score:
$$C = \frac{\log(\text{params}) - \log(\text{min\_params})}{\log(\text{max\_params}) - \log(\text{min\_params})}$$

#### 2. Alignment Weight ($W_a$)
Each band has a "Difficulty Centroid" ($D_b$) defined in `curriculum.yaml`. We calculate the alignment between the model's current capacity ($C$) and the band's difficulty ($D_b$):
$$W_a = \exp(-\lambda \cdot |D_b - C|)$$
*$\lambda$ (lambda_align) controls how "sharp" the focus is on matched difficulty.*

#### 3. Proportion Computation
The final proportion for a band ($P_b$) is calculated by combining the base distribution (what occurs naturally in the data) with the alignment weight, then enforcing floors and caps:

1.  **Raw Weight**: $R_b = \text{BaseDist}_b \cdot W_a$
2.  **Constraint Enforcement**: $R'_b = \max(\min(R_b, \text{Cap}_b), \text{Floor}_b)$
3.  **Renormalization**: $P_b = \frac{R'_b}{\sum R'_b}$

#### Usage
To compute proportions for a new dataset:
```bash
uv run python scripts/calculate_proportions.py data/your_file.parquet --recompute
```

---

## 4. Tokenizer Calibration (The T-Scale)
### What is Tokenizer Calibration?
High-frequency words (e.g., "the", "and") are assigned low IDs by modern tokenizers (like Llama-3), while rare technical terms or complex symbols are assigned much higher IDs. We use this relationship to calibrate **Tokenizer Difficulty Levels (T0–T5)**.

### Calibration Thresholds
A document is assigned a T-level by comparing its token ID distribution against these three calibrated thresholds:

| Level | Intent | Avg Token ID | Max Token ID | P95 Token ID |
|-------|--------|--------------|--------------|--------------|
| **T0** | Very Simple | < 5,000 | < 10,000 | < 8,000 |
| **T1** | Everyday Language | < 10,000 | < 20,000 | < 15,000 |
| **T2** | Structured Knowledge | < 20,000 | < 40,000 | < 30,000 |
| **T3** | Technical Content | < 40,000 | < 70,000 | < 60,000 |
| **T4** | Complex Reasoning | < 70,000 | < 100,000 | < 90,000 |
| **T5** | Advanced / Rare | ∞ | ∞ | ∞ |

### Why It Matters
While the **Flesch-Kincaid (FK)** score measures linguistic complexity (sentence/word length), the **Tokenizer Level (T-Scale)** measures "vocabulary density." A text can have simple sentences (high FK) but very rare terminology (high T-level), making it a candidate for a higher Curriculum Band.

> [!CAUTION]
> The current thresholds are calibrated for the **Llama-3.3-70B** tokenizer. This calibration **MUST be re-done** once the final custom tokenizer is provided by **Team 7 (Tokenizer Lab)**, as token ID distributions and vocabulary ranges will shift.

---

## 6. Data Sampler

### What It Does (Non-Technical Summary)

A **convenience tool** for exploring and sampling large datasets stored in AWS S3. When you have terabytes of training data, you can't download everything — this tool lets you:
- Randomly sample files from cloud storage
- Preview file contents without downloading
- Filter by filename patterns and folders

### Features

- 🎲 Random sampling from S3 buckets
- 🔍 Filtering by filename patterns (regex) and folder paths
- 📊 TODO: Preview Parquet and JSONL files (S3 and local)
- 💾 Download with configurable limits
- 🌐 TODO:Web interface (dark-themed UI)
- 🔓 Support for public S3 buckets (anonymous access)

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/scripts/` |
| **Purpose** | Sample and preview large datasets from S3 and local storage |
| **Interfaces** | CLI + Web UI |

### Run Command


```bash
# Process S3 dataset with Ray (distributed)
uv run python scripts/s3_loader.py
```

---

## 7. Benchmarking

Run performance benchmarks with per-metric timing and memory tracking:

```bash
# Run benchmark
python -m curriculum_extractor.scripts.benchmark \
    --curriculum curriculum.yaml \
    --input ./sample_data \
    --batch-size 1000 \
    --max-records 10000
```

### Example Output

```
================================================================================
PIPELINE BENCHMARK RESULTS
================================================================================

Results:
  Total records: 100,000
  Rejected: 12,345 (12.3%)
  Total time: 45.23s
  Throughput: 2,211 records/sec
  
Per-Metric Timing:
  length_filter (L0):     avg=0.05ms, total=5.00s
  language_filter (L0):   avg=0.12ms, total=12.00s
  perplexity (L1):        avg=0.45ms, total=39.42s
  domain_classifier (L2): avg=0.23ms, total=20.15s
  
Memory:
  Peak: 1.2 GB
```

---

## Examples

Comprehensive examples are in [examples/](examples/):

| Example | Description |
|---------|-------------|
| [01_basic_extraction.py](examples/01_basic_extraction.py) | Basic read-only extraction with timing |
| [02_parquet_processing.py](examples/02_parquet_processing.py) | StateManager for incremental processing |
| [03_custom_metrics.py](examples/03_custom_metrics.py) | Creating custom metrics with rejection |
| [04_band_assignment.py](examples/04_band_assignment.py) | Post-processing band assignment |
| [05_metadata_analysis.py](examples/05_metadata_analysis.py) | MetadataAnalyzer usage |
| [06_batch_creation.py](examples/06_batch_creation.py) | Deterministic batch creation |
| [07_benchmarking.py](examples/07_benchmarking.py) | Running performance benchmarks |

---

## What's Next

| Item | Status | Notes |
|------|--------|-------|
| Finalize metric plugins | 🔄 In Progress | Waiting for final data format from Team 1 |
| Freeze curriculum.yaml | 📋 Pending | Will freeze after Team 1 confirmation |
| Band threshold calibration | 📋 Pending | Requires sample data run |
| Integration testing | 📋 Pending | End-to-end pipeline with real data |
| Lightning Sampler Deploy | 📋 Pending | Deploy to GitHub Pages |

---

## Quick Reference: File Locations

| Component | Path |
|-----------|------|
| Curriculum Policy | [`curriculum.yaml`](./curriculum.yaml) |
| Metrics Config | [`metrics_config.yaml`](./metrics_config.yaml) |
| **Extraction Package** | [`curriculum_extractor/`](./curriculum_extractor/) |
| **Reader Package** | [`curriculum_reader/`](./curriculum_reader/) |
| **Lightning Sampler** | [`lightning_dataset_sampler/`](./lightning_dataset_sampler/) |
| Band Assignment Script | [`curriculum_extractor/scripts/assign_bands.py`](./curriculum_extractor/scripts/assign_bands.py) |
| Benchmark Script | [`curriculum_extractor/scripts/benchmark.py`](./curriculum_extractor/scripts/benchmark.py) |
| Examples | [`examples/`](./examples/) |
| Tests | `tests/` |

---

## Changelog

### v2.0.0 (Current)

- **Breaking**: Records are now read-only (no plugin chaining)
- **Breaking**: Band assignment moved to post-processing
- **Added**: Level-based metric execution for potential parallelism
- **Added**: Early rejection for efficiency
- **Added**: Per-metric timing in benchmark
- **Added**: Lightning Dataset Sampler webapp
- **Added**: curriculum_reader package
- **Added**: Comprehensive examples (01-07)

### v1.0.0

- Initial implementation with plugin chaining
- Basic curriculum configuration
- StateManager for incremental processing

---

**Team 2 — Curriculum Architects**  
*Building the educational pathway for LLM training*
