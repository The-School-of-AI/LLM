# Team 2: Curriculum Architects — Capstone Progress Update

> **Team Focus**: Building the "educational pathway" for training Large Language Models — deciding what data the model learns, when it learns it, and how to progressively increase difficulty.

---

## Overview

Our team is responsible for **Curriculum Learning** — the idea that, just like students in school, AI models learn better when they start with simpler content and gradually progress to more complex material.

We've developed three interconnected systems:

| Component | Purpose |
|-----------|---------|
| **Metadata Tagging System** | Analyzes and labels every piece of training data with quality and difficulty metrics |
| **Curriculum Constitution** | The "rulebook" defining how training data should be organized by difficulty level |
| **Data Sampler** | Tool for exploring and sampling datasets from cloud storage |

---

## 1. Metadata Tagging System

### What It Does (Non-Technical Summary)

Think of this as a **quality inspector** for training data. Before we feed billions of documents to train an AI model, we need to know:
- How difficult is this text? (Is it a children's book or a PhD thesis?)
- What type of content is it? (Code, news, academic paper?)
- Which "grade level" should it be assigned to?

The tagging system automatically analyzes each document and adds these labels ("metadata tags") so the training pipeline knows when to use each piece of data.

### Technical Details

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/curriculum_tags/` |
| **Purpose** | Auto-discovering plugin system that computes curriculum metadata tags for training datasets |
| **Input Format** | JSONL records in Parquet files (normalized key/value pairs from Team 1) |
| **Output** | Original Parquet files with added `curriculum_tags` field + separate metadata summary file |

### Current Metrics (Plugin-Based)

| Metric | What It Measures |
|--------|------------------|
| **Difficulty** | Overall text complexity score (0-1 scale) |
| **Readability** | Flesch-Kincaid grade level, Flesch Reading Ease |
| **Modality** | Content type classification (general text, code, math, etc.) |
| **Band Assignment** | Maps difficulty to curriculum "grade level" (B0-B5) |
| **Tokenizer Difficulty** | Token-level complexity analysis |
| **Entropy** | Information density and predictability |
| **Diversity** | Lexical richness (MTLD, vocabulary variety) |

> [!NOTE]
> The metrics are not fully exhaustive yet. This is a **plugin system** — we will finalize the complete metric set after receiving final data format specifications from Team 1.

### How It Works

### How It Works

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  S3 Parquet     │ ──▶ │  CurriculumTagger │ ──▶ │  Tagged Parquet     │ ──▶ │  Team 3             │
│  (from Team 1)  │     │  (runs metrics)   │     │  + Metadata File    │     │  (Coreset Eng.)     │
└─────────────────┘     └───────────────────┘     └─────────────────────┘     └─────────────────────┘
```

1. Read Parquet files from S3 in **distributed manner** (batch processing)
2. Apply each metric plugin in sequence (metrics can see results from previous metrics)
3. Assign curriculum band (B0-B5) based on computed metrics
4. Write processed files back to S3 with updated tags
5. Generate separate metadata file for analytics and statistics

### Setup/Dependency

```bash
cd experiments/2_curirculum_architects
uv pip install -e .
```

### Run Command

*(These are examples, not the actual runs. See `examples/` folder for more.)*

```bash
# Basic usage
uv run python examples/basic_usage.py

# Process Parquet files
uv run python examples/parquet_processing.py

# Run tests
uv run pytest tests/ -v
```

---

## 2. Curriculum YAML Structure (The "Constitution")

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
> See [band_assignment.yaml](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/curriculum_tags/metrics/band_assignment.yaml) and [BAND_ASSIGNMENT.md](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/curriculum_tags/metrics/BAND_ASSIGNMENT.md) for complete implementation details.

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

## 3. Data Sampler

### What It Does (Non-Technical Summary)

A **convenience tool** for exploring and sampling large datasets stored in AWS S3. When you have terabytes of training data, you can't download everything — this tool lets you:
- Randomly sample files from cloud storage
- Preview file contents without downloading
- Filter by filename patterns and folders

### Features

- 🎲 Random sampling from S3 buckets
- 🔍 Filtering by filename patterns (regex) and folder paths
- 📊 Preview Parquet and JSONL files (S3 and local)
- 💾 Download with configurable limits
- 🌐 Web interface (dark-themed UI)
- 🔓 Support for public S3 buckets (anonymous access)

| Attribute | Details |
|-----------|---------|
| **Path** | `experiments/2_curirculum_architects/data_sampler/` |
| **Purpose** | Sample and preview large datasets from S3 and local storage |
| **Interfaces** | CLI + Web UI |

### Setup/Dependency

```bash
cd experiments/2_curirculum_architects/data_sampler
uv sync
```

### Run Command

*(These are examples, not the actual runs.)*

```bash
# CLI - Sample files from S3
uv run python -m src.cli sample my-bucket 50

# CLI - Preview a file
uv run python -m src.cli preview s3://my-bucket/data/file.parquet

# Web Interface
uv run python web/app.py
# → Access at http://localhost:5000
```

---

## What's Next

| Item | Status | Notes |
|------|--------|-------|
| Finalize metric plugins | 🔄 In Progress | Waiting for final data format from Team 1 |
| Freeze curriculum.yaml | 📋 Pending | Will freeze after Team 1 confirmation |
| Band threshold calibration | 📋 Pending | Requires sample data run |
| Integration testing | 📋 Pending | End-to-end pipeline with real data |

---

## Quick Reference: File Locations

| Component | Path |
|-----------|------|
| Curriculum Policy | [`curriculum.yaml`](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/curriculum.yaml) |
| Metrics Config | [`metrics_config.yaml`](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/metrics_config.yaml) |
| Tagging Library | [`metrics/README.md`](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/curriculum_tags/metrics/README.md) |
| Data Sampler | [`data_sampler/README.md`](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/data_sampler/README.md) |
| Examples | [`examples/README.md`](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/examples/README.md) |
| Tests | `experiments/2_curirculum_architects/tests/` |

---

**Team 2 — Curriculum Architects**  
*Building the educational pathway for LLM training*
