# Curriculum Construction Playbook

This repository provides a reusable, dataset-agnostic implementation for creating high-quality training curricula without training expensive proxy models.

---

## Core Philosophy

1.  **Signals-First**: Use cheap, fast-to-compute signals (token counts, entropy, regex-based modality tags) instead of heavy model inference.
2.  **Dataset Agnostic**: Works on any JSONL dataset (FineWeb, Dolma, etc.) with a configurable text key.
3.  **Principled Optimization**: Instead of arbitrary heuristics, we use **KL-regularized optimization** to match target difficulty distributions while preserving the natural diversity of your data.

---

## Key Components

### 1. Difficulty Scoring (`src/curriculum_tools.py`)

We compute a continuous difficulty score based on multiple signals:
- Flesch-Kincaid Grade level (on long docs)
- Character Entropy (proxy for information density/code)
- Modality bumps (Code, Math, Chain-of-Thought)

Scores are then mapped to **6 Difficulty Bands (B0-B5)** using quantile edges computed from your specific data.

### 2. Curriculum Generation (`src/curriculum_yaml_generator.py`)

We automatically generate a `curriculum.yaml` configuration that defines:
- **Growth Schedule**: How the curriculum evolves from 1B -> 70B scale.
- **Stage Profiles**: Specific weights for each band at each stage.
- **Guardrails**: Safety caps on specific domains (e.g., "Max 8% Hindi", "Max 6% CoT").

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Experiment
Use the provided script to process your data and generate a curriculum.
Supports `.jsonl` (text field auto-detected) and `.parquet` (requires pandas/pyarrow).

```bash
python scripts/run_experiment.py <input_file> <output_directory>
```

**Example:**
```bash
python scripts/run_experiment.py data/sample_10k.jsonl outputs/run_1
# OR
python scripts/run_experiment.py data/sample.parquet outputs/run_1
```

### 3. Output
The script produces:
- `curriculum.yaml`: The master configuration file.
- `base_distribution.json`: The difficulty profile of your raw data.
- `band_edges.json`: The score thresholds used for banding.

---

## How It Works (Under the Hood)

### Automated Text Key Detection
The script automatically finds the text field in your JSONL data:

```python
# scripts/run_experiment.py
def pick_text_key(row):
    for k in ["text", "content", "raw", "document", "body"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return k
    return None
```

### Band Weight Optimization
To ensure we meet capacity targets (e.g., "harder data for 8B models") without running out of data, we use an optimizer:

```python
# src/curriculum_tools.py
def optimize_band_weights(base, target, ...):
    # Optimizes weights to match target difficulty
    # while minimizing KL Divergence from the base distribution
    # ...
```

