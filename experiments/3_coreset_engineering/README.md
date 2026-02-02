# Team 3: Coreset Engineering Experiment

## Overview
This directory contains the engineering pipeline for creating **Stage-Specific Coresets** for LLM Training.
The pipeline absorbs raw data chunks, scores them for difficulty, removes duplicates, and samples them into curriculum-aligned stages (1B -> 3B -> 8B -> 70B).

**Key Features:**
*   **Production Curriculum Support**: Parses `curriculum.yaml` to enforce specific Profiles, Modalities (Code, CoT, etc.), and Difficulty Bands (B0-B5).
*   **Stratified Sampling**: Ensures strict adherence to `band_weights` and `modality_weights`.
*   **Reproducibility**: Deterministic pipeline with verifiable manifest outputs.

## Pipeline Workflow

```text
+----------------+      +-----------+       +-------+
| Raw Data Input | ---> | Ingestion | --->  | Dedup |
+----------------+      +-----------+       +-------+
                                                |
                                           (Unique Only)
                                                |
                                                v
                                       +-------------------+
                                       | Difficulty Scorer |
                                       +-------------------+
                                                |
                                                v
                                         +--------------+
                                         |   Bucketer   |
                                         +--------------+
                                                |
                                     (Assigns B0-B5 + Modality)
                                                |
                                                v
 +-------------------+             +-----------------------+
 | Curriculum Config | ----------->|   Stratified Sampler  | <--- [ Unified Bucket Pool ]
 +-------------------+             +-----------------------+
                                                |
                                  +-------------+-------------+
                                  |             |             |
                            +-----------+ +-----------+ +-----------+
                            | Stage 1B  | | Stage 3B  | | Stage ... |
                            +-----------+ +-----------+ +-----------+
                                  |             |             |
                                  v             v             v
                           +---------------------------------------+
                           |           Output Manifests            |
                           +---------------------------------------+
                                                |
                                                v
                                     +---------------------+
                                     | Audit Visualization |
                                     +---------------------+
```

## Directory Structure
- **src/**: Core Python package `coreset_engine`.
    - `selection/curriculum.py`: Parses the YAML and resolves Stage Profiles.
    - `selection/bucketer.py`: Maps data to B0-B5 bands based on difficulty scores.
    - `selection/sampler.py`: Implements multidimensional sampling logic (Band + Modality).
- **configs/**: Configuration files.
    - `curriculum.yaml`: **Primary Production Config** (The "Brain" of the pipeline).
    - `curriculum_proto.yaml`: Legacy/Prototype config (Archived).
- **scripts/**: CLI entry points.
    - `coreset_builder.py`: Main execution script.
    - `generate_mock_data.py`: Generates test data with rich metadata.
    - `audit_visuals.py`: Visualization suite.
- **data/**: Local directory for input datasets.
- **output/**: Generated manifests and indices.
- **tests/**: Unit tests (located in `tests/3_coreset_engineering` at project root).

## Usage

All commands should be run from the **project root**. We standardize on `uv run` for consistent environment management.

### 1. Setup
Ensure dependencies are installed:
```bash
uv sync
```
*(Or manually: `pip install PyYAML pandas matplotlib seaborn`)*

### 2. Generate Mock Data
Generate a mock dataset containing realistic metadata (Domains, Modalities, Difficulty Scores) to test the B0-B5 logic.
```bash
uv run experiments/3_coreset_engineering/scripts/generate_mock_data.py \
    --count 10000 \
    --output experiments/3_coreset_engineering/data/mock_input
```

### 3. Run the Pipeline
Run the builder using the Production Curriculum (`curriculum.yaml`):
```bash
uv run experiments/3_coreset_engineering/scripts/coreset_builder.py \
    --config experiments/3_coreset_engineering/configs/curriculum.yaml \
    --data experiments/3_coreset_engineering/data/mock_input \
    --output experiments/3_coreset_engineering/output/manifests
```

### 4. Visualization Audits (Post-Hoc)
Generate quality and distribution reports (PNG charts) from the manifests to verify curriculum compliance:
```bash
uv run experiments/3_coreset_engineering/scripts/audit_visuals.py \
    --manifest_dir experiments/3_coreset_engineering/output/manifests \
    --output_dir experiments/3_coreset_engineering/output/audits
```

**Generated Plots (`output/audits/`):**
*   `{stage}_band_dist.png`: Difficulty Band Histogram (verifies Band Weights).
*   `{stage}_modality_dist.png`: Modality Breakdown (verifies Modality Weights).
*   `{stage}_score_box.png`: Quality Score box-and-whiskers per Band (verifies Bucketing).

### 5. Running Tests
Run the unit tests specific to this experiment:
```bash
uv run --package coreset-engineering pytest tests/3_coreset_engineering
```

## Configuration Details (`curriculum.yaml`)

This file is the "Brain" of the pipeline. It controls *what* data gets selected for *which* stage.

### 1. Growth Schedule
Defines the timeline of training.
```yaml
growth_schedule:
  stages:
    - name: "1B"
      curriculum_profile: "base"           # Uses the 'base' profile
    - name: "3B"
      curriculum_profile: "harder_shift_1" # Switches to harder profile
```

### 2. Difficulty Bands (B0-B5)
The canonical definition of difficulty ranges. The pipeline maps raw scores (e.g., compression ratio) to these bands.
*   **B0 (Nursery)**: Very easy, repetitive text.
*   **B1-B3**: Standard web/academic text.
*   **B4 (Graduate)**: Technical text, basic code.
*   **B5 (PhD)**: Advanced complex reasoning, heavy code, pseudo-code.

### 3. Modalities
Explicit definitions of data types.
*   `general_text`: Standard NLP training data.
*   `code`: Programming languages.
*   `cot_reasoning`: Chain-of-Thought traces.
*   `agentic_traces`: Tool use examples.

### 4. Stage Profiles (The Logic)
This is where the sampling distribution is defined.
```yaml
stage_profiles:
  base:  # 1B Stage
    band_weights:     # MUST sum to 1.0 (approx)
      B0: 0.30        # 30% Easy data
      ...
      B5: 0.02        # Only 2% Hard data
    modality_weights: # Preferences within bands
      general_text: 0.86
      code: 0.12
```

## Output Artifacts
Manifests are written to `output/manifests/`.
*   `*_index.jsonl`: The actual dataset index used by DataLoaders. Contains `file_path`, `line_number`, `token_count`.
*   `*_manifest.json`: Summary metadata including token counts and distribution stats.

## Code to Task Mapping (Summary)

| L2 Task | Implementation File | Status / Note |
| :--- | :--- | :--- |
| **Exact & Near Dedup** | `filters/dedup.py` | Implements Exact (MD5) and MinHash logic. *Note: Current loop uses Exact Only for speed.* |
| **Diversity Metrics** | `scoring/perplexity.py` | Uses Compression Ratio as a proxy for entropy/diversity. |
| **Bucket Scoring** | `selection/bucketer.py` | Maps raw scores -> B0-B5 Bands. |
| **Protected Slices** | `selection/sampler.py` | Enforced via `modality_weights`. (0.0 weight = excluded). |
| **Coverage Audits** | `selection/builder.py` | Logs distribution stats (Band/Modality) to manifest footer. |
| **Visual Audits** | `scripts/audit_visuals.py` | Generates PNGs for Band/Modality/Score distributions. |
