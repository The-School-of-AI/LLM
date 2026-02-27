# Team 2: Curriculum Architects

**Scope:** Curriculum policy design and difficulty-band assignment for the LLM pretraining corpus
**Status:** Production — EMR Serverless

---

## What This Team Does

Team 2 assigns a difficulty band (B0–B5) to every document in the ~4TB training corpus. These band labels drive which content the model sees at each growth stage (1B → 3B → 8B → 70B parameters). The core thesis from the team's charter:

> Curriculum errors do not fail loudly. They silently waste trillions of tokens. This team exists to prevent that.

Two concrete deliverables:

1. **`curriculum.yaml`** — the canonical policy: band definitions, growth schedule, domain restrictions, and guardrails. This is the contract with the training team.
2. **Band assignment pipeline** — three EMR Serverless PySpark jobs that label every document in the corpus.

---

## Repository Structure

```
2_curriculum_architects/
│
├── curriculum.yaml              # Canonical curriculum policy (start here)
├── ARCHITECTURE_AND_DECISIONS.md # Full technical history, decisions, and rationale
│
├── pipeline/                    # Production EMR Serverless jobs
│   ├── README.md                # How to run — job args, Spark config, expected output
│   └── jobs/
│       ├── main_job.py          # Large-scale datasets: RedPajama, FineWeb, Dolma, arXiv (B0–B5)
│       ├── curated_datasets_job.py  # 17 HuggingFace datasets, source-clamped per dataset
│       └── student_data_job.py  # ERAv4 Q&A drills + Samvaad conversation (B0–B2)
│
├── docs/                        # All documentation
│   ├── CHANGELOG.md             # Full version history r2.1 → r7.1
│   ├── band_assignment_methodology.md  # Algorithm spec: formulas, thresholds, worked example
│   ├── pipeline_evolution.md    # PatternRefinement r5.0 → WeakSignals r7.1 comparison
│   ├── band_definitions.md      # B0–B5 canonical definitions
│   ├── design_principles.md     # Core design principles
│   └── analysis/                # Empirical validation reports (333K-sample runs)
│
├── src/                         # Reference Python libraries (not the Spark jobs)
│   ├── curriculum_extractor/    # Single-record extraction reference implementation
│   ├── curriculum_reader/       # Band assignment utilities reference implementation
│   ├── examples/                # Usage examples for the libraries above
│   └── tests/                   # Test suite for the libraries above
│
├── scripts/                     # Analysis, exploration, and utility scripts
│   └── postprocess/             # Post-processing pipeline
│
├── curriculum_tags/             # Phase 1 historical reference (Python, pre-Spark)
└── logs/                        # EMR Serverless run logs
```

---

## Quick Start

### Run the Band Assignment Pipeline

```bash
# Submit main job to EMR Serverless
aws emr-serverless start-job-run \
  --application-id <APP_ID> \
  --execution-role-arn <ROLE_ARN> \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "s3://your-bucket/scripts/main_job.py",
      "entryPointArguments": [
        "--INPUT_BASE", "s3://your-bucket/t1-output/",
        "--OUTPUT_BASE", "s3://your-bucket/t2-output/",
        "--JOB_NAME", "t2_main_r7"
      ]
    }
  }'
```

See `pipeline/README.md` for full config: Spark settings, all three jobs, expected output structure, and rejection rate targets.

### Use the Reference Library

`src/curriculum_extractor/` is a Python library for single-record extraction and local analysis. Its metrics and band assignment logic differ from the production Spark jobs — use it for experimentation and debugging, not for reproducing pipeline output.

```bash
# Install
cd experiments/2_curirculum_architects
uv pip install -e .

# Run examples
uv run python src/examples/01_basic_extraction.py

# Run tests
uv run pytest src/tests/ -v
```

---

## The Band System

Six bands map the model's growing cognitive capacity:

| Band | Name | Difficulty | Typical Content |
|------|------|-----------|----------------|
| B0 | Nursery | 0.05 | Simple web text, language drills |
| B1 | Primary | 0.20 | Clean prose, news, simple Q&A |
| B2 | High School | 0.35 | Wikipedia, tutorials, structured dialogue |
| B3 | Undergraduate | 0.55 | Technical docs, meaningful code, multi-step problems |
| B4 | Graduate | 0.75 | Math proofs, algorithms, research papers |
| B5 | PhD | 0.90 | Tool-use traces, advanced math, planning workflows |

The bands are fixed. What changes across training stages is their **sampling proportion** — B0 drops from 49% at 1B to 16% at 70B as the model gains capacity. See `curriculum.yaml` for the full growth schedule.

---

## Key Documents

| Document | What It Covers |
|----------|---------------|
| `curriculum.yaml` | Band policy, growth schedule, domain restrictions, guardrails — the source of truth |
| `ARCHITECTURE_AND_DECISIONS.md` | Full technical history: why we built what we built, what failed and why, all design decisions |
| `docs/band_assignment_methodology.md` | Algorithm specification: difficulty formula, probabilistic banding, source clamping, worked example |
| `docs/CHANGELOG.md` | Version history r2.1 → r7.1 with rationale for every change |
| `pipeline/README.md` | How to run the three jobs end-to-end |
| `docs/analysis/` | Empirical validation reports from 333K-sample runs |

---

## Pipeline Overview

```
T1 Output (Parquet on S3)
         │
    ┌────┴──────────────────────────────────┐
    │                │                      │
main_job.py    curated_datasets_job.py    student_data_job.py
Web/book/code  17 Golden HF datasets   ERAv4 + Samvaad
B0–B5 (full)   Source-clamped           B0–B2 only
    │                │                      │
    └────────────────┴──────────────────────┘
                     │
           Unified output schema
           partitioned by band
                     │
             T3 Training Jobs
```

All three jobs run in parallel on EMR Serverless, one job per dataset source. Output is Parquet, zstd compressed, partitioned by `band`. The schema is identical across all three jobs — T3 reads from a single `OUTPUT_BASE` prefix.

---

## Output Schema

All three jobs write the same column set, partitioned by `band`:

| Group | Columns |
|-------|---------|
| Identity | `uuid`, `id`, `file_path`, `source`, `domain`, `hash`, `language`, `metadata` |
| Band | `assigned_band`, `band`, `difficulty_score`, `band_p_B0`–`band_p_B5` |
| Modality flags | `has_code`, `has_cot`, `has_reasoning`, `has_agentic` |
| Scores | `code_score`, `math_score`, `reasoning_score`, `agentic_score`, `cot_score` |
| Size | `byte_length`, `word_count`, `unique_token_ratio`, `compression_ratio`, `token_count_estimate` |
| Rejection | `is_rejected`, `rejection_reason`, `rejection_level` (rejected records only) |

---

## Upstream / Downstream

**Upstream (Team 1):** T2 consumes Parquet files from Team 1's normalization pipeline. Required fields: `id`, `text`, `source`, `domain`, `language`, `metadata`.

**Downstream (Team 3):** T3 uses `assigned_band` to construct per-stage training batches. The output schema has been kept stable across all pipeline versions to avoid requiring T3 changes.

**Cross-team dependencies:**
- **Team 6 (Tokenizer Lab):** Tokenizer difficulty proxy planned as a secondary validation signal — not yet integrated.
- **Team 17 (Agentic):** B5 agentic assignment is in place; tool-use trace format enforcement is pending Team 17's spec.

