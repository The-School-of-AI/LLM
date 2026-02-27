# Repository Cleanup

This document lists what can be safely removed now that the final folder structure is in place.

---

## Delete — Superseded Script Folders

Everything in these folders has been superseded by `pipeline/jobs/`. The final versions are there with full git history (V4.0 → V5.0 → v7.1 in `main_job.py`).

| Path | Reason |
|------|--------|
| `final_scripts/` | Entire folder — contents moved to `pipeline/jobs/` and `docs/` |
| `emr_serverless/` | Entire folder — superseded by `pipeline/jobs/` |
| `emr/` | Entire folder — classical EMR cluster, replaced by EMR Serverless |
| `glue_jobs/new_datasets/` | Subfolder — scripts moved to `pipeline/jobs/` |
| `glue_jobs/prod_runs/` | Subfolder — contains a copy of V5, authoritative copy is in git history |
| `glue_jobs/t2_fast_metrics_glue.py` | Early fast Glue attempt |
| `glue_jobs/metrics_calculator_single.py` | Single-file test runner |
| `glue_jobs/gluetoemr.py` | One-off migration utility, migration complete |
| `glue_jobs/"t2_metrics_calculator_v5 copy.py"` | Copy of V5 |
| `glue_jobs/__pycache__/` | Python cache |

---

## Delete — Working Notes and Transient Files

| Path | Reason |
|------|--------|
| `glue_jobs/notes/` | Entire folder (raw working notes, .ignore/ old versions, ad-hoc changelogs) — consolidated into `docs/CHANGELOG.md` |
| `glue_jobs/run_glue_jobs.sh` | Glue orchestration shell script, no longer used |
| `glue_jobs/run_glue_jobs_flex.txt` | Notes on Glue FLEX execution |
| `glue_jobs/size_vs_workers.csv` | Internal benchmarking |
| `glue_jobs/my_runs_glue_job_triggers.sh` | Personal run triggers |
| `data_processing/` | Entire folder — `design_principles.md` moved to `docs/`, `notes.md` and `dataset_size_and_status.csv` are transient |
| `chat_history.md` | Internal conversation log — context captured in `docs/` |
| `training_data_metrics.csv` | Transient run output |
| `emr/deploy_emr_fast.txt`, `emr/high_compute_emr*.txt`, `emr/run_production_fast.txt` | EMR classical cluster deployment notes |

---

## Delete — data_explorer Folder

| Path | Reason |
|------|--------|
| `data_explorer/` | Standalone local exploration tool, not part of the pipeline. The large `.parquet` files inside should not be in source control. Analysis outputs are in `analysis/reports/`. |

---

## Keep — Historical Reference (Do Not Delete)

| Path | Why |
|------|-----|
| `glue_jobs/claude_reviewed/v1_t2_metrics_calculator_v5.py` | V5 Glue baseline — use `git log pipeline/jobs/main_job.py` to see the same content in history |
| `glue_jobs/docs/T2_V5.0_CHANGELOG.md` | V5 metric removal with evidence — content is in `docs/CHANGELOG.md` but original is more detailed |
| `glue_jobs/docs/T2_V5.0_PROJECT_STATE_UPDATE.md` | V5 state snapshot |
| `glue_jobs/b5_patterns/` | B5 pattern research docs |
| `glue_jobs/v5_metadata_aware_difficulty.md` | Explains metadata override (and why it was removed) |
| `curriculum_tags/` | Phase 1 (Python plugin extractor) — historical, no Spark |
| `scripts/exploration/` | Analysis scripts (analysis outputs already in `analysis/reports/`) |

---

## Final Folder Structure (After Cleanup)

```
2_curriculum_architects/
├── curriculum.yaml                    # Canonical policy (source of truth)
├── ARCHITECTURE_AND_DECISIONS.md     # Architecture, decisions, reproduction guide
├── charter.md                         # Team mandate
├── README.md                          # Quick start
├── pyproject.toml
│
├── pipeline/                          # Production EMR Serverless jobs
│   ├── README.md                      # How to run
│   └── jobs/
│       ├── main_job.py                # Large-scale datasets (git history: V4→V5→v7.1)
│       ├── curated_datasets_job.py    # Curated HF datasets
│       └── student_data_job.py        # Student-generated data
│
├── docs/                              # All documentation
│   ├── CHANGELOG.md                   # Complete version history
│   ├── band_assignment_methodology.md
│   ├── pipeline_evolution.md
│   ├── band_definitions.md
│   └── design_principles.md
│
├── src/                               # Reference Python libraries
│   ├── curriculum_extractor/          # Single-record extraction reference
│   ├── curriculum_reader/             # Batch creation utilities
│   ├── band_assignment.yaml           # Config for curriculum_extractor
│   └── metrics_config.yaml
│
├── analysis/
│   └── reports/                       # Empirical validation reports
│
├── glue_jobs/                         # Historical — V5 Glue reference
│   ├── claude_reviewed/v1_t2_metrics_calculator_v5.py
│   ├── docs/
│   ├── b5_patterns/
│   └── v5_metadata_aware_difficulty.md
│
├── curriculum_tags/                   # Historical — Phase 1 Python extractor
├── scripts/exploration/               # Analysis scripts
├── examples/                          # Usage examples
├── postprocess/                       # Post-processing pipeline
├── tests/                             # Test suite
└── logs/                              # EMR Serverless run logs
```
