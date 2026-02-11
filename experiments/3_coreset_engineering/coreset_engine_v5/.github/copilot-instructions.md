<!-- GitHub Copilot instructions for coding agents in this repo -->
# Coreset Engine — Copilot Instructions

Purpose: help AI coding agents be immediately productive in this repo by highlighting the architecture, workflows, conventions, and concrete code examples.

- **Big picture**: this project is a deterministic, curriculum-driven pipeline that compresses large token pools into stage-specific coresets.
  - Pipeline stages: Data loading → Dedup (exact/near) → Curriculum validation → Diversity scoring → Stratified selection → Protected slice enforcement → Output/Manifests
  - Key entrypoint: `coreset_builder.py`

- **Core components (read these first)**:
  - `src/core/config.py` — pipeline & stage config, `PipelineConfig.compute_hash()` for reproducibility
  - `src/curriculum/loader.py` — frozen curriculum semantics (do not edit frozen curricula)
  - `src/dedup/deduplicator.py` — exact (xxhash) and near (simhash/minhash) dedup logic
  - `src/diversity/scorer.py` — token frequency analyzer and composite scoring
  - `src/selection/engine.py` — orchestration: buckets, scoring, language policy, rolling-window, protected slices
  - `src/io/loaders.py` — `ChunkLoader` and `CoresetWriter` (parquet/jsonl outputs)

- **Important repository conventions & patterns**:
  - Determinism is enforced via `PipelineConfig.curriculum.deterministic_seed`; tests and runs assume fixed seeds.
  - Curriculum is authoritative and often marked FROZEN — changing it breaks reproducibility.
  - Stage names are canonical strings: `1B`, `3B`, `8B`, `70B`, `SFT`, `ALIGNMENT`.
  - Selection enforces curriculum distribution (band ratios) over the raw data distribution — code intentionally distributes band targets across allowed domains equally.
  - Protected slices (B4/B5, code, agentic, indic) are restored only up to curriculum targets — implemented in `SelectionEngine._enforce_protected_slices`.

- **Developer workflows / common commands**:
  - Run full pipeline (default config):
    - `python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml`
  - Run specific stages: `--stages 1B 3B 8B 70B`
  - Ablation example: `--config config/ablation_no_neardup.yaml`
  - Virtualenv: use Python 3.10+ and `pip install -r requirements.txt` before running.
  - Logs: pipeline uses `coreset_selection.log` and per-run manifests in `output/coresets/<stage>/manifest.json`.

- **Testing & debugging**:
  - Unit tests live in `tests/`; run `pytest -q` from repo root.
  - Common debugging pattern: run a single stage locally with a smaller dataset (set `io.num_parallel_loaders` lower and `io.input_dataset_path` to a sample folder).

- **Integration and deployment notes**:
  - I/O supports object stores (S3/GCS) via `IOConfig.use_object_store` and `object_store_type`; `boto3` required for S3.
  - Output formats: parquet (preferred) and JSONL/CSV are supported by `CoresetWriter`.

- **Concrete code examples to reference**:
  - Orchestration + CLI: `coreset_builder.py` — shows `CoresetBuilder.build_coresets()` usage and manifest creation.
  - Selection flow: `SelectionEngine.select_for_stage()` → `_create_buckets()` → `_score_chunks_in_bucket()` → `_stratified_sample_from_bucket()`.
  - Dedup: `ExactDeduplicator.compute_hash()` and `NearDeduplicator.compute_signature()` live in `src/dedup/deduplicator.py`.

- **When editing code, pay attention to**:
  - Preserving deterministic behavior: do not change or remove seed usage without updating manifest/versioning.
  - Curriculum constraints: changes that affect band ratios, allowed domains, or language_policy need curriculum-team coordination.
  - Output interface: keep `CoresetWriter.save_selected_indices()` signatures stable (used downstream).

If anything here is unclear or you want more detail (examples, lines to inspect, or tests to run), say which area and I will expand.
