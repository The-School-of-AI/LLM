# REPORT_GENERATION_GUIDE

## Report Generation Flow

- Reports are generated automatically at the end of a successful coreset run.
- Default output path: `output/manifests/ablation_validation_report.md`.
- Generated report format is Markdown and includes reduction, coverage,
  methods, proxy-comparison, and recommendation sections.

### Minimal Run Command

```bash
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --input-path <input_path> \
  --input-format parquet
```

### Troubleshooting (Report Generation)

- Confirm pipeline completion without fatal errors.
- Confirm `output/manifests/` is writable.
- Confirm stage results are present before report generation.

## Inputs

- Clean, approved raw datasets and metadata from Team 1
- Curriculum definitions, ratios, and guardrails from Team 2
- Early benchmark targets and proxy evaluation criteria from benchmarking teams (as available)

## Tools

- Chunking and hashing utilities
- Exact and near-duplicate detection (hashing, MinHash / SimHash)
- Token-signature analysis (token histograms, compression proxies)
- Optional small embedding models for limited protected slices only
- FAISS / clustering (restricted to small subsets)
- Python (NumPy, PyTorch)
- Visualization notebooks for coverage and ablations

## Required Submissions (Brief)

### `coreset_builder.py`

Deterministic, configurable pipeline for coreset generation.

### Stage-wise index manifests

- selected indices
- token counts
- band/domain composition
- seeds and config hashes

### Ablation and validation report

- methods evaluated
- achieved reduction ratios
- coverage diagnostics
- proxy training comparisons (coreset vs full)

## Outputs (Charter)

- Four stage-specific coresets totaling ~400B tokens
- Reproducible index files and manifests
- Clear justification for:
  - selection strategy
  - protection rules
  - curriculum adherence
- Evidence that the coreset improves efficiency without degrading learning.

## Success Criteria

- Coresets are approved for training use
- Early training converges faster or equally fast versus full data
- Early benchmark deltas (MMLU, code, math, agentic, Indic) are not degraded
- No curriculum violations or domain spikes are observed
- Downstream teams can consume outputs without rework

## Failure Conditions

- Curriculum ratios are violated
- Sudden domain or difficulty spikes occur
- B4/B5 signal is diluted or lost
- Proxy runs show slower learning or degraded benchmarks
- Results are non-deterministic or irreproducible
