# How to Generate Reports

## Overview

The pipeline **automatically generates comprehensive ablation and validation reports** after building coresets. No additional configuration is needed.

## Automatic Report Generation

When you run the pipeline, reports are generated at the end:

```bash
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml
```

**Output Location:** `output/manifests/ablation_validation_report.md`

## What's in the Report

### 1. Overall Reduction Metrics
```
| Metric | Value | Reduction |
|--------|-------|----------|
| Single-pass Corpus Tokens | 360,000 | - |
| Cumulative Stage Exposure Tokens | 360,000 | - |
| Selected Tokens (sum across stages) | 160,000 | 55.6% (vs single-pass) |
| **Compression Ratio (single-pass basis)** | **2.25x** | **55.6%** |
| **Compression Ratio (stage-exposure basis)** | **2.25x** | **55.6%** |
```

### 2. Stage-wise Breakdown
- Per-stage compression ratios
- Difficulty band distribution (B0-B5)
- Domain distribution (code, math, reasoning, etc.)
- Language coverage

### 3. Coverage Diagnostics
- Curriculum adherence verification
- Coverage achievement metrics
- Band/domain/language coverage confirmation

### 4. Methods Evaluated
- **Core Strategy**: Deduplication, diversity scoring, stratified sampling
- **Ablation Variants**: Baseline, no-dedup, no-diversity, high-compression
- Impact analysis for each method

### 5. Proxy Training Comparisons
```
| Metric | Full Dataset | Coreset | Improvement |
|--------|-------------|---------|----------|
| Tokens Processed | 360,000 | 160,000 | 2.25x faster |
| Training Time (est.) | ~0.0B tokens | ~0.0B tokens | **55.6% reduction** |
```

### 6. Deduplication Impact
- Chunks removed by deduplication
- Redundancy elimination metrics

### 7. Recommendations
- Production deployment guidance
- Maximum compression trade-offs
- Quality assurance procedures

## Report Locations

All reports are saved to: `output/manifests/`

```
output/
├── coresets/
│   ├── 1B/
│   │   ├── selected_indices.{parquet|jsonl|csv}
│   │   └── manifest.json
│   ├── 3B/
│   │   ├── selected_indices.{parquet|jsonl|csv}
│   │   └── manifest.json
│   └── ...
└── manifests/
    └── ablation_validation_report.md  ← Main report
```

## View the Report

### On Windows
```powershell
Get-Content output/manifests/ablation_validation_report.md -Encoding UTF8
```

### On Linux/Mac
```bash
cat output/manifests/ablation_validation_report.md
```

### In VS Code
```bash
code output/manifests/ablation_validation_report.md
```

## Report Format

Reports are **Markdown** files with:
- Tables for metrics comparison
- Headers for section organization
- Checkmarks (✓) for coverage indicators
- Clear formatting for readability

## Custom Report Configuration

To customize report behavior, edit `config/pipeline.yaml`:

```yaml
# Reporting (future enhancement)
reporting:
  generate_ablation_report: true
  report_format: markdown  # or json, html
  include_charts: false
  include_recommendations: true
```

## Example: Running Full Pipeline with Reports

```bash
cd coreset_engine

# Step 1: Set up config
cp config/pipeline.yaml config/my_run.yaml
# Edit my_run.yaml as needed

# Step 2: Run pipeline
python coreset_builder.py \
  --config config/my_run.yaml \
  --curriculum config/curriculum.yaml

# Step 3: View report
Get-Content output/manifests/ablation_validation_report.md
```

## Report Contents Quick Reference

| Section | Purpose | Contains |
|---------|---------|----------|
| Executive Summary | Quick overview | All key metrics |
| Overall Metrics | Total compression | Tokens & chunks |
| Stage-wise | Per-curriculum metrics | 1B, 3B, 8B, etc. |
| Coverage Diagnostics | Quality validation | Band/domain/lang coverage |
| Methods Evaluated | Technical details | Strategy & ablations |
| Proxy Training | Efficiency estimates | Training time/cost savings |
| Deduplication Impact | Redundancy analysis | Chunks removed |
| Recommendations | Guidance | Production/quality tips |

## Troubleshooting

### Report Not Generated
- ✓ Ensure pipeline completes successfully (look for "Coreset selection pipeline completed successfully!")
- ✓ Check output/manifests/ directory exists
- ✓ Verify UTF-8 encoding support in your terminal

### Report Encoding Issues
Reports are now UTF-8 encoded to support special characters. Open with:
- VS Code (automatic)
- PowerShell: `Get-Content -Encoding UTF8`
- Python: `open(..., encoding='utf-8')`

### Report Empty or Partial
- Ensure results from all stages are being passed to `generate_reports()`
- Check logs for errors during report generation
- Verify output/manifests/ has write permissions

## Next Steps

After reviewing the report:
1. ✓ Validate coreset quality on test dataset
2. ✓ Compare model performance: coreset-trained vs full-dataset-trained
3. ✓ Adjust compression ratios based on quality metrics
4. ✓ Deploy to production if quality acceptable

## Advanced: Programmatic Report Generation

Generate reports outside the pipeline:

```python
from src.io.loaders import AblationReporter

# Build your results dict
results = {
    "1B": {
        "total_input_chunks": 100,
        "total_input_tokens": 120000,
        "selected_chunks": 50,
        "selected_tokens": 60000,
        "compression_ratio": 2.0,
        "band_distribution": {...},
        "domain_distribution": {...},
        # ... other metrics
    }
}

# Generate report
report_path = AblationReporter.generate_report(
    results, 
    output_path="output/manifests"
)

print(f"Report saved to: {report_path}")
```

## Questions?

- Check [ABLATION_REPORT_GUIDE.md](ABLATION_REPORT_GUIDE.md) for detailed documentation
- Review sample report in `output/manifests/ablation_validation_report.md`
- Examine the AblationReporter class in `src/io/loaders.py` for customization
