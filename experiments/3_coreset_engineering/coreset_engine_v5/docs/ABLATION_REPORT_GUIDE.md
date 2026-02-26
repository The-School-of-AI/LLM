# Ablation & Validation Report Generation

## Overview

The coreset engine now generates comprehensive **ablation and validation reports** that document:

✅ **Achieved Reduction Ratios**
- Overall compression metrics (tokens, chunks)
- Per-stage breakdown
- Speedup estimates for training

✅ **Methods Evaluated**  
- Core selection strategy components (dedup, diversity, stratification)
- Ablation variants (baseline, no-dedup, no-diversity, high-compression)
- Expected impact of each method

✅ **Coverage Diagnostics**
- Difficulty band coverage (B0-B5)
- Domain distribution (code, math, reasoning, agentic, indic, clean_web)
- Language coverage
- Curriculum adherence metrics

✅ **Proxy Training Comparisons**
- Coreset vs full dataset efficiency gains
- Estimated training time reduction
- Compute cost savings
- Quality retention estimates

## Automatic Report Generation

Reports are generated automatically at the end of the pipeline run.

### Output Location

```
output/
├── manifests/
│   └── ablation_validation_report.md  ← Comprehensive report
└── coresets/
    ├── 1B/
    │   ├── manifest.json
    │   └── selected_indices.jsonl
    ├── 3B/
    │   ├── manifest.json
    │   └── selected_indices.parquet
    └── ...
```

### Report File

**File:** `output/manifests/ablation_validation_report.md`

**Format:** Markdown with tables and metrics

**Size:** ~20-50 KB per run

### How Reports Are Generated

1. **During Pipeline Execution**
   ```python
   builder = CoresetBuilder(config, curriculum)
   results = builder.build_coresets()  # Collects stats for each stage
   builder.generate_reports(results)   # Generates comprehensive report
   ```

2. **Configuration (pipeline.yaml)**
   ```yaml
   reproducibility:
     emit_reproducibility_manifest: true  # Enable report generation
   
   ablation:
     enable_ablation_mode: false
     track_metrics:
       - compression_ratio
       - band_coverage
       - domain_coverage
       - protected_preservation
   ```

3. **Automatic at Pipeline End**
   - `coreset_builder.py::build_all_coresets()` calls `generate_reports()`
   - No manual action required
   - Reports appear in `output/manifests/`

## Report Contents

### 1. Executive Summary
Overview of what the report documents and key findings.

### 2. Overall Reduction Metrics
```
| Metric | Value | Reduction |
|--------|-------|-----------|
| Single-pass Corpus Tokens | 2,000,000,000 | - |
| Cumulative Stage Exposure Tokens | 6,000,000,000 | - |
| Selected Tokens (sum across stages) | 100,000,000 | 95.0% (vs single-pass) |
| Compression Ratio (single-pass basis) | 20.0x | 95.0% |
| Compression Ratio (stage-exposure basis) | 60.0x | 98.3% |
| Total Input Chunks | 1,000,000 | - |
| Selected Chunks | 50,000 | 95.0% |
| Chunk Reduction | 20.0x | 95.0% |
```

### 3. Stage-wise Breakdown

For each curriculum stage (1B, 3B, 8B, etc.):

**Selection Metrics:**
- Input tokens, selected tokens, compression ratio
- Chunk counts

**Band Distribution (Difficulty Mix):**
```
| Band | Ratio | Tokens | Coverage |
|------|-------|--------|----------|
| B0 | 10% | 10,000,000 | ✓ |
| B1 | 15% | 15,000,000 | ✓ |
| B2 | 20% | 20,000,000 | ✓ |
| ... | ... | ... | ✓ |
```

**Domain Distribution (Content Diversity):**
```
| Domain | Ratio | Tokens |
|--------|-------|--------|
| code | 20% | 20,000,000 |
| math | 15% | 15,000,000 |
| reasoning | 20% | 20,000,000 |
| ... | ... | ... |
```

**Language Distribution (Linguistic Coverage):**
```
| Language | Ratio | Tokens |
|----------|-------|--------|
| en | 92% | 92,000,000 |
| hi | 8% | 8,000,000 |
| ... | ... | ... |
```

### 4. Coverage Diagnostics

**Curriculum Adherence:**
- Ensures learning progression across difficulty bands
- Provides diverse content coverage
- Covers target languages with specified ratios

**Coverage Achievement:**
- Number of difficulty bands covered (e.g., 6/6)
- Number of domains covered (e.g., 6 domains)
- Number of languages covered (e.g., 2 languages)

### 5. Methods Evaluated

**Core Selection Strategy Components:**

1. **Deduplication**
   - Exact deduplication: Removes byte-identical chunks
   - Near-deduplication: Filters similar chunks (SimHash threshold)
   - Impact quantified

2. **Diversity Scoring**
   - Token frequency analysis for rare/tail token prioritization
   - Rare token boost: 1.5x weight
   - Tail token boost: 2.0x weight
   - Domain and language diversity weights

3. **Stratified Curriculum Sampling**
   - Band distribution enforcement
   - Domain preservation
   - Language coverage
   - Protected slice enforcement (B4, B5, code, agentic, indic)

4. **Non-overlap Enforcement**
   - Ensures no chunk appears in multiple stages
   - Prevents data leakage

**Ablation Variants:**

| Variant | Key Changes | Expected Impact |
|---------|-------------|-----------------|
| Baseline | Full pipeline | Balanced selection |
| No Near-Dedup | Dedup disabled | Higher redundancy, larger size |
| No Diversity | Uniform sampling | Less rare/tail coverage |
| High Compression | Aggressive sampling | Smaller coreset, potential quality loss |

### 6. Proxy Training Comparisons

**Estimated Training Efficiency Gains:**

```
| Metric | Full Dataset | Coreset | Improvement |
|--------|-------------|---------|------------|
| Tokens Processed | 2,000,000,000 | 100,000,000 | 20.0x faster |
| Training Time (est.) | ~2.0B tokens | ~0.1B tokens | 95.0% reduction |
| Compute Cost (est.) | 100% | 5.0% | 95.0% savings |
| Convergence Speed | Baseline | ~20.0x faster | Expected 20.0x speedup |
```

**Expected Quality Trade-offs:**
- Training time reduction: **95.0%**
- Compute cost reduction: **~95.0%**
- Estimated quality retention: **85-95%**
- Quality loss (estimated): **5-15%**

**Effectiveness Metrics:**
- Coverage score based on domain coverage
- Difficulty balance verification
- Linguistic diversity metrics

### 7. Deduplication Impact

- Chunks removed by deduplication: X chunks (Y%)
- Redundancy elimination impact on data quality
- Trade-off analysis: compression vs quality

### 8. Recommendations

**For Production Deployment:**
- Recommended compression ratio
- Expected training time reduction
- Coverage targets verification

**For Maximum Compression:**
- Suggest high-compression variant
- Trade-off analysis
- Quality assurance guidance

**For Quality Assurance:**
- Validation procedures
- Comparison methodology
- Adjustment recommendations

## Using the Reports

### 1. Read the Report

```bash
# On Linux/Mac
cat output/manifests/ablation_validation_report.md

# On Windows PowerShell
Get-Content output/manifests/ablation_validation_report.md | less
```

### 2. Parse Programmatically

```python
from pathlib import Path

report_path = Path("output/manifests/ablation_validation_report.md")
report_text = report_path.read_text()

# Extract compression ratios
import re
def _extract_ratio(label: str) -> float | None:
    # Matches markdown table rows like:
    # | **Compression Ratio (single-pass basis)** | **1.68x** | **40.4%** |
    pattern = rf"Compression Ratio \\({re.escape(label)}\\).*?(\\d+\\.\\d+)x"
    m = re.search(pattern, report_text)
    return float(m.group(1)) if m else None

single_pass = _extract_ratio("single-pass basis")
stage_exposure = _extract_ratio("stage-exposure basis")
print(f"Single-pass compression: {single_pass:.2f}x" if single_pass else "Single-pass compression: N/A")
print(f"Stage-exposure compression: {stage_exposure:.2f}x" if stage_exposure else "Stage-exposure compression: N/A")
```

### 3. Compare Across Runs

```bash
# Generate reports for multiple config variants
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml
cp output/manifests/ablation_validation_report.md reports/baseline_report.md

python coreset_builder.py --config config/ablation_high_compression.yaml --curriculum config/curriculum.yaml
cp output/manifests/ablation_validation_report.md reports/compression_report.md

# Compare reports side-by-side
diff reports/baseline_report.md reports/compression_report.md
```

## Report Customization

### Add Custom Metrics

Extend `AblationReporter.generate_report()` to include additional metrics:

```python
from src.io.loaders import AblationReporter

class CustomAblationReporter(AblationReporter):
    @staticmethod
    def generate_report(stages_results, output_path):
        # Call parent to get base report
        report_path = AblationReporter.generate_report(stages_results, output_path)
        
        # Add custom metrics
        report_file = Path(report_path)
        report_text = report_file.read_text()
        
        # Append custom section
        custom_section = """
## Custom Metrics

Your custom analysis here...
"""
        report_text += custom_section
        report_file.write_text(report_text)
        
        return str(report_path)
```

### Change Report Format

Generate reports in alternative formats:

```python
def generate_json_report(stages_results, output_path):
    """Generate report as JSON instead of Markdown"""
    import json
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'compression_ratio': ...,
        'stages': {
            stage: {
                'tokens': results['selected_tokens'],
                'chunks': results['selected_chunks'],
                'distributions': ...
            }
            for stage, results in stages_results.items()
        }
    }
    
    output_file = Path(output_path) / "report.json"
    output_file.write_text(json.dumps(report_data, indent=2))
    return str(output_file)
```

## Report Example

Here's an example section from a generated report:

```markdown
# Coreset Selection Ablation & Validation Report

## Executive Summary

This report documents comprehensive coreset selection results including:
- Reduction ratios achieved across all curriculum stages
- Coverage diagnostics and quality metrics
- Ablation study comparing different selection strategies
- Proxy training comparisons (coreset vs full dataset baseline)

## Overall Reduction Metrics

| Metric | Value | Reduction |
|--------|-------|-----------|
| Total Input Tokens | 2,000,000,000 | - |
| Selected Tokens | 100,000,000 | 95.0% |
| **Compression Ratio** | **20.0x** | **95.0%** |
| Total Input Chunks | 1,000,000 | - |
| Selected Chunks | 50,000 | 95.0% |
| **Chunk Reduction** | **20.0x** | **95.0%** |

## Stage-wise Breakdown

### 1B

**Selection Metrics:**
- Input Tokens: 500,000,000
- Selected Tokens: 25,000,000
- Compression Ratio: **20.0x** (reduction: 95.0%)
- Selected Chunks: 12,500

...rest of report...
```

## Troubleshooting

### Report Not Generated

**Issue:** `ablation_validation_report.md` file doesn't appear

**Solutions:**
1. Check `reproducibility.emit_reproducibility_manifest` is `true`
2. Verify `output_manifest_path` is writable
3. Check logs for generation errors

### Incorrect Metrics

**Issue:** Report shows zero or NaN values

**Solutions:**
1. Ensure chunks are loaded successfully
2. Verify curriculum stages are configured
3. Check that selection was executed (not skipped)

### Report Generation Errors

**Issue:** Exception during report generation

**Solutions:**
1. Check Python version (3.8+)
2. Verify pandas is installed for DataFrame operations
3. Check disk space for output directory

## Performance Notes

- Report generation: ~1-5 seconds for typical run
- Report file size: ~20-50 KB (markdown)
- Minimal overhead: <1% of total pipeline time
- Can be disabled if needed (though not recommended)

## Next Steps

After reviewing the report:

1. **Validate Results:** Compare expected vs actual compression
2. **Quality Check:** Run proxy training comparison
3. **Adjust Parameters:** Modify ablation settings if needed
4. **Archive Reports:** Store reports in version control
5. **Monitor Trends:** Track compression/quality across runs

See [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for integration examples and [QUICKSTART.md](QUICKSTART.md) for pipeline usage.
