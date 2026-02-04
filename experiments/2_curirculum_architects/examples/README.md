# Curriculum Architects - Examples

This directory contains comprehensive examples demonstrating how to use the curriculum extraction and analysis pipeline.

## 📁 Output Directory

All examples write output to `downloads/<example_name>/` in the examples directory (not temporary directories). This makes it easy to inspect results after running examples.

## 🚀 Quick Start

Make sure the package is installed:

```bash
# From the curriculum_tags directory
uv pip install -e .
```

Then run any example:

```bash
cd experiments/2_curirculum_architects
uv run python examples/01_basic_extraction.py
```

## 📚 Available Examples

### 1. Basic Extraction (`01_basic_extraction.py`)

**What it demonstrates:**
- Initializing the CurriculumExtractor
- Processing individual records (read-only)
- Handling rejections
- Viewing extracted metadata
- Timing statistics

**Usage:**
```bash
uv run python examples/01_basic_extraction.py
```

**Output:** Console only (no files created)

**Key concepts:**
- Records are NEVER modified (read-only processing)
- Each metric plugin can extract features or reject records
- Timing can be tracked per metric

---

### 2. Parquet Processing (`02_parquet_processing.py`)

**What it demonstrates:**
- Processing parquet files with metadata output
- Incremental processing with StateManager
- Resuming failed/interrupted jobs
- Rejection layer output

**Usage:**
```bash
# Use sample data (default)
uv run python examples/02_parquet_processing.py

# Use your own parquet file
uv run python examples/02_parquet_processing.py --parquet /path/to/your/file.parquet

# Specify custom output directory
uv run python examples/02_parquet_processing.py --output ./my_output
```

**Arguments:**
- `--parquet`: Path to input parquet file (optional, creates sample if not provided)
- `--output`: Output directory (default: `examples/downloads/02_parquet_processing/`)

**Output structure:**
```
downloads/02_parquet_processing/
├── data/
│   └── sample_data.parquet  (if created)
├── metadata/
│   └── file_name=sample_data/
│       └── *.parquet
├── rejections/
│   └── file_name=sample_data/
│       └── *.parquet
└── state/
    └── *.json
```

**Key concepts:**
- State management prevents reprocessing
- Metadata and rejections are partitioned by source file
- Can resume after interruptions

---

### 3. Custom Metrics (`03_custom_metrics.py`)

**What it demonstrates:**
- Creating custom MetricPlugin classes
- Implementing rejection logic
- Using read-only records
- Setting metric levels for execution order

**Usage:**
```bash
uv run python examples/03_custom_metrics.py
```

**Output:** Console only (no files created)

**Key concepts:**
- Level 0 metrics run first (fast filters)
- Higher level metrics run after filtering
- Plugins can reject records early to save computation

---

### 4. Band Assignment (`04_band_assignment.py`)

**What it demonstrates:**
- Running extraction WITHOUT band assignment
- Assigning bands after extraction by reading metadata layer
- Using different band configurations
- Updating metadata layer with band information

**Usage:**
```bash
# Use sample data (default)
uv run python examples/04_band_assignment.py

# Specify custom output directory
uv run python examples/04_band_assignment.py --output ./my_bands
```

**Arguments:**
- `--output`: Output directory (default: `examples/downloads/04_band_assignment/`)

**Output structure:**
```
downloads/04_band_assignment/
├── metadata/  (sample metadata created)
├── metadata_with_bands/  (default config)
└── metadata_custom_bands/  (weighted config)
```

**Key concepts:**
- Bands are assigned AFTER extraction
- Can use different scoring strategies
- Supports weighted combinations of metrics

---

### 5. Metadata Analysis (`05_metadata_analysis.py`)

**What it demonstrates:**
- MetadataReader for accessing metadata
- MetadataAnalyzer for statistics
- RejectionReader for quality analysis
- Various query patterns

**Usage:**
```bash
# Use sample data (default)
uv run python examples/05_metadata_analysis.py

# Use your own metadata and rejections
uv run python examples/05_metadata_analysis.py \
  --metadata /path/to/metadata \
  --rejections /path/to/rejections

# Specify custom output directory for sample data
uv run python examples/05_metadata_analysis.py --output ./my_analysis
```

**Arguments:**
- `--metadata`: Path to existing metadata directory (optional)
- `--rejections`: Path to existing rejections directory (optional)
- `--output`: Output directory for sample data (default: `examples/downloads/05_metadata_analysis/`)

**Output structure:**
```
downloads/05_metadata_analysis/
├── metadata/  (if sample created)
├── rejections/  (if sample created)
└── analysis_report.json
```

**Key concepts:**
- Metadata layer contains extracted features
- Rejection layer explains why records were filtered
- Analyzer provides dataset statistics

---

### 6. Batch Creation (`06_batch_creation.py`)

**What it demonstrates:**
- BatchCreator for reproducible training data loading
- Deterministic ordering with xxhash
- Auto-increment and seek operations
- Stratified sampling

**Usage:**
```bash
# Use sample data (default, 500 records)
uv run python examples/06_batch_creation.py

# Use your own metadata
uv run python examples/06_batch_creation.py --metadata /path/to/metadata

# Create sample with custom size
uv run python examples/06_batch_creation.py --num-records 1000

# Specify custom output directory
uv run python examples/06_batch_creation.py --output ./my_batches
```

**Arguments:**
- `--metadata`: Path to existing metadata directory (optional)
- `--output`: Output directory (default: `examples/downloads/06_batch_creation/`)
- `--num-records`: Number of records in sample data (default: 500)

**Output structure:**
```
downloads/06_batch_creation/
├── metadata/  (if sample created)
└── batch_state/  (batch state files)
```

**Key concepts:**
- Deterministic batching for reproducibility
- Same seed = same batch order
- Can resume from any batch number
- Stratified sampling maintains class proportions

---

### 7. Benchmarking (`07_benchmarking.py`)

**What it demonstrates:**
- Running benchmarks with timing per metric
- Memory tracking
- Interpreting benchmark results

**Usage:**
```bash
# Use sample data (default, 500 records)
uv run python examples/07_benchmarking.py

# Use your own parquet file
uv run python examples/07_benchmarking.py --parquet /path/to/your/file.parquet

# Custom parameters
uv run python examples/07_benchmarking.py \
  --num-records 1000 \
  --batch-size 200 \
  --output ./my_benchmark
```

**Arguments:**
- `--parquet`: Path to input parquet file (optional)
- `--output`: Output directory (default: `examples/downloads/07_benchmarking/`)
- `--num-records`: Number of records in sample data (default: 500)
- `--batch-size`: Batch size for processing (default: 100)

**Output structure:**
```
downloads/07_benchmarking/
└── benchmark_data.parquet  (if sample created)
```

**Key concepts:**
- Measures throughput (records/sec)
- Tracks per-metric timing
- Monitors memory usage
- Identifies bottlenecks

---

## 🔄 Complete Workflow Example

Here's how the examples fit together in a typical workflow:

```bash
# 1. Start with basic extraction to understand the pipeline
uv run python examples/01_basic_extraction.py

# 2. Process a real parquet file
uv run python examples/02_parquet_processing.py --parquet /data/my_corpus.parquet --output ./output

# 3. Assign bands to the extracted metadata
uv run python examples/04_band_assignment.py --output ./output

# 4. Analyze the results
uv run python examples/05_metadata_analysis.py \
  --metadata ./output/metadata_with_bands \
  --rejections ./output/rejections

# 5. Create training batches
uv run python examples/06_batch_creation.py \
  --metadata ./output/metadata_with_bands \
  --output ./training

# 6. Benchmark performance for optimization
uv run python examples/07_benchmarking.py --parquet /data/my_corpus.parquet
```

## 📊 Understanding Output Files

### Metadata Layer
- **Format:** Partitioned parquet files
- **Partitioning:** By source file (`file_name=...`)
- **Contents:** Extracted features (difficulty, readability, modality, etc.)
- **Schema:** Defined by curriculum.yaml

### Rejection Layer
- **Format:** Partitioned parquet files
- **Partitioning:** By source file (`file_name=...`)
- **Contents:** Rejected records with reason and metric name
- **Schema:** `[uuid, id, file_path, rejected_reason, rejected_at]`

### State Files
- **Format:** JSON
- **Purpose:** Track processed files to enable resumption
- **Location:** `state/` directory

## 🎯 Tips for Production Use

1. **Start with benchmarking** to understand performance on your data
2. **Use StateManager** for large-scale processing to enable resumption
3. **Monitor rejection rates** to tune metric thresholds
4. **Assign bands post-extraction** to easily experiment with different strategies
5. **Use deterministic batching** for reproducible training runs
6. **Partition metadata by source** for easier data management

## 🐛 Troubleshooting

**Example creates sample data but I want to use my own:**
- Use `--parquet` or `--metadata` arguments to provide your own files

**Output directory is full of test data:**
- Delete the `downloads/` directory or specify a different `--output` path

**State manager prevents reprocessing:**
- Delete files in the `state/` directory to reset

**Need more detailed output:**
- Check the console output for step-by-step progress
- Look at rejection files to understand why records were filtered

## 📖 Further Reading

- See the main README for architecture and design decisions
- Check `docs/` for detailed API documentation
- Review `curriculum.yaml` to understand available metrics
