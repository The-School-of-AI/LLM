# Metadata tagging system

> **Auto-discovering metadata tagging system for training datasets**

Add curriculum metadata tags to TB-scale training data with automatic metric discovery and plugin chaining.

## Setup

**First time only:**
```bash
cd experiments/2_curirculum_architects
uv pip install -e .
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Quick Start

```python
from curriculum_tags import CurriculumTagger

# That's it! Auto-loads metrics from metrics_config.yaml
tagger = CurriculumTagger("curriculum.yaml")

# Tag your data
sample = {"text": "Quantum mechanics...", "id": "123"}
tagged = tagger.tag_sample(sample)
print(tagged["curriculum_tags"])
# {
#   "difficulty": {"score": 0.89},
#   "modality": {"primary_modality": "general_text", ...},
#   "band_assignment": {"band": "B5", "reason": "Very high complexity text"},
#   "readability": {"flesch_kincaid_grade": 29.7, ...}
# }
```

## Features

✅ **Full auto-discovery** - Metrics auto-import from files AND auto-load from config  
✅ **Metric chaining** - Later metrics see results from earlier ones  
✅ **Curriculum-driven** - Metrics use curriculum.yaml for classification thresholds  
✅ **Scalable** - Efficiently process TBs of parquet data  
✅ **Extensible** - Drop in custom metrics without touching any existing code

## How It Works

**Imports work naturally** with editable install:

```python
# Just import directly!
from curriculum_tags import CurriculumTagger

# Metrics auto-import too! No need to edit __init__.py
from curriculum_tags import DifficultyMetric, YourCustomMetric

# Or import specific components
from curriculum_tags.metrics.difficulty import DifficultyMetric
from curriculum_tags.utils.curriculum_loader import CurriculumConfig
```

**One-time setup** (run from project directory):
```bash
cd experiments/2_curirculum_architects
uv pip install -e .
```

This installs the package in "editable" mode, so:
- ✅ Changes to code take effect immediately
- ✅ No PYTHONPATH needed
- ✅ Works like any Python package
- ✅ Not published - stays local to your project

**Two-level auto-discovery:**
1. **Import level**: Package scans `metrics/` directory and auto-imports all metric classes
2. **Runtime level**: `CurriculumTagger` reads `metrics_config.yaml` and loads enabled metrics
3. Convention: `DifficultyMetric` → `curriculum_tags/metrics/difficulty.py`
4. No manual registration or import editing needed!


## Contributing - Adding a New Metric

### Step 1: Create your metric class

`curriculum_tags/metrics/sentiment.py`:
```python
from curriculum_tags.core.plugin import MetricPlugin

class SentimentMetric(MetricPlugin):
    name = "sentiment"
    
    def compute(self, sample):
        # Your logic here
        return {"score": 0.75, "category": "positive"}
```

### Step 2: Add to metrics_config.yaml

```yaml
metrics:
  - name: sentiment
    class: SentimentMetric
    enabled: true
```

### Step 3: Done!

```python
# Your metric is automatically imported!
from curriculum_tags import SentimentMetric

tagger = CurriculumTagger("curriculum.yaml")
# Your metric is automatically loaded and used!
```

## Project Structure

```
experiments/2_curirculum_architects/
├── curriculum.yaml              # Curriculum policy (bands, modalities, etc)
├── metrics_config.yaml          # Metrics to load (auto-discovery)
├── curriculum_tags/             # Main package (no src/ wrapper!)
│   ├── __init__.py             # Package exports (enables imports)
│   ├── core/
│   │   ├── plugin.py           # MetricPlugin base class
│   │   └── tagger.py           # CurriculumTagger (auto-discovery)
│   ├── metrics/                # Built-in metrics
│   │   ├── README.md           # How to add new metrics
│   │   ├── difficulty.py
│   │   ├── modality.py
│   │   └── readability.py
│   └── utils/
│       └── curriculum_loader.py # CurriculumConfig (k1.k2.k3 access)
├── examples/
│   ├── basic_usage.py          # Simple auto-discovery example
│   └── custom_plugin.py        # Add custom metrics
└── tests/
    ├── test_difficulty_metric.py
    ├── test_modality_metric.py
    └── test_readability_metric.py
```

## Running Examples

All commands run from `experiments/2_curirculum_architects`:

```bash
cd experiments/2_curirculum_architects

# One-time setup: Install in editable mode
uv pip install -e .

# Run examples (no PYTHONPATH needed!)
uv run python examples/basic_usage.py
uv run python examples/custom_plugin.py

# Process parquet files
uv run python examples/parquet_processing.py

# Run tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_difficulty_metric.py -v

# Format code (respects parent .pre-commit-config.yaml)
uv run black curriculum_tags/ tests/ examples/
uv run isort curriculum_tags/ tests/ examples/
uv run ruff check --fix curriculum_tags/ tests/ examples/
```

### Processing Parquet Files

```python
from curriculum_tags import CurriculumTagger

tagger = CurriculumTagger("curriculum.yaml")

# Process entire parquet file
stats = tagger.process_parquet(
    input_path="data/train.parquet",
    output_path="data/train_tagged.parquet",
    batch_size=10000
)

print(f"Processed {stats['total_rows']} rows")
```

### Batch Processing

```python
samples = [
    {"id": "1", "text": "Sample 1"},
    {"id": "2", "text": "Sample 2"},
]

tagged_samples = tagger.process_batch(samples)
```

### Calculating Band Proportions

The `calculate_proportions.py` script calculates optimal curriculum band distributions for different model sizes based on the "Capacity-Aware Curriculum" logic.

**Features:**
- Samples metadata from processed parquet files (default 0.5% sample)
- Aligns difficulty distribution to model capacity (1B, 3B, 8B, 70B stages)
- Enforces curriculum floors defined in `curriculum.yaml`
- Outputs results for all stages

**Usage:**

```bash
# Calculate and print to console
uv run python scripts/calculate_proportions.py data/train.metadata.parquet

# Specify custom sampling rate (e.g. 10%)
uv run python scripts/calculate_proportions.py data/train.metadata.parquet --sampling-rate 0.1

# Save output to JSON
uv run python scripts/calculate_proportions.py data/train.metadata.parquet --output-json proportions.json
```

**Output Example:**
```text
Stage: 8B (Params: 8.0B, Cap: 0.4895)
----------------------------------------
  B0: 0.1872
  B1: 0.2107
  B2: 0.2709
...
```

## Configuration Files

### metrics_config.yaml
Controls which metrics run and in what order:
```yaml
metrics:
  - name: difficulty
    class: DifficultyMetric
    enabled: true
  - name: modality
    class: ModalityMetric
    enabled: true
  - name: band_assignment
    class: BandAssignmentMetric
    enabled: true
```

### curriculum.yaml
Policy and thresholds for classification:
```yaml
difficulty_system:
  bands:
    B0: {name: "Nursery", ...}
    B1: {name: "Primary", ...}
```

Metrics access these values:
```python
self.config.get("difficulty_system.bands.B0.name")  # "Nursery"
```

## Architecture

**Simple, no magic:**

1. **MetricPlugin** - Base class with `compute(sample) -> dict`
2. **CurriculumTagger** - Loads metrics from config, runs them in order
3. **Metric chaining** - Each metric sees `sample["curriculum_tags"]` from previous metrics
4. **Auto-discovery** - `importlib` dynamically loads classes from config
## Testing

All 49 tests pass:
```bash
uv run pytest tests/ -v
```

Organized by metric to avoid merge conflicts:
- `test_difficulty_metric.py` - Difficulty classification
- `test_modality_metric.py` - Modality detection  
- `test_readability_metric.py` - Readability scores
- `test_tagger.py` - Auto-discovery and chaining

## Contributing

1. Create metric in `src/curriculum_tags/metrics/`
2. Add to `metrics_config.yaml`
3. Add tests in `tests/test_yourmetric.py`
4. Format: `uv run black src/ tests/`
5. Submit PR

**Naming convention:**
- File: `sentiment.py` (lowercase, no "metric")
- Class: `SentimentMetric` (PascalCase with "Metric")
- Config: `class: SentimentMetric`

Auto-discovery maps: `SentimentMetric` → `sentiment.py`

---

**Built with simplicity in mind. No registries, no decorators - just clean Python.**

### Best Practices

- **Keep it stateless**: Plugins should be stateless for parallel processing
- **Handle edge cases**: Empty text, malformed data, etc.
- **Return consistent schema**: Always return same keys
- **Use curriculum config**: Leverage YAML config for thresholds/parameters
- **Add tests**: Write unit tests for your plugin

## Performance Considerations

For large-scale datasets (TBs):

- **Use batch processing**: Default `batch_size=10000` is a good start
- **Parallel processing**: Process multiple files in parallel
- **Filter plugins**: Only load metrics you need
- **Monitor memory**: Adjust batch size based on available RAM
- **Stream processing**: Parquet format supports efficient streaming

## Example for parallel processing:

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def process_file(file_path):
    tagger = CurriculumTagger("curriculum.yaml")
    output = file_path.with_suffix(".tagged.parquet")
    return tagger.process_parquet(file_path, output)

files = list(Path("data/").glob("*.parquet"))

with ProcessPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_file, files)
```

> Team 2 - Curriculum Architects
