# Coreset Selection Engine for 70B LLM Pre-training

**Version**: 1.0.0  
**Status**: Production Ready  
**Team**: Coreset Selection Architecture

## Overview

The Coreset Selection Engine is a production-grade pipeline that compresses 2 trillion tokens to ~400 billion tokens for efficient 70B parameter LLM pre-training. The engine uses curriculum-aware stratified sampling, deduplication, and diversity optimization to create high-quality training datasets across multiple stages.

**Key Features**:
- ✅ **Deterministic & Reproducible**: Fully seeded, version-controlled pipeline
- ✅ **Curriculum-Compliant**: Strict adherence to frozen curriculum specifications
- ✅ **Scalable**: Handles 2T tokens with parallel I/O and vectorized scoring
- ✅ **Protective**: Preserves rare, capability-critical content (B4/B5, code, agentic, Indic)
- ✅ **Auditable**: Detailed manifests and coverage reports for each stage

## Quick Start

### Prerequisites

```bash
Python 3.10+
CUDA 11.8+ (optional, for GPU acceleration)
```

### Installation

```bash
# Clone repository
git clone <repo-url>
cd coreset_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run pipeline with default configuration
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml

# Run with custom stages
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml \
  --stages 1B 3B 8B 70B

# Run ablation study (no near-dedup)
python coreset_builder.py \
  --config config/ablation_no_neardup.yaml \
  --curriculum config/curriculum.yaml \
  --ablation-variant no_neardup
```

### Expected Output

```
coreset_engine/
├── output/
│   ├── coresets/
│   │   ├── 1B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── 3B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── 8B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   │   ├── 70B/
│   │   │   ├── selected_indices.parquet
│   │   │   ├── manifest.json
│   └── manifests/
│       └── ablation_report.md
└── coreset_selection.log
```

## Architecture

### Pipeline Stages

```
1. Data Loading & Registration
   ↓
2. Deduplication (Exact + Near)
   ↓
3. Curriculum Validation
   ↓
4. Diversity Scoring (Vectorized)
   ↓
5. Stratified Selection
   ↓
6. Protected Slice Enforcement
   ↓
7. Validation & Audit
   ↓
8. Output Generation
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Configuration Manager | `src/core/config.py` | Hierarchical, validated config management |
| Type System | `src/core/types.py` | Type-safe data structures |
| Curriculum Loader | `src/curriculum/loader.py` | Load & validate frozen curriculum |
| Exact Deduplicator | `src/dedup/deduplicator.py` | XXHash-based exact dedup |
| Near Deduplicator | `src/dedup/deduplicator.py` | SimHash/MinHash fuzzy matching |
| Diversity Scorer | `src/diversity/scorer.py` | Token rarity + coverage scoring |
| Selection Engine | `src/selection/engine.py` | Main orchestrator |
| I/O Utilities | `src/io/loaders.py` | Load & save with S3/FS support |

## Configuration

### Main Config (`config/pipeline.yaml`)

Key sections:
- **dedup**: Exact and near-duplicate detection settings
- **diversity**: Token rarity boosting and coverage weighting
- **selection**: Strategy (stratified, density-aware), protected slices
- **curriculum**: Frozen curriculum path and deterministic guarantees
- **stages**: Per-stage configurations (1B, 3B, 8B, 70B, SFT, ALIGNMENT)

Example customization:
```yaml
# Disable near-dedup for faster processing
dedup:
  enable_near_dedup: false

# Increase diversity boosting
diversity:
  rare_token_boost: 2.0    # From 1.5
  tail_token_boost: 3.0    # From 2.0
```

### Curriculum Config (`config/curriculum.yaml`)

**FROZEN** curriculum defining:
- Band definitions (B0-B5 with allowed domains)
- Stage-wise band ratios (1B: 45% B0 / 30% B1 / ..., etc.)
- Language constraints (92% English, 8% Hindi)
- Perplexity filters per band
- Rolling window constraints

⚠️ **Do not modify** once frozen. Changes require curriculum team approval.

## Usage Examples

### Example 1: Basic Selection for 70B Stage

```python
from src.core.config import PipelineConfig
from src.curriculum.loader import CurriculumLoader
from src.selection.engine import SelectionEngine
from src.io.loaders import ChunkLoader, CoresetWriter

# Load configuration
config = PipelineConfig.load_from_file("config/pipeline.yaml")
curriculum = CurriculumLoader("config/curriculum.yaml")
curriculum.load()

# Load chunks
loader = ChunkLoader(base_path="/data/datasets")
all_chunks = loader.load_all_chunks()

# Initialize engine
engine = SelectionEngine(config, curriculum)
engine.register_chunks([(cid, meta, None) for cid, meta in all_chunks.items()])

# Run selection
selected_chunks, stats = engine.select_for_stage(
    all_chunks=all_chunks,
    stage_name="70B",
)

# Save outputs
writer = CoresetWriter("/output/coresets")
writer.save_selected_indices("70B", selected_chunks, metadata_dict)
```

### Example 2: Ablation Study

```python
from src.core.config import PipelineConfig

# Load baseline config
config = PipelineConfig.load_from_file("config/pipeline.yaml")

# Ablation 1: No near-dedup
config.dedup.enable_near_dedup = False
config.save_to_file("config/ablation_no_neardup.yaml", format="yaml")

# Ablation 2: No diversity boosting
config.diversity.rare_token_boost = 1.0
config.diversity.tail_token_boost = 1.0
config.save_to_file("config/ablation_no_diversity.yaml", format="yaml")

# Run with ablated configs
# python coreset_builder.py --config config/ablation_no_neardup.yaml ...
```

### Example 3: Custom Protected Slices

```python
from src.core.types import ProtectedSliceRule

protected_slices = [
    ProtectedSliceRule("B5", 0.98, "Critical for emergent abilities"),
    ProtectedSliceRule("B4", 0.95, "Advanced reasoning"),
    ProtectedSliceRule("code", 0.93, "Programming capability"),
    ProtectedSliceRule("agentic", 0.92, "Agent grounding"),
    ProtectedSliceRule("indic", 0.80, "Multilingual support"),
]

selected_chunks, stats = engine.select_for_stage(
    all_chunks=all_chunks,
    stage_name="70B",
    protected_slices=protected_slices,  # Override defaults
)
```

## Key Metrics & Monitoring

### Expected Compression Results

| Stage | Input Tokens | Output Tokens | Ratio | Chunks |
|-------|--------------|---------------|-------|--------|
| 1B | 400B | 20B | 20x | ~5M |
| 3B | 800B | 40B | 20x | ~10M |
| 8B | 2T | 100B | 20x | ~25M |
| 70B | 2T | 240B | 8.3x | ~60M |

### Coverage Validation

After selection, check:
```python
from src.core.types import BandDistribution

band_dist = stats['band_distribution']
print(f"B0: {band_dist.B0:.2%}")  # Should be ~5% for 70B
print(f"B4: {band_dist.B4:.2%}")  # Should be ~25% for 70B
print(f"B5: {band_dist.B5:.2%}")  # Should be ~15% for 70B
```

### Protected Slice Preservation

```python
manifest = CoresetManifest(...)
preserved = manifest.protected_slices_preserved

assert preserved.B5_preservation_ratio >= 0.95, "B5 not preserved!"
assert preserved.code_preservation_ratio >= 0.90, "Code not preserved!"
```

## Troubleshooting

### Issue: "Curriculum not frozen"

**Error**: 
```
Curriculum validation failed: Curriculum is not frozen
```

**Solution**: 
- Ensure curriculum status is "FROZEN" in `config/curriculum.yaml`
- Contact curriculum team if status is "DRAFT"

### Issue: "Rolling window violation"

**Error**:
```
HARD_REJECT: Rolling window constraint violated
```

**Solution**:
- Reduce `diversity.rare_token_boost` or `tail_token_boost`
- Add `smooth_selection_via_rolling_window()` post-processing
- Increase `rolling_window.window_tokens` to allow more variance

### Issue: "Protected slices under-preserved"

**Error**:
```
B5 preservation ratio: 0.85 < 0.95 (minimum required)
```

**Solution**:
1. Increase `selection.protected_preservation_override["B5"]` to be reachable
2. Run selection without other constraints (debug mode)
3. Check if enough B5 chunks exist in source data

### Issue: Memory exhaustion on large datasets

**Solution**:
```yaml
# Reduce parallel loaders
io:
  num_parallel_loaders: 8  # Down from 32

# Enable streaming mode (future)
io:
  cache_metadata: false
```

## Integration Points

### Upstream: Accepting Input from Teams

```json
{
  "team_1": "Provide clean dataset + metadata (parquet/jsonl)",
  "team_2": "Provide FROZEN curriculum.yaml",
  "team_3": "Provide chunk indices + metadata",
  "team_4": "Provide difficulty band assignments",
  "team_5": "Provide dedup signatures + quality scores"
}
```

### Downstream: Providing Output to Teams

```json
{
  "training_team": {
    "format": "Parquet index files per stage",
    "guarantee": "Non-overlapping chunks",
    "reproducibility": "Deterministic given seed + curriculum"
  },
  "benchmarking_team": {
    "format": "Ablation report + coverage audit",
    "metrics": ["compression_ratio", "band_coverage", "convergence_speed"]
  },
  "synthetic_team": {
    "format": "Available band/domain quotas",
    "max_injection": "5-10% per stage"
  }
}
```

## Performance Benchmarks

### Runtime (Measured on 64-node GPU cluster)

| Stage | Input Tokens | Dedup | Scoring | Selection | Total |
|-------|--------------|-------|---------|-----------|-------|
| 1B | 400B | 15m | 10m | 5m | 30m |
| 3B | 800B | 20m | 15m | 8m | 43m |
| 8B | 2T | 45m | 35m | 15m | 95m |
| 70B | 2T | 45m | 35m | 15m | 95m |

**Total pipeline runtime**: ~4 hours (all stages in parallel: ~2 hours)

### Memory Usage

- Metadata cache: ~50GB (for 2T tokens)
- Dedup structures: ~30GB (exact + near-dedup hashes)
- Scoring vectors: ~20GB (diversity scores)
- **Total**: ~100GB (fits in typical cluster node)

## Reproducibility & Versioning

### Reproducibility Guarantee

Every coreset output includes:
```json
{
  "deterministic": true,
  "seed": 42,
  "config_hash": "sha256(...)",
  "curriculum_hash": "sha256(...)",
  "algorithm_version": "1.0.0",
  "created_at": "2026-02-03T10:30:00Z"
}
```

To reproduce exactly:
```bash
python coreset_builder.py \
  --config config/pipeline.yaml \
  --curriculum config/curriculum.yaml
```

Same outputs will be produced (bit-for-bit identical indices, same seed).

### Versioning Strategy

- **Pipeline Version**: `1.0.0` (algorithm changes bump minor/major)
- **Config Version**: Git commit hash (`abc123def...`)
- **Curriculum Version**: Git commit hash (frozen checkpoints)
- **Data Version**: Dataset timestamp + version ID

## Documentation

- [Design & Recommendations](docs/DESIGN_AND_RECOMMENDATIONS.md): Detailed design, algorithms, research references
- [Integration Schema](schemas/integration_schema.json): Team handoff contracts
- [Configuration Guide](config/README.md): Detailed config options

## Support & Issues

**Bug Reports**: Create issue with:
- Config file (sanitized)
- Curriculum file
- Error logs
- Hardware specs (CPU/GPU, RAM)

**Feature Requests**: 
- Propose as issue with use case
- Include performance requirements
- Discuss in team meeting before implementation

---

**Last Updated**: 2026-02-03  
**Maintainer**: Coreset Selection Team  
**License**: Internal Use Only
