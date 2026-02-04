# Project Structure

Complete folder structure for the Coreset Engineering project (Team 3).

```
8_moe_architecture/
│
├── .github/                          # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml                   # Continuous integration (test, lint, typecheck)
│       ├── validation.yml           # Coreset validation workflow
│       └── release.yml              # Release packaging workflow
│
├── config/                          # Stage-specific configurations
│   ├── stage_1b.yaml               # 1B model stage (20B tokens)
│   ├── stage_3b.yaml               # 3B model stage (40B tokens)
│   ├── stage_8b.yaml               # 8B model stage (100B tokens)
│   └── stage_moe.yaml              # MoE stage (240B tokens)
│
├── data/                            # Raw input data (gitignored)
│   └── (raw corpus and metadata from Team 1)
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md             # System architecture and data flow
│   └── CURRICULUM.md               # Curriculum structure and bands
│
├── logs/                            # Log files (gitignored)
│
├── notebooks/                       # Analysis notebooks
│   └── analysis_template.ipynb     # Template for coreset analysis
│
├── outputs/                         # Generated outputs (gitignored)
│   ├── coresets/                   # Final coreset files
│   ├── manifests/                  # Stage manifests (JSON)
│   └── validation_reports/         # Validation reports
│
├── scripts/                         # Execution scripts
│   ├── run_pipeline.sh             # Main pipeline execution
│   └── validate_outputs.sh         # Validation script
│
├── src/                             # Source code
│   ├── coreset_builder/            # Core pipeline
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point
│   │   └── pipeline.py             # Pipeline orchestration
│   │
│   ├── deduplication/              # Deduplication methods
│   │   ├── __init__.py
│   │   ├── exact_dedup.py          # Exact hash-based dedup
│   │   └── near_dedup.py           # MinHash/SimHash near-dedup
│   │
│   ├── selection/                   # Selection strategies
│   │   ├── __init__.py
│   │   └── stratified_sampling.py  # Curriculum-aware sampling
│   │
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   └── manifest.py             # Manifest generation
│   │
│   └── validation/                  # Validation and metrics
│       ├── __init__.py
│       ├── curriculum_validator.py # Curriculum checks
│       └── validate.py             # Main validation script
│
├── tests/                           # Test suite
│   ├── test_exact_dedup.py        # Deduplication tests
│   └── test_pipeline.py           # Pipeline tests
│
├── .gitignore                      # Git ignore patterns
├── Makefile                        # Common tasks automation
├── PROJECT_STRUCTURE.md           # This file
├── pyproject.toml                 # Python project config
├── README.md                       # Project overview
├── requirements.txt               # Python dependencies
├── setup.py                        # Package setup
└── task.md                         # Original task description
```

## Key Components

### Configuration (`config/`)
Stage-specific YAML files defining:
- Target token counts (20B/40B/100B/240B)
- Curriculum ratios (bands B0-B5, domains)
- Protected slice minimums
- Deduplication settings
- Selection strategies

### Source Code (`src/`)

#### `coreset_builder/`
- Pipeline orchestration
- Main entry point for coreset generation
- Coordinates dedup → selection → validation

#### `deduplication/`
- **Exact**: xxhash-based exact duplicate removal
- **Near**: MinHash LSH for similarity-based removal

#### `selection/`
- Stratified sampling across curriculum bands and domains
- Protected slice enforcement
- Rolling-window smoothness constraints

#### `validation/`
- Curriculum ratio validation
- Smooth transition checks
- Protected slice verification

#### `utils/`
- Manifest generation with reproducibility guarantees
- Config hashing and versioning

### Scripts (`scripts/`)
- `run_pipeline.sh`: Executes all 4 stages sequentially
- `validate_outputs.sh`: Validates generated manifests

### GitHub Actions (`.github/workflows/`)

#### `ci.yml`
- Runs on push/PR to main/develop
- Tests across Python 3.9, 3.10, 3.11
- Linting (ruff), type checking (mypy), tests (pytest)
- Coverage reporting

#### `validation.yml`
- Manual workflow dispatch
- Validates specific manifest files
- Uploads validation reports as artifacts

#### `release.yml`
- Triggered on version tags (v*)
- Packages manifests and validation reports
- Creates GitHub releases

## Usage

### Setup
```bash
make install          # Install dependencies
make setup-dirs       # Create output directories
```

### Development
```bash
make test            # Run tests
make lint            # Check code style
make typecheck       # Run type checking
make format          # Format code
```

### Execution
```bash
make run-all         # Run full pipeline (all stages)
make validate        # Validate outputs
```

### Manual Execution
```bash
# Single stage
python -m src.coreset_builder.main --config config/stage_1b.yaml --seed 42

# Validation
python -m src.validation.validate --manifest outputs/manifests/stage_1b.json
```

## Outputs

### Manifests (`outputs/manifests/`)
JSON files containing:
- Selected indices
- Token counts and distributions
- Band/domain composition
- Config hash for reproducibility
- Seed and timestamp

### Validation Reports (`outputs/validation_reports/`)
- Curriculum adherence checks
- Protected slice verification
- Smoothness analysis
- Coverage diagnostics

## Data Flow

```
Raw Corpus (2T tokens, from Team 1)
    ↓
Config (curriculum ratios from Team 2)
    ↓
Chunk-level Processing
    ↓
Exact Deduplication (xxhash)
    ↓
Near Deduplication (MinHash LSH)
    ↓
Stratified Sampling (curriculum-aware)
    ↓
Validation (ratios, smoothness, protected slices)
    ↓
Stage Coresets
    ├── 1B:  20B tokens
    ├── 3B:  40B tokens
    ├── 8B:  100B tokens
    └── MoE: 240B tokens
    ↓
Manifests + Validation Reports
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- **numpy, torch, pandas**: Data processing
- **datasketch**: MinHash/SimHash
- **xxhash**: Fast hashing
- **datasets, tokenizers**: HuggingFace tools
- **pytest, ruff, mypy**: Development tools

## Reproducibility

All operations are:
- **Deterministic**: Seed-controlled random operations
- **Versioned**: Config hashes in manifests
- **Auditable**: Full index tracking and metadata
- **Documented**: Comprehensive logging and reports

## Timeline

- **Jan 29-30**: Setup & pipeline skeleton
- **Jan 30 - Feb 5**: Core execution (selection, validation, proxy training)
- **Feb 2-4**: Final validation & stress testing
- **Feb 9**: Freeze deadline
