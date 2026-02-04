# Quick Start Guide

## Initial Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Create output directories
make setup-dirs
```

## Running the Pipeline

### All Stages at Once
```bash
# Run all 4 stages (1B, 3B, 8B, MoE)
make run-all

# Or with custom settings:
SEED=123 OUTPUT_DIR=my_outputs bash scripts/run_pipeline.sh
```

### Single Stage
```bash
# Run specific stage
python -m src.coreset_builder.main \
    --config config/stage_1b.yaml \
    --seed 42 \
    --output-dir outputs
```

## Validation

```bash
# Validate all outputs
make validate

# Validate specific manifest
python -m src.validation.validate \
    --manifest outputs/manifests/stage_1b.json
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Type check
make typecheck

# Format code
make format

# Clean build artifacts
make clean
```

## Configuration

Edit stage configs in `config/` directory:
- `stage_1b.yaml` - 1B model (20B tokens)
- `stage_3b.yaml` - 3B model (40B tokens)
- `stage_8b.yaml` - 8B model (100B tokens)
- `stage_moe.yaml` - MoE model (240B tokens)

Key parameters to adjust:
- `target_tokens`: Total tokens for the stage
- `curriculum_ratios.bands`: B0-B5 ratios
- `curriculum_ratios.domains`: Domain ratios
- `protected_slices`: Minimum token counts for critical content
- `deduplication`: Dedup settings
- `selection`: Selection strategy parameters

## Common Tasks

### Add New Test
Create file in `tests/` directory:
```python
# tests/test_my_feature.py
import pytest
from src.my_module import MyClass

def test_my_feature():
    obj = MyClass()
    assert obj.method() == expected_result
```

### Add New Selection Strategy
1. Create file in `src/selection/`
2. Implement sampler class
3. Update pipeline to use new strategy
4. Add tests

### Analyze Results
```bash
# Start Jupyter
jupyter notebook notebooks/analysis_template.ipynb
```

## Troubleshooting

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Missing Dependencies
```bash
# Update requirements
pip install -r requirements.txt
```

### Permission Denied on Scripts
```bash
chmod +x scripts/*.sh
```

## Output Files

```
outputs/
├── manifests/           # JSON manifests with indices and metadata
├── coresets/           # Final coreset data files
├── validation_reports/ # Validation results
└── logs/              # Execution logs
```

## GitHub Actions

### CI (Automatic)
- Triggers on push/PR to main/develop
- Runs tests, linting, type checking

### Manual Validation
1. Go to Actions tab
2. Select "Coreset Validation"
3. Click "Run workflow"
4. Enter manifest path
5. Download validation report from artifacts

### Release
1. Create version tag: `git tag v1.0.0`
2. Push tag: `git push origin v1.0.0`
3. Release workflow packages and publishes artifacts

## Key Metrics to Monitor

- **Token counts**: Should match targets (20B/40B/100B/240B)
- **Curriculum ratios**: Within 5% of targets
- **Protected slices**: Meet minimum thresholds
- **Deduplication rate**: Track duplicates removed
- **Coverage**: All bands and domains represented

## Next Steps

1. Receive raw data from Team 1
2. Confirm curriculum specs with Team 2
3. Run initial pipeline on sample data
4. Validate and iterate
5. Full production run
6. Deliver to training team (Team 10)
