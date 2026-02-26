# repro-registry

Authoritative reproducibility SDK owned by Team 19.

## Installation

pip install repro-registry

## Supported Pipelines

- Training
- Coresets

No other pipelines are supported.

## Training Usage

```python
from repro.registry import start_training_run, finalize_run
from repro.seeds import set_all_seeds

ctx = start_training_run(config, seed=42)

try:
    set_all_seeds(42)
    train(ctx)
    finalize_run(ctx, status="COMPLETED")
except Exception:
    finalize_run(ctx, status="FAILED")
    raise
