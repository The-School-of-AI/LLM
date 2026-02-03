# ERA4 Lightning LLM Capstone

This repository contains the multi-team LLM training project used for the ERA4 Lightning Capstone.

Quick entry points:

- **Module 3 (Coreset Engineering)**: experiments/3_coreset_engineering/ — core pipeline, configs, tests, and reproducibility artifacts.
- **Reproducibility & Policies**: experiments/3_coreset_engineering/REPRODUCIBILITY_POLICY.md and IMPLEMENTATION_GUIDE.md for deterministic execution guidance.
- **Tests**: experiments/3_coreset_engineering/tests/ — consolidated module tests and run instructions.

Contribution & workflow:

- Use `uv` for environment and reproducible runs: `uv sync`, `uv run <cmd>`.
- Run module tests from the project root:

```powershell
uv run pytest experiments/3_coreset_engineering/tests -v
```

For module-specific documentation and implementation details see [experiments/3_coreset_engineering/README.md](experiments/3_coreset_engineering/README.md).