# GitHub Copilot Instructions for ERA4 Lightning LLM Capstone

## Project Overview

**ERA4 Lightning LLM** is a multi-phase LLM training project following a DeepSeek MoE architecture:

```
1B Dense (Seed) → 3B MoE → 8B Dense → 70B MoE (Flagship)
```

The project coordinates across **20 research teams** (numbered 1-20), each owning specific aspects of training pipeline.

**Key files**: [main.py](main.py), [pyproject.toml](pyproject.toml), [README.md](README.md)

---

## Critical Architecture: The 20-Team Model

### Team Ownership Map

| Phase | Teams | Responsibility |
|-------|-------|-----------------|
| **1_Data Radar** | Team 1 | Raw data acquisition, validation |
| **2_Curriculum** | Team 2 | Stage-specific curriculum design (defines 1B→70B progression) |
| **3_Coreset** | Team 3 | Data sampling & stratification (produces manifests) |
| **4_Synthetic** | Team 4 | Self-distillation, synthetic data generation |
| **5_QA/Leakage** | Team 5 | Deduplication, test set isolation |
| **6_Tokenizer** | Team 6 | BPE design, difficulty proxy via token count |
| **7_Token-MoE** | Team 7 | Token-expert interaction analysis |
| **8_MoE Arch** | Team 8 | Expert network design, router logic |
| **9_Training Ops** | Team 9 | Training loop, checkpointing, recovery |
| **10_SLM Training** | Team 10 | Small LM (1B/3B) training harness |
| **11_Weight Transfer** | Team 11 | Knowledge distillation, upcycling, weight initialization |
| **12_Training Ops** | Team 12 | Distributed training orchestration |
| **13_AWS Org** | Team 13 | AWS account structure, IAM, cost centers |
| **14_Cost Monitor** | Team 14 | Budget tracking, spend alerts |
| **15_GPU Util** | Team 15 | Resource scheduling, utilization optimization |
| **16_Early Warning** | Team 16 | Lightweight evaluation, failure prediction |
| **17_Benchmarks** | Team 17 | Final evaluation harness, safety rules |
| **18_Alignment** | Team 18 | SFT/RL alignment, final quality gates |
| **19_Reproducibility** | Team 19 | Experiment tracking, artifacts, versioning |
| **20_Research** | Team 20 | Narrative writing, findings synthesis |

**Critical interdependencies**:
- **Team 2 (Curriculum) → Team 3 (Coreset)**: Curriculum YAML drives sampling strategy
- **Team 3 (Coreset) → Team 10 (Training)**: Manifest indices consumed by trainer
- **Team 6 (Tokenizer) → Team 3**: Difficulty scoring via tokenizer proxy
- **Team 9/12 (Training Ops) → Team 11 (Weight Transfer)**: Knowledge distillation checkpoints

---

## Key Patterns & Conventions

### 1. Documentation-First Design

**Pattern**: Each module has `docs/{N}_team_name/README.md` documenting:
- Why this phase exists
- Inputs (what it consumes from prior teams)
- Outputs (what it produces for downstream teams)
- Interfaces (function signatures, data formats)

**Example**: [docs/8_moe_architecture/router_design.md](docs/8_moe_architecture/router_design.md) documents MoE router implementation with math formulas, code examples, and evolutionary roadmap.

**When contributing**: Update corresponding doc before or alongside code changes.

### 2. Experiment-Centric Development

**Pattern**: Each team maintains `experiments/{N}_team_name/` with:
- `src/`: Production library code
- `scripts/`: CLI entry points (often with argparse)
- `configs/`: YAML config files (curriculum.yaml is canonical)
- `tests/`: Unit tests (under `tests/{N}_team_name/` at project root)
- `README.md`: Local setup instructions

**Example**:
```
experiments/3_coreset_engineering/
├── src/coreset_engine/
│   ├── selection/curriculum.py      # Parses curriculum.yaml
│   ├── selection/sampler.py         # Stratified sampling logic
│   └── scoring/difficulty.py        # Band assignment B0-B5
├── scripts/
│   ├── coreset_builder.py           # Main entry point
│   └── generate_mock_data.py        # Test data generation
├── configs/curriculum.yaml          # Production schema
└── pyproject.toml                   # Team-specific deps
```

**When starting new work**: Copy the directory structure from an existing team as template.

### 3. Configuration as Contract

**Pattern**: Team outputs are driven by YAML configs, not code changes. This prevents team crosstalk.

**Example**: Team 3 (Coreset) reads `curriculum.yaml`:
```yaml
growth_schedule:
  stages:
    - name: "1B"
      curriculum_profile: "base"
      band_weights: {B0: 0.4, B1: 0.3, ...}

stage_profiles:
  base:
    target_tokens: 1_000_000_000
```

**Team 3 produces**: `manifests/{1B,3B,8B,70B}/manifest.json` with indices keyed to band/modality constraints.

**When modifying outputs**: Change config YAML, not code. Code is frozen relative to config.

### 4. Reproducibility Via Determinism

**Pattern**: All sampling, deduplication, and scoring use seeded randomness. Identical input → identical output.

**Code convention**:
```python
random.seed(42)  # Global seed for determinism
hashlib.md5(text).hexdigest()  # For dedup, use hashing not equality
```

**When debugging manifests**: Re-run with same seed to confirm determinism.

### 5. Curriculum Progression: Easy → Hard

**Pattern**: Stages progress from low-difficulty to high-difficulty. Each stage includes all previous stages.

```
1B stage: B0-heavy (easy), some B1
  ↓ (includes 1B data)
3B stage: B0-B3 mixed, harder shift
  ↓ (includes 1B+3B data)
8B stage: B1-B4, advanced curriculum
  ↓ (includes 1B+3B+8B data)
70B stage: B2-B5, full difficulty range
```

**Key file**: [docs/8_moe_architecture/router_design.md](docs/8_moe_architecture/router_design.md) shows context-aware routing (buckets by context length, adjusts temperature per difficulty band).

**When implementing sampler**: Verify subset property: `indices_1B ⊂ indices_3B ⊂ indices_8B ⊂ indices_70B`.

---

## Developer Workflows

### Building & Testing

**Project uses `uv` (Python package manager) for deterministic builds**:

```bash
# Install dependencies
uv sync

# Run tests from project root
uv run pytest tests/ -v

# Run team-specific tests
uv run pytest tests/3_coreset_engineering/ -v --cov=experiments/3_coreset_engineering/src

# Run single test
uv run pytest tests/3_coreset_engineering/test_builder_regression.py::TestCoresetBuilderRegressions::test_curriculum_parsing -v
```

**Key commands**:
- `uv sync`: Lock + install all dependencies (idempotent)
- `uv run <script>`: Execute Python in locked environment
- `uv add <package>`: Add dependency (updates lock file)

### Pre-commit Hooks

**Pattern**: `.pre-commit-config.yaml` enforces:
- JSON validation (all configs must parse)
- YAML validation
- Linting (ruff)
- Type checking (mypy)

```bash
# Run manually
pre-commit run --all-files

# Auto-run before commit
pre-commit install
```

### GitHub Workflows

**Pattern**: CI/CD pipeline auto-runs on PR/push:

1. **Tests**: All tests pass before merge (2 reviewers required)
2. **Build**: Docker image built for staging/prod deploy
3. **Deploy**: Auto-deploy to AWS on main/develop push

**View**: Repository → Actions → "Coreset Engineering - Build, Test & Deploy"

---

## Data Flow: End-to-End Example

### Team 3 (Coreset) Workflow

**Input**: Raw data + curriculum.yaml

```python
# 1. Load curriculum (Team 2 design)
config = CurriculumConfig.load("configs/curriculum.yaml")

# 2. Ingest raw data (from Team 1)
data = load_jsonl("s3://data/raw_input")

# 3. Deduplicate (Team 5 guidelines)
dedup = TextDeduplicator()
unique_data = dedup.deduplicate(data)

# 4. Score difficulty (Team 6 proxy: token count heuristic)
scorer = DifficultyScorer()
scored = scorer.score(unique_data)

# 5. Assign bands B0-B5
bucketer = DifficultyBucketer(config)
bucketed = bucketer.bucket(scored)

# 6. Stratified sample per stage (respecting band/modality weights)
sampler = StratifiedSampler(config)
manifests = {}
for stage in config.stages:
    stage_config = config.get_stage(stage.name)
    manifests[stage.name] = sampler.sample(bucketed, stage_config)

# 7. Write manifests to S3 (for Team 10 trainer)
for stage_name, manifest in manifests.items():
    write_s3(f"s3://manifests/{stage_name}/manifest.json", manifest)
```

**Key integration points**:
- **Team 2**: Defines `curriculum.yaml` schema
- **Team 6**: Provides difficulty scoring (initially heuristic, later via tokenizer)
- **Team 5**: Validates no test set leakage
- **Team 10**: Consumes manifests for training

---

## Common Pitfalls & Solutions

### Pitfall 1: Ignoring Curriculum Constraints

**Problem**: Hardcoding band weights instead of reading from curriculum.yaml

**Solution**: Always parameterize via YAML:
```python
stage_profile = config.get_stage_profile(stage.name)
band_weights = stage_profile['band_weights']  # ✓ Correct
```

### Pitfall 2: Non-deterministic Sampling

**Problem**: Different results on re-runs (breaks reproducibility)

**Solution**: Seed random + use deterministic structures:
```python
random.seed(42)  # Global seed
indices = sorted(set(indices))  # Deterministic order
```

### Pitfall 3: Breaking Subset Invariant

**Problem**: 3B stage missing samples from 1B stage

**Solution**: Build cumulatively:
```python
stage_1b = sample(all_data, size=1B_size)
stage_3b = union(stage_1b, sample_new(remaining_data, size=3B_size - 1B_size))
stage_8b = union(stage_3b, sample_new(...))
```

### Pitfall 4: Ignoring Cross-Team Contracts

**Problem**: Changing output format without notifying downstream team

**Solution**: Treat YAML configs and output manifests as APIs—version them:
```yaml
# curriculum.yaml
version: "0.2"  # Increment on breaking changes
```

---

## File Navigation Quick Reference

| Goal | File |
|------|------|
| Understand LLM architecture | [docs/8_moe_architecture/router_design.md](docs/8_moe_architecture/router_design.md) |
| Curriculum design | [experiments/3_coreset_engineering/configs/curriculum.yaml](experiments/3_coreset_engineering/configs/curriculum.yaml) |
| Coreset pipeline | [experiments/3_coreset_engineering/src/coreset_engine/](experiments/3_coreset_engineering/src/coreset_engine/) |
| Coreset tests | [tests/3_coreset_engineering/test_builder_regression.py](tests/3_coreset_engineering/test_builder_regression.py) |
| AWS deployment | [experiments/3_coreset_engineering/AWS_DEPLOYMENT.md](experiments/3_coreset_engineering/AWS_DEPLOYMENT.md) |
| GitHub CI/CD | [.github/workflows/coreset-deploy.yml](.github/workflows/coreset-deploy.yml) |
| Contribution rules | README.md → Contribution Guidelines |

---

## Quick Checklist for New Contributors

- [ ] Read relevant `docs/{N}_team_name/README.md` for your team
- [ ] Understand team inputs/outputs (which teams feed you data? whom do you feed?)
- [ ] Copy experiment directory structure: `src/`, `scripts/`, `configs/`, `tests/`
- [ ] Use `uv run pytest` for local testing (not bare `python`)
- [ ] Keep YAML configs versioned; treat as APIs
- [ ] Verify reproducibility: run pipeline twice with same seed
- [ ] Add tests before code (TDD-encouraged pattern)
- [ ] Update docs alongside code changes
- [ ] Run pre-commit hooks: `pre-commit run --all-files`
- [ ] Create PR with 2-reviewer requirement from CODEOWNERS

---

## Useful Commands Reference

```bash
# Setup
uv sync                              # Install all deps
cd experiments/3_coreset_engineering # Navigate to team
uv run scripts/coreset_builder.py \
  --config configs/curriculum.yaml \
  --data data/input \
  --output output/

# Testing
uv run pytest tests/3_coreset_engineering -v
uv run pytest tests/ --cov=src/coreset_engine
uv run pytest tests/ -m regression              # Regression tests only

# AWS (after setup)
aws ecs update-service --cluster coreset-staging --service coreset-builder --force-new-deployment
aws logs tail /ecs/coreset-staging --follow
aws s3 ls s3://llm-coreset-artifacts-*/manifests/

# Pre-commit
pre-commit run --all-files
pre-commit install  # Auto-run before commits
```

---

## Questions? Ask About...

- **Architecture**: "How does the router adjust for different context lengths?" → See router_design.md
- **Integration**: "Which teams depend on my output?" → Check CODEOWNERS or Team dependency docs
- **Testing**: "How do I verify band distribution correctness?" → See tests/3_coreset_engineering/TEST_GUIDE.md
- **AWS**: "How do I deploy my code?" → See AWS_DEPLOYMENT.md or CORESET_QUICKSTART.md
