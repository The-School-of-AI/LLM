# Experiment 17: Final Pretraining Benchmarks

This experiment contains benchmark validations and evaluations for pretrained language models.

## Structure

### [`leval/`](./leval/)
**L-Eval (Long Context Evaluation Suite) Benchmarking**

Complete implementation for validating OLMES CLI commands and verifying metrics using L-Eval benchmark suite.

- **Status**: ✅ Environment Setup Complete
- **Dependencies**: Installed via UV
- **Platform**: Optimized for M3 Mac with 18GB RAM  
- **Model**: Qwen/Qwen2.5-1.5B-Instruct (test model)

#### Quick Start
```bash
cd leval/
./setup.sh
source .venv/bin/activate
python run_validation.py
```

#### Features
- CLI validation (Python, PyTorch, Transformers, dependencies)
- Smoke testing with sample inferences  
- Metrics verification (Exact Match, ROUGE, Character Accuracy)
- Results export to JSON
- Apple Silicon MPS support

See [`leval/README.md`](./leval/README.md) for detailed documentation and [`leval/VALIDATION_RESULTS.md`](./leval/VALIDATION_RESULTS.md) for setup results.

## Future Benchmarks

This folder will contain additional benchmark implementations:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag
- TruthfulQA
- HumanEval (Code generation)
- And more...

## Overview

Final pretraining benchmarks validate model quality before fine-tuning and deployment. These benchmarks provide:

1. **Capability Assessment** - Test reasoning, knowledge, and generation quality
2. **Progress Tracking** - Monitor improvements across training checkpoints  
3. **Comparison Baselines** - Compare against public model benchmarks
4. **Early Warning** - Detect regressions or training issues

## Methodology

All benchmarks follow a consistent approach:

1. **Environment Setup** - UV-based dependency management
2. **CLI Validation** - Test all required tools and libraries
3. **Smoke Testing** - Quick sanity checks before full evaluation  
4. **Full Evaluation** - Run complete benchmark suite
5. **Metrics Verification** - Validate metric calculations
6. **Results Documentation** - Save and document all outputs

---

**Last Updated**: February 28, 2026  
**Status**: L-Eval validation complete, environment ready for benchmarking

### What is OLMES?
OLMES is the Open Language Model Evaluation Suite - a standardized evaluation framework for benchmarking language models. It provides:
- Structured evaluation protocols
- Consistent metric calculations
- CLI-based command execution
- Integration with various evaluation suites
- Reproducible benchmarking workflows

### What is L-Eval?
L-Eval (Long Context Evaluation Suite) is a specialized benchmark designed to evaluate language models on:
- **Long context understanding** (documents of varying lengths)
- **Information retrieval** within long documents
- **Reasoning** with extended context
- **Summarization** of long-form content
- **Question-answering** based on lengthy source material

---

## Task Breakdown

### What You Need to Do

#### 1. **Understand the CLI Commands**
   - Know what commands OLMES provides
   - Understand command syntax and options
   - Identify which commands apply to L-Eval

#### 2. **Validate CLI Execution**
   - Ensure OLMES is properly installed
   - Run sample CLI commands
   - Verify no errors in command execution
   - Test both basic and advanced options

#### 3. **Verify Metrics**
   - Run L-Eval benchmark using OLMES
   - Collect performance metrics (accuracy, F1, similarity scores, etc.)
   - Validate metric calculations
   - Cross-check results with expected outputs
   - Document metric values for your 70B model

#### 4. **Document Results**
   - Create test cases and validation reports
   - Record CLI command examples
   - Store metric outputs
   - Note any issues or findings

---

## Setup & Installation

### Prerequisites
```bash
# Required
- Python 3.9+
- pip or conda
- Git
- 16GB+ RAM (for model loading)
- Storage for L-Eval datasets (~10-50GB depending on configuration)
```

### Step 1: Install OLMES

```bash
# Option A: From source (if available)
git clone https://github.com/[repository]/olmes.git
cd olmes
pip install -e .

# Option B: From pip
pip install olmes

# Option C: With L-Eval support
pip install olmes[leval]
```

### Step 2: Verify Installation

```bash
# Check OLMES CLI is available
olmes --version

# Check help information
olmes --help

# Check L-Eval support
olmes eval --help
```

### Step 3: Download L-Eval Dataset

```bash
# This typically downloads benchmark datasets
olmes download leval

# Or manually if needed
# L-Eval datasets are usually available from official sources
```

---

## CLI Command Validation

### Common OLMES CLI Commands for L-Eval

#### 1. **List Available Benchmarks**
```bash
olmes list benchmarks
olmes list benchmarks --filter leval
```
**Expected Output**: Lists all available evaluation suites including L-Eval variants

#### 2. **Validate CLI Syntax**
```bash
# Show help for eval command
olmes eval --help

# Show benchmark-specific options
olmes eval leval --help
```

#### 3. **Run Evaluation (Test Phase)**
```bash
# Test with a smaller open-source model (e.g., Llama 2 7B, Mistral 7B, or similar)
olmes eval \
  --benchmark leval \
  --model-path [test-model-path] \
  --output ./results/

# With specific configuration for testing
olmes eval \
  --benchmark leval \
  --model-path [test-model-path] \
  --device cuda \
  --batch-size 1 \
  --num-shots 0 \
  --output ./results/leval_test_results.json

# With small subset for quick validation (2-3 minutes)
olmes eval \
  --benchmark leval \
  --model-path [test-model-path] \
  --num-tasks 10 \
  --output ./results/leval_smoke_test.json
```

**Note**: Use any available model (7B-13B open-source) for this pre-check. Results won't represent final performance but will validate the pipeline works.

#### 4. **Parse and Verify Results**
```bash
# Display results
olmes results ./results/leval_results.json

# Export results to different formats
olmes export \
  --input ./results/leval_results.json \
  --format csv \
  --output ./results/leval_metrics.csv

# Compare results
olmes compare \
  results/leval_results.json \
  other_results/baseline.json
```

### Validation Checklist

- [ ] OLMES version is correct and up-to-date
- [ ] All CLI commands execute without errors
- [ ] Help text displays properly for all commands
- [ ] Model can be loaded via `--model-path`
- [ ] Results are saved in expected format
- [ ] Metrics are calculated and displayed
- [ ] Output files are created successfully

---

## Metrics Verification

### Key Metrics to Verify

#### L-Eval Typically Measures:
1. **Accuracy** - Correct answer selection (%)
2. **F1 Score** - Precision and recall balance
3. **EM (Exact Match)** - Exact answer matching (%)
4. **ROUGE Scores** - Text similarity for generation tasks
5. **MRR (Mean Reciprocal Rank)** - Ranking quality
6. **Context Length Performance** - Performance vs context window size

### How to Verify Metrics

```bash
# 1. Run evaluation and capture metrics (using test model)
olmes eval --benchmark leval --model-path [test-model-path] --output metrics.json

# 2. Validate JSON structure
cat metrics.json | python -m json.tool > validated_metrics.json

# 3. Check for expected metric keys
python << 'EOF'
import json

with open('metrics.json', 'r') as f:
    results = json.load(f)

# Expected top-level keys
expected_keys = ['accuracy', 'f1', 'em', 'rouge', 'mrr', 'metadata']
actual_keys = set(results.keys())

print("✓ Metrics found:")
for key in actual_keys:
    print(f"  - {key}: {results[key]}")

print("\nValidation:")
for key in expected_keys:
    if key in actual_keys:
        print(f"  ✓ {key}")
    else:
        print(f"  ✗ {key} (missing)")
EOF
```

**Goal**: Ensure metrics are generated, properly formatted, and contain expected fields. Actual performance values are less important at this stage than confirming the pipeline works.

### Cross-Validation Steps

1. **Verify metric values are reasonable**
   - Accuracy should be 0-100%
   - F1 scores should be 0-1.0
   - ROUGE scores should be 0-1.0

2. **Check consistency**
   - Run evaluation twice, compare results
   - Results should be identical or very close

3. **Benchmark against baseline**
   - Compare metrics with reported L-Eval baselines
   - Your 70B model should exceed typical baselines

4. **Document findings**
   - Record all metrics in a results file
   - Note any anomalies or issues
   - Store for final report

---

## Recommended Approach

1. **Quick Smoke Test** (Time: ~5-10 minutes)
   ```bash
   # Test with 10 tasks only - just verify pipeline works
   olmes eval --benchmark leval --num-tasks 10 --model-path [small-test-model]
   ```

2. **Full Validation** (Time: ~30-60 minutes with 7B model)
   ```bash
   # Test with 100 tasks - validate metrics properly
   olmes eval --benchmark leval --num-tasks 100 --model-path [small-test-model]
   ```

3. **Test Model Options**
   ```bash
   # Use any of these freely available models for testing:
   # - meta-llama/Llama-2-7b-hf
   # - mistralai/Mistral-7B
   # - tiiuae/falcon-7b
   # - EleutherAI/pythia-6.9b
   
   olmes eval --benchmark leval --model-path meta-llama/Llama-2-7b-hf
   ```

4. **Cache Strategy** (Save for 70B model phase)
   ```bash
   # Create cache of test data and configs
   olmes eval --benchmark leval --use-cache --model-path [test-model]
   ```

### Priorities:
- ✅ Validate all CLI commands execute without errors
- ✅ Verify metrics pipeline produces valid output
- ✅ Test end-to-end workflow (setup → eval → results)
- ✅ Document any issues or fixes needed

---

## Deliverables

### Infrastructure & Setup
- [ ] OLMES installed and verified
- [ ] All CLI commands tested and working
- [ ] CLI help output documented
- [ ] Test model(s) selected and loaded successfully

### Validation Results
- [ ] Smoke test completed (10 tasks)
- [ ] Full validation completed (100 tasks)
- [ ] Metrics generated and valid
- [ ] Results saved in JSON/CSV format
- [ ] Pipeline produces expected output structure

### Documentation
- [ ] CLI commands and usage recorded
- [ ] Test results and logs archived
- [ ] Issues found and solutions documented
- [ ] Ready-to-run evaluation scripts prepared

---

## File Structure

```
OLMES/
├── README.md                      # This file
├── results/
│   ├── leval_results.json        # Raw evaluation results
│   ├── leval_metrics.csv         # Exported metrics
│   └── comparison_baseline.json   # Baseline comparison
├── scripts/
│   ├── run_leval.sh              # Main evaluation script
│   ├── validate_metrics.py       # Metrics validation
│   └── generate_report.py        # Result summary
├── logs/
│   └── evaluation.log            # Execution logs
└── docs/
    ├── cli_commands.md           # CLI documentation
    └── metrics_explanation.md    # Metrics guide
```

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `olmes: command not found` | Ensure pip install completed; check PATH |
| Model fails to load | Check model path and available VRAM |
| Out of memory errors | Reduce batch size or context length |
| Metrics not generated | Check output directory permissions |
| Invalid JSON in results | Verify OLMES version compatibility |
| Connection errors | Check internet for dataset download |

---

## Execution Steps

1. **Install OLMES** - Follow setup instructions above
2. **Validate CLI** - Run all commands from validation checklist
3. **Run Smoke Test** - Execute with 10 tasks on a test model (5 min)
4. **Run Full Validation** - Execute with 100 tasks on test model (1 hour)
5. **Collect Results** - Save and verify metrics are generated
6. **Document Pipeline** - Record what worked, issues found, fixes applied
7. **Prepare Infrastructure** - Scripts, configs, and logs ready

---

## Resources & References

- **OLMES GitHub**: [Check official repository]
- **L-Eval Paper**: Long Context Evaluation Suite documentation
- **70B Model**: [Your specific model name and source]
- **Budget Constraints**: Document expected compute time and cost

---

## Contact & Support

For issues or questions:
1. Check OLMES documentation and GitHub issues
2. Consult L-Eval official resources
3. Review model-specific documentation
4. Document issues for course instructors

---

**Last Updated**: February 28, 2026
**Status**: Ready for validation
**Phase**: Infrastructure Validation & Testing
