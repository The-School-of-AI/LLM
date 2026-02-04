# Quick Start Guide

## Setup (One-time)

```bash
cd experiments/17_final_pretraining_benchmarks/benchmark_analysis
pip install -r requirements.txt
```

## Usage

### Option 1: Analyze All Benchmarks

```bash
cd scripts
python analyze_all.py
```

This will:
- Read all benchmarks from `../benchmarks-list.txt`
- Count tokens for each test dataset
- Identify evaluation metrics
- Save results to `../results/benchmark_summary.csv` and `.json`

### Option 2: Analyze Single Benchmark

```bash
cd scripts
python analyze_all.py --benchmark "MMLU"
```

### Option 3: Use Individual Tools

**Count tokens only:**
```bash
cd scripts
python token_counter.py --dataset "gsm8k" --split "test" --config "main"
```

**Find metrics only:**
```bash
cd scripts
python metric_finder.py --file ../benchmarks-list.txt --output metrics.csv
```

## Expected Output

After running the analysis, you'll get:

1. **Terminal output**: Progress and summary statistics
2. **CSV file**: `results/benchmark_summary.csv` with columns:
   - Benchmark name
   - Dataset info
   - Test set size
   - Total tokens
   - Average tokens per sample
   - Evaluation metric
   - Status (success/error/manual)

3. **JSON file**: `results/benchmark_summary.json` (same data, more detailed)

## Example Output

```
Progress: [1/25]
================================================================================
Analyzing: MMLU
================================================================================

[1/2] Finding evaluation metric...
✓ Metric: accuracy

[2/2] Counting tokens in test dataset...
Loading dataset: cais/mmlu (split: test)
✓ Analyzed 14,042 samples
  Total tokens: 2,450,000
  Avg tokens/sample: 174.50
```

## Customization

### Use Different Tokenizer

```bash
python analyze_all.py --tokenizer "meta-llama/Llama-2-7b-hf"
```

### Output to Different Directory

```bash
python analyze_all.py --output-dir "./my_results"
```

## Next Steps

1. Review `results/benchmark_summary.csv`
2. Manually verify datasets marked as "manual" status
3. Calculate FLOPs requirements using total token counts
4. Update `../configs/` with appropriate batch sizes based on token counts
5. Document findings in the main `README.md`

## Troubleshooting

**Issue**: Dataset not found on HuggingFace
- **Solution**: Status will be "manual" - you'll need to manually download/analyze

**Issue**: Authentication error for gated datasets
- **Solution**: Run `huggingface-cli login` and provide your token

**Issue**: Memory error during analysis
- **Solution**: The script processes datasets in streaming mode when possible, but some datasets may be large. Consider analyzing them individually.
