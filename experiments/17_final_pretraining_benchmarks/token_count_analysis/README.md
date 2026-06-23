# Benchmark Analysis & Dataset Investigation

This folder contains scripts and utilities for analyzing benchmark datasets before finalizing them for the pretraining evaluation pipeline.

## Objective

For each benchmark in `benchmarks-list.txt`, we need to determine:
1. **Total tokens in test data** - To compute FLOPs and resource requirements
2. **Evaluation metric** - The standard metric used (accuracy, F1, exact match, etc.)

## Folder Structure

```
benchmark_analysis/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── scripts/              
│   ├── analyze_all.py    # Main orchestrator - runs all analyses
│   ├── token_counter.py  # Token counting utilities
│   └── metric_finder.py  # Metric identification from papers/docs
├── data/                 # Downloaded test datasets (gitignored)
└── results/              # Analysis outputs (CSV/JSON)
    └── benchmark_summary.csv
```

## Usage

### 1. Install Dependencies

```bash
cd benchmark_analysis
pip install -r requirements.txt
```

### 2. Run Analysis for All Benchmarks

```bash
python scripts/analyze_all.py
```

This will:
- Load the benchmark list from `../benchmarks-list.txt`
- For each benchmark, attempt to download test data
- Count tokens using GPT-2 tokenizer (as baseline)
- Identify the standard evaluation metric
- Output results to `results/benchmark_summary.csv`

### 3. Run Analysis for Specific Benchmark

```bash
python scripts/analyze_all.py --benchmark "MMLU"
```

### 4. View Token Counts Only

```bash
python scripts/token_counter.py --dataset "path/to/dataset"
```

## Results Format

The analysis outputs a CSV with the following columns:

| Benchmark | Dataset Source | Test Set Size | Total Tokens | Avg Tokens/Sample | Metric | Notes |
|-----------|---------------|---------------|--------------|-------------------|--------|-------|
| MMLU | HuggingFace | 14,042 | 2,450,000 | 174 | Accuracy | 57 subjects |
| ... | ... | ... | ... | ... | ... | ... |

## Findings & Notes

### Benchmark-Specific Details

#### MMLU
- **Dataset**: `cais/mmlu`
- **Total tokens**: TBD
- **Metric**: Multi-class accuracy
- **Notes**: 

#### TriviaQA
- **Dataset**: `trivia_qa`
- **Total tokens**: TBD
- **Metric**: Exact match / F1
- **Notes**: 

#### GPQA Diamond
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### GSM8K
- **Dataset**: `gsm8k`
- **Total tokens**: TBD
- **Metric**: Exact match accuracy
- **Notes**: 

#### BBH (Big Bench Hard)
- **Dataset**: `lukaemon/bbh`
- **Total tokens**: TBD
- **Metric**: Accuracy (varies by task)
- **Notes**: 

#### ARC-Challenge
- **Dataset**: `ai2_arc`
- **Total tokens**: TBD
- **Metric**: Accuracy
- **Notes**: 

#### MATH
- **Dataset**: `hendrycks/math`
- **Total tokens**: TBD
- **Metric**: Exact match accuracy
- **Notes**: 

#### IFEval
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### SimpleQA_Verified
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### HumanEval
- **Dataset**: `openai_humaneval`
- **Total tokens**: TBD
- **Metric**: pass@k
- **Notes**: 

#### APPS
- **Dataset**: `codeparrot/apps`
- **Total tokens**: TBD
- **Metric**: Test case pass rate
- **Notes**: 

#### AIME 2025
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### MSGS
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### BLiMP
- **Dataset**: `blimp`
- **Total tokens**: TBD
- **Metric**: Accuracy
- **Notes**: 

#### IndicGLUE
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### IndicQA
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### L-Eval (Long Context Evaluation Suite)
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### RULER
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### TruthfulQA
- **Dataset**: `truthful_qa`
- **Total tokens**: TBD
- **Metric**: Accuracy (MC1, MC2)
- **Notes**: 

#### Indic-Bias (FairITales)
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### HELM Safety
- **Dataset**: TBD
- **Total tokens**: TBD
- **Metric**: TBD
- **Notes**: 

#### SWE-bench Verified
- **Dataset**: `princeton-nlp/SWE-bench`
- **Total tokens**: TBD
- **Metric**: Resolution rate
- **Notes**: 

#### HellaSwag
- **Dataset**: `hellaswag`
- **Total tokens**: TBD
- **Metric**: Accuracy
- **Notes**: 

#### Winogrande
- **Dataset**: `winogrande`
- **Total tokens**: TBD
- **Metric**: Accuracy
- **Notes**: 

## Next Steps

1. Run the analysis script to populate token counts
2. Verify metrics by checking original papers
3. Update the findings section with specific numbers
4. Use this data to finalize the configs in `../configs/`
5. Calculate FLOPs requirements for each band
