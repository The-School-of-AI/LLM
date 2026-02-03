# Tokenizer Benchmark Framework

A **benchmark-neutral** tokenizer evaluation system for comparing custom tokenizers against industry standards (GPT-4o, Qwen, DeepSeek).

## 🎯 Key Principles

> **Benchmark Neutrality (Non-Negotiable)**
> - Uses **format-only probes** (math layouts, MCQ templates, code shells)
> - Uses **synthetic instruction variants**, NOT real benchmark content
> - Validates that tokenizer behavior does not mirror benchmark data

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python run_benchmark.py --config config.yaml
```

## 📁 Project Structure

```
├── config.yaml              # Configuration
├── src/
│   ├── tokenizer_loader.py  # Tokenizer loading interface
│   ├── reference_tokenizers.py
│   ├── probes/              # Format-only probe generators
│   ├── metrics/             # Compression, fertility, speed
│   ├── validation/          # Neutrality checks
│   └── reporting/           # Tables and charts
├── tokenizers/              # [YOUR TOKENIZER HERE]
├── datasets/                # [YOUR CORPORA HERE]
└── reports/                 # Generated reports
```

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **Compression Ratio** | Tokens per byte/character |
| **Fertility** | Tokens per word (per language) |
| **Speed** | Encode/decode throughput |
| **Code Quality** | Keyword preservation, identifier handling |

## ✅ Validation Checks

1. **Benchmark Mirroring Detection** - Ensures tokenizer doesn't artificially optimize for known benchmarks
2. **Curriculum Difficulty Analysis** - Validates consistent behavior across difficulty bands
3. **Routing Skew Check** - Prevents uneven token distributions

## 🔧 Configuration

Edit `config.yaml` to:
- Add your custom tokenizer path
- Configure dataset locations
- Enable/disable specific metrics
- Adjust probe generation settings

## 📋 Usage

```bash
# Full benchmark
python run_benchmark.py --config config.yaml

# Dry run (no tokenizer required)
python run_benchmark.py --config config.yaml --dry-run

# Specific tokenizers only
python run_benchmark.py --tokenizers custom,gpt4o
```
