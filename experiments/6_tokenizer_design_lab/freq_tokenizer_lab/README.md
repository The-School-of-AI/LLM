# Frequency-Aware Tokenizer Selection & Reindexing Lab

A comprehensive system for selecting, evaluating, and adapting frontier tokenizers with frequency-aware token ID reindexing.

## Objective

Build a 128k-token tokenizer that is:
- **Indic-capable**: Strong Devanagari support with minimal byte-fallback
- **Code-optimized**: Efficient Python, JS/TS, C/C++ tokenization
- **JSON-friendly**: Clean handling of structured data and tool-calling
- **MoE-safe**: Minimized routing skew through frequency-aware ID allocation
- **Dataset-aware**: Token IDs encode soft inverse frequency from target corpus

## Philosophy

**We do not train tokenizers.** Instead, we:
1. Select from existing frontier tokenizers
2. Measure and evaluate their behavior
3. Surgically adapt through ID reindexing (preserving token strings and merges)

## Available Tokenizers (Candidates)

- `byted` - ByteDance tokenizer (77k tokens, 10k Indic)
- `ds` - DeepSeek base (77k tokens, 10k Indic)
- `dscode` - DeepSeek Code optimized
- `gemma` - Google Gemma
- `gptoss` - GPT Open Source
- `mistral` - Mistral AI
- `olmo` - OLMo base
- `olmocode` - OLMo Code
- `qwen` - Qwen base (100k tokens, 8.8k Indic)
- `qwencode` - Qwen Code

## Project Structure

```
freq_tokenizer_lab/
├── src/
│   ├── tokenizer_evaluator.py    # Benchmark suite for evaluation
│   ├── frequency_analyzer.py     # Streaming frequency computation
│   ├── id_reindexer.py          # Frequency-aware remapping
│   ├── special_tokens.py        # Special token definitions
│   └── validation_suite.py      # Test harness
├── data/                        # Temporary data cache
├── results/
│   ├── evaluation_results.json
│   ├── frequency_stats/
│   └── reindexed_tokenizers/
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── ID_SCHEME.md
│   └── USAGE.md
├── config.yaml                  # Configuration
└── requirements.txt
```

## Workflow

### Phase 1: Evaluation & Selection

```bash
python src/tokenizer_evaluator.py --config config.yaml
```

Evaluates all candidate tokenizers on:
- **Indic benchmarks**: Devanagari quality, byte-fallback rate, fragmentation
- **Code benchmarks**: Python/JS/C++ tokenization efficiency
- **JSON benchmarks**: Structured data handling

Produces `evaluation_results.json` with scorecard and top 3 recommendations.

### Phase 2: Frequency Analysis

```bash
python src/frequency_analyzer.py \
  --tokenizer ds_filtered \
  --config config.yaml \
  --dataset indic \
  --output results/frequency_stats/ds_indic_freq.json
```

Streams through target datasets (IndicCorpV2, Dolma) and computes:
- Token frequency distribution
- Percentile bands (p50, p75, p90, p95, p99)
- Head/torso/tail classification
- Log-smoothed frequencies for MoE safety

### Phase 3: Token ID Reindexing

```bash
python src/id_reindexer.py \
  --tokenizer ds_filtered \
  --frequency-stats results/frequency_stats/ds_combined_freq.json \
  --config config.yaml \
  --output results/reindexed_tokenizers/ds_reindexed/
```

Generates:
- `tokenizer_reindexed.json` - New token → ID mapping
- `id_mapping.json` - Old ID → New ID lookup
- `merges_reindexed.txt` - BPE merges with new IDs
- `vocab_metadata.json` - ID ranges, percentiles, documentation

### Phase 4: Validation

```bash
python src/validation_suite.py \
  --original-tokenizer ds_filtered \
  --reindexed-tokenizer results/reindexed_tokenizers/ds_reindexed/ \
  --config config.yaml
```

Validates:
- Encoding/decoding equivalence
- Special token handling
- No token string changes
- Merge rule preservation

## ID Allocation Strategy

### Category Blocks (Default)

```
ID Range     | Category        | Description
-------------|-----------------|-----------------------------------
0-255        | Special Tokens  | Reserved for control tokens
256-10,000   | High Frequency  | Top 10% (head) - common words, symbols
10,000-80,000| Medium Frequency| Middle 50% (torso) - regular vocab
80,000-128,000| Low Frequency  | Bottom 40% (tail) - rare tokens, junk
```

IDs within each block are sorted by **log-smoothed frequency** to reduce MoE routing artifacts.

## Special Tokens

128 special tokens reserved in ID range 0-255:

- **Document**: `<|begin_of_text|>`, `<|end_of_text|>`, `<|chunk_sep|>`
- **Chat**: `<|system|>`, `<|user|>`, `<|assistant|>`
- **Code**: `<|code_begin|>`, `<|code_end|>`, `<|lang:python|>`, etc.
- **JSON/Tools**: `<|json_begin|>`, `<|tool_call|>`, `<|tool_result|>`
- **Metadata**: `<|source:wikipedia|>`, `<|source:github|>`, etc.

See `config.yaml` for full definitions.

## Target Datasets

### Indic: ai4bharat/IndicCorpV2
- Multi-lingual Indic corpus (Hindi, Bengali, Tamil, Telugu, etc.)
- Devanagari-heavy (Hindi, Marathi, Sanskrit)
- 10GB sample for frequency analysis

### Code/English: allenai/dolma3_dolmino_mix-100B-1125
- Code-heavy mix from Dolma 3
- Python, JS, C++, structured configs
- 20GB sample for frequency analysis

## Key Design Decisions

### ID Reordering Strategy
- **Category blocks** (default): Balances frequency with interpretability
- Alternative: Pure frequency descending

### Frequency Smoothing
- **Log smoothing** (default): `new_freq = log(1 + freq)`
- Prevents exact rank = exact frequency (reduces MoE skew)

### Special Token Count
- **128 reserved** (0-127): Defined tokens
- **128 future** (128-255): Reserved for expansion

## Usage Example

```python
from src.tokenizer_evaluator import TokenizerEvaluator
from src.frequency_analyzer import FrequencyAnalyzer
from src.id_reindexer import TokenIDReindexer

# 1. Evaluate tokenizers
evaluator = TokenizerEvaluator("config.yaml")
results = evaluator.run_all_benchmarks()
best_tokenizer = results["top_ranked"][0]

# 2. Compute frequencies
analyzer = FrequencyAnalyzer(best_tokenizer, "config.yaml")
freq_stats = analyzer.analyze_datasets()

# 3. Reindex IDs
reindexer = TokenIDReindexer(best_tokenizer, freq_stats, "config.yaml")
reindexer.reindex_and_save("results/reindexed_tokenizers/final/")
```

## Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers`, `tokenizers` - Tokenizer loading
- `datasets`, `huggingface_hub` - Dataset streaming
- `numpy`, `pandas` - Data processing
- `pyyaml` - Configuration

## Team Charter

This team:
- ✅ Selects frontier tokenizers
- ✅ Measures and evaluates behavior
- ✅ Reindexes token IDs based on frequency
- ✅ Defines special token schemes
- ❌ Does NOT train tokenizers from scratch

## License

MIT

## Authors

Token Reindexing Lab Team
