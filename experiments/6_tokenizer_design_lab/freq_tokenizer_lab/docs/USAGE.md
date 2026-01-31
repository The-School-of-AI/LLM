# Usage Guide

Complete workflow guide for tokenizer evaluation, frequency analysis, and reindexing.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import transformers, datasets, yaml; print('✓ All dependencies installed')"
```

---

## Complete Workflow

### Phase 1: Tokenizer Evaluation

**Goal**: Evaluate all candidate tokenizers and select the best one for your use case.

#### Step 1.1: Run Evaluation Suite

```bash
cd freq_tokenizer_lab/src

python tokenizer_evaluator.py --config ../config.yaml
```

**What it does**:
- Loads all tokenizers from `filtered_tokenizer/` directory
- Runs benchmarks on Indic text (Devanagari quality, byte-fallback, fragmentation)
- Runs benchmarks on code (Python, JS, C++, tokenization efficiency)
- Runs benchmarks on JSON (structure preservation)
- Generates scorecard with rankings

**Output**: `../results/evaluation_results.json`

#### Step 1.2: Review Results

```bash
# View evaluation results
cat ../results/evaluation_results.json | jq '.summary'
```

**Expected output**:
```json
{
  "total_evaluated": 10,
  "passed_filters": 7,
  "failed_filters": 3,
  "recommended": "ds_filtered"
}
```

**Decision Point**: Select top 1-3 tokenizers for frequency analysis.

---

### Phase 2: Frequency Analysis

**Goal**: Compute token frequency statistics on target datasets (IndicCorpV2, Dolma).

#### Step 2.1: Analyze Indic Dataset

```bash
python frequency_analyzer.py \
  --tokenizer ds_filtered \
  --dataset indic \
  --config ../config.yaml \
  --output ../results/frequency_stats/ds_indic_freq.json
```

**What it does**:
- Streams IndicCorpV2 dataset (configurable sample size, default 10GB)
- Tokenizes with selected tokenizer
- Builds frequency distribution
- Computes percentiles, head/torso/tail classification
- Applies log smoothing for MoE safety

**Time estimate**: 30-60 minutes (depends on dataset sample size)

**Output**: `../results/frequency_stats/ds_indic_freq.json`

#### Step 2.2: Analyze Code Dataset

```bash
python frequency_analyzer.py \
  --tokenizer ds_filtered \
  --dataset code \
  --config ../config.yaml \
  --output ../results/frequency_stats/ds_code_freq.json
```

**What it does**:
- Streams Dolma dataset (configurable sample size, default 20GB)
- Tokenizes code-heavy documents
- Builds frequency distribution for code patterns

**Time estimate**: 45-90 minutes

**Output**: `../results/frequency_stats/ds_code_freq.json`

#### Step 2.3: Merge Frequency Stats

```python
# Create a script to merge stats from multiple datasets
from src.frequency_analyzer import FrequencyAnalyzer
import json

analyzer = FrequencyAnalyzer("../config.yaml")

# Load individual stats
with open("../results/frequency_stats/ds_indic_freq.json") as f:
    indic_stats = json.load(f)

with open("../results/frequency_stats/ds_code_freq.json") as f:
    code_stats = json.load(f)

# Create FrequencyStats objects and merge
from dataclasses import dataclass
from frequency_analyzer import FrequencyStats

indic_obj = FrequencyStats(**indic_stats)
code_obj = FrequencyStats(**code_stats)

merged = analyzer.merge_frequency_stats(
    [indic_obj, code_obj],
    output_path="../results/frequency_stats/ds_merged_freq.json"
)

print(f"Merged stats: {merged.total_tokens:,} total tokens")
```

**Output**: `../results/frequency_stats/ds_merged_freq.json`

---

### Phase 3: Token ID Reindexing

**Goal**: Create new tokenizer with frequency-aware ID allocation.

#### Step 3.1: Run Reindexer

```bash
python id_reindexer.py \
  --tokenizer ds_filtered \
  --frequency-stats ../results/frequency_stats/ds_merged_freq.json \
  --config ../config.yaml \
  --output ../results/reindexed_tokenizers/ds_reindexed/
```

**What it does**:
- Loads original tokenizer vocabulary
- Loads merged frequency statistics
- Applies frequency-aware reindexing strategy (category blocks by default)
- Allocates special tokens (IDs 0-255)
- Sorts tokens by log-smoothed frequency within each block
- Generates new token → ID mappings
- Saves reindexed tokenizer files

**Time estimate**: < 1 minute

**Output**: `../results/reindexed_tokenizers/ds_reindexed/`
- `tokenizer_reindexed.json` - New vocab (token → ID)
- `id_mapping.json` - Old ID → New ID lookup
- `id_to_token.json` - New ID → Token lookup
- `metadata.json` - ID ranges, statistics

#### Step 3.2: Review Metadata

```bash
cat ../results/reindexed_tokenizers/ds_reindexed/metadata.json | jq
```

**Expected output**:
```json
{
  "tokenizer_name": "ds_filtered",
  "strategy": "category_blocks",
  "vocab_size": 77900,
  "id_ranges": {
    "special": {"start": 0, "end": 255},
    "high_frequency": {"start": 256, "end": 10000},
    "medium_frequency": {"start": 10000, "end": 80000},
    "low_frequency": {"start": 80000, "end": 128000}
  },
  "special_tokens_count": 64,
  "head_tokens_count": 9744,
  "torso_tokens_count": 58092,
  "tail_tokens_count": 10000
}
```

---

### Phase 4: Validation

**Goal**: Verify that reindexing preserved tokenization behavior.

#### Step 4.1: Run Validation Suite

```bash
python validation_suite.py \
  --original ds_filtered \
  --reindexed ../results/reindexed_tokenizers/ds_reindexed/ \
  --config ../config.yaml \
  --output ../results/validation_report.json
```

**What it does**:
- Loads original and reindexed tokenizers
- Tests vocab size preservation
- Tests token string preservation (no changes)
- Tests ID mapping consistency
- Tests special token handling
- Tests encode/decode equivalence on sample texts
- Tests frequency ordering (category blocks)

**Time estimate**: < 1 minute

**Output**: `../results/validation_report.json`

#### Step 4.2: Review Validation Report

```bash
cat ../results/validation_report.json | jq '.results[] | {test: .test_name, passed: .passed}'
```

**Expected output**:
```json
{"test": "vocab_size", "passed": true}
{"test": "token_strings_preserved", "passed": true}
{"test": "id_mapping_consistency", "passed": true}
{"test": "special_tokens", "passed": true}
{"test": "encode_decode_equivalence", "passed": true}
{"test": "frequency_ordering", "passed": true}
```

**Decision Point**: All tests must pass before using reindexed tokenizer.

---

## Advanced Usage

### Custom Frequency Sampling

Control dataset sample size:

```bash
# Analyze smaller sample (1000 documents)
python frequency_analyzer.py \
  --tokenizer qwen_filtered \
  --dataset indic \
  --max-samples 1000 \
  --output ../results/frequency_stats/qwen_indic_sample.json
```

### Custom ID Range Configuration

Edit `config.yaml` to adjust block sizes:

```yaml
reindexing:
  category_blocks:
    special_tokens:
      start_id: 0
      end_id: 511  # Increased for more special tokens

    high_frequency:
      start_id: 512
      end_id: 20000  # Larger head block

    medium_frequency:
      start_id: 20000
      end_id: 100000  # Larger torso

    low_frequency:
      start_id: 100000
      end_id: 128000
```

### Pure Frequency Strategy

Switch to pure frequency ordering (no category blocks):

```yaml
reindexing:
  strategy: "pure_frequency"  # Change from "category_blocks"
```

### Add Custom Special Tokens

Edit `config.yaml`:

```yaml
special_tokens:
  # Add new category
  custom:
    - name: "thought_begin"
      token: "<|thought_begin|>"
      id: 100
    - name: "thought_end"
      token: "<|thought_end|>"
      id: 101
```

Then re-run reindexer.

---

## Working with Reindexed Tokenizers

### Load in Python

```python
import json

# Load reindexed tokenizer
with open("results/reindexed_tokenizers/ds_reindexed/tokenizer_reindexed.json") as f:
    token_to_id = json.load(f)

with open("results/reindexed_tokenizers/ds_reindexed/id_to_token.json") as f:
    id_to_token_str = json.load(f)
    id_to_token = {int(k): v for k, v in id_to_token_str.items()}

print(f"Vocab size: {len(token_to_id)}")

# Check a token's ID
print(f"Token 'the' has ID: {token_to_id.get('the')}")

# Check what token has ID 1000
print(f"ID 1000 is token: {id_to_token.get(1000)}")
```

### Use with Special Tokens

```python
from src.special_tokens import SpecialTokenRegistry, SpecialTokenEncoder

# Load special token registry
registry = SpecialTokenRegistry("config.yaml")

# Create encoder
encoder = SpecialTokenEncoder(registry)

# Wrap text with special tokens
doc = encoder.wrap_document("This is a test.")
print(doc)
# Output: <|begin_of_text|>This is a test.<|end_of_text|>

# Wrap code block
code = encoder.wrap_code_block("def hello():\n    print('hi')", language="python")
print(code)
# Output: <|code_begin|><|lang:python|>def hello():\n    print('hi')<|code_end|>
```

### Check Token Frequency Band

```python
# Quick heuristic based on ID ranges
def get_frequency_band(token_id):
    if token_id < 256:
        return "special"
    elif token_id < 10_000:
        return "head (high frequency)"
    elif token_id < 80_000:
        return "torso (medium frequency)"
    else:
        return "tail (low frequency)"

# Example usage
token_id = token_to_id.get("the")
print(f"Token 'the' is in: {get_frequency_band(token_id)}")
```

---

## Troubleshooting

### Issue: Dataset Loading Fails

**Error**: `ConnectionError: Couldn't reach https://huggingface.co`

**Solution**:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Try with authentication token
huggingface-cli login

# Re-run frequency analyzer
python frequency_analyzer.py ...
```

### Issue: Memory Error During Frequency Analysis

**Error**: `MemoryError: Unable to allocate array`

**Solution**:
```bash
# Reduce sample size in config.yaml
# frequency_analysis → datasets → indic → sample_size_gb: 5  # Reduced from 10

# Or use max-samples flag
python frequency_analyzer.py --max-samples 5000 ...
```

### Issue: Validation Fails (Token Strings Not Preserved)

**Error**: `token_strings_preserved: FAIL`

**Root Cause**: Likely an issue with special token handling or merging

**Solution**:
1. Check that special tokens are defined correctly in config
2. Review `id_mapping.json` for inconsistencies
3. Re-run reindexer with fresh frequency stats

### Issue: Encode/Decode Not Equivalent

**Error**: `encode_decode_equivalence: FAIL`

**Root Cause**: Token collision or byte-fallback handling issue

**Solution**:
1. Check validation report details: `jq '.results[] | select(.test_name == "encode_decode_equivalence")' validation_report.json`
2. Inspect mismatched samples
3. Verify original tokenizer has byte-fallback tokens (`<0xXX>`)

---

## Best Practices

### 1. Always Validate After Reindexing

```bash
# Never skip validation
python validation_suite.py --original X --reindexed Y --config config.yaml
```

### 2. Keep Original Tokenizers Unchanged

```bash
# Create backups before any operations
cp -r filtered_tokenizer/ filtered_tokenizer_backup/
```

### 3. Document Your Decisions

Create a `DECISIONS.md` in your results directory:

```markdown
# Tokenizer Selection Decisions

- **Evaluated**: 10 candidates
- **Selected**: ds_filtered
- **Reason**: Best Devanagari quality (score: 87.3), low byte-fallback (12%)
- **Frequency Analysis**: 10GB Indic + 20GB Dolma (total: 3.2B tokens)
- **Reindexing Strategy**: category_blocks with log smoothing (temp=0.1)
- **Validation**: All tests passed ✓
```

### 4. Version Control Your Config

```bash
# Track config changes
git add config.yaml
git commit -m "Update: Increased high_freq block to 20k tokens"
```

### 5. Monitor MoE Routing (If Applicable)

After training with reindexed tokenizer:

```python
# Check if routing is frequency-biased
from scipy.stats import spearmanr

token_ids = [...]  # Extract from validation set
router_probs = [...]  # Extract from model

correlation = spearmanr(token_ids, router_probs)
print(f"Routing-ID correlation: {correlation.correlation:.3f}")
# Target: < 0.3 (low correlation)
```

---

## Quick Reference

### File Locations

```
freq_tokenizer_lab/
├── config.yaml                         # Main configuration
├── src/
│   ├── tokenizer_evaluator.py         # Phase 1: Evaluation
│   ├── frequency_analyzer.py          # Phase 2: Frequency analysis
│   ├── id_reindexer.py                # Phase 3: Reindexing
│   ├── validation_suite.py            # Phase 4: Validation
│   └── special_tokens.py              # Utilities
├── results/
│   ├── evaluation_results.json        # Evaluation scorecard
│   ├── frequency_stats/               # Frequency distributions
│   ├── reindexed_tokenizers/          # Output tokenizers
│   └── validation_report.json         # Validation results
└── docs/
    ├── ID_SCHEME.md                   # ID allocation details
    └── USAGE.md                       # This file
```

### Command Cheat Sheet

```bash
# Evaluate all tokenizers
python tokenizer_evaluator.py --config config.yaml

# Analyze frequency (Indic)
python frequency_analyzer.py --tokenizer ds_filtered --dataset indic --output freq.json

# Analyze frequency (Code)
python frequency_analyzer.py --tokenizer ds_filtered --dataset code --output freq.json

# Reindex tokenizer
python id_reindexer.py --tokenizer ds_filtered --frequency-stats freq.json --output reindexed/

# Validate reindexed tokenizer
python validation_suite.py --original ds_filtered --reindexed reindexed/ --output report.json

# Print special tokens
python special_tokens.py --config config.yaml --demo
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-31
**Maintainer**: Token Reindexing Lab Team
