# Curriculum Band Classifier for LLM Pretraining

This tool classifies dataset records into **B0-B5 curriculum bands** based on token ID frequencies from Meta's Llama 3.3 70B tokenizer (BPE).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Classify your dataset (with local tokenizer)
python classify_curriculum_bands.py \
    --input your_dataset.jsonl \
    --output classified_dataset.jsonl \
    --text-field text \
    --local-tokenizer <path>\tokenizer

# Or try the example
python example_usage.py --local-tokenizer <path>\tokenizer
```

## Core Concept: Token ID vs Frequency

### Key Principle
In BPE tokenizers, **token IDs are inversely related to frequency**:
- **High frequency tokens** → **Low token IDs** (0, 1, 2, 3...)
- **Low frequency tokens** → **High token IDs** (10000, 20000, 50000...)

### Why This Works
BPE tokenizers build vocabulary by:
1. Starting with individual characters
2. Iteratively merging the **most frequent** pairs
3. Assigning IDs in order of creation

Result: Common words get low IDs, rare words get high IDs.

## Example

### B0 (Nursery) - High Frequency Tokens
```
Text: "The cat sat on the mat. It was happy."

Token IDs: [0, 45, 120, 8, 0, 350, 3, 12, 15, 280, 3]
Average ID: 85.5  (very low = high frequency)
Max ID: 350       (low = common words)
```

### B5 (PhD) - Low Frequency Tokens
```
Text: "The antidisestablishmentarian algorithm utilizes zephyr-based optimization."

Token IDs: [0, 25000, 8000, 12000, 15000, 18000]
Average ID: 14666  (very high = low frequency)
Max ID: 25000      (high = rare/technical terms)
```

## Curriculum Bands (B0-B5)

**Note**: Thresholds are tokenizer-specific. The values below are calibrated for Llama 3.3 (vocab size ~128K). Use `calibrate_thresholds.py` to find optimal thresholds for your tokenizer.

| Band | Name | Token ID Range (Llama 3.3) | Description |
|------|------|----------------------------|-------------|
| **B0** | Nursery | avg < 5000, max < 10000 | Very high frequency tokens, simple language, basic grammar |
| **B1** | Primary | avg < 10000, max < 20000 | High frequency tokens, everyday language, common knowledge |
| **B2** | High School | avg < 20000, max < 40000 | Medium frequency, structured knowledge, explanations |
| **B3** | Undergraduate | avg < 40000, max < 70000 | Lower frequency, technical content, code, algorithms |
| **B4** | Graduate | avg < 70000, max < 100000 | Low frequency, complex reasoning, research content |
| **B5** | PhD | avg ≥ 70000, max ≥ 100000 | Very low frequency, advanced/rare terms, agentic traces |

## Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install transformers torch numpy tqdm matplotlib
```

**Note**: `matplotlib` is optional but required for visualization features.

## Usage

### Basic Usage

```bash
# With local tokenizer (recommended)
python classify_curriculum_bands.py \
    --input dataset.jsonl \
    --output classified_dataset.jsonl \
    --text-field text \
    --local-tokenizer <path>\tokenizer

# With HuggingFace model (alternative)
python classify_curriculum_bands.py \
    --input dataset.jsonl \
    --output classified_dataset.jsonl \
    --text-field text \
    --model-id meta-llama/Llama-3.3-70B-Instruct
```

### Command Line Arguments

- `--input`: Path to input dataset file (JSON or JSONL)
- `--output`: Path to output classified dataset file
- `--text-field`: Name of field containing text (default: `"text"`)
- `--model-id`: HuggingFace model ID (default: `"meta-llama/Llama-3.3-70B-Instruct"`)
- `--local-tokenizer`: Local path to tokenizer files (overrides `--model-id`)
- `--format`: Input format - `json` or `jsonl` (default: `jsonl`)

### Example

```bash
python classify_curriculum_bands.py \
    --input my_dataset.jsonl \
    --output my_dataset_classified.jsonl \
    --text-field content \
    --format jsonl \
    --local-tokenizer <path>\tokenizer
```

## Input Format

### JSONL Format (Recommended)
Each line is a JSON object:
```json
{"text": "The cat sat on the mat.", "id": 1, "source": "web"}
{"text": "The algorithm uses dynamic programming.", "id": 2, "source": "code"}
```

### JSON Format
Array of objects:
```json
[
  {"text": "The cat sat on the mat.", "id": 1},
  {"text": "The algorithm uses dynamic programming.", "id": 2}
]
```

## Output Format

Each record is enhanced with classification metadata:

```json
{
  "text": "The cat sat on the mat.",
  "id": 1,
  "curriculum_band": "B0",
  "token_stats": {
    "avg": 85.5,
    "max": 350,
    "min": 0,
    "p50": 45.0,
    "p95": 280.0,
    "p99": 350.0,
    "count": 11
  },
  "classification_metadata": {
    "band": "B0",
    "description": "Nursery: Very high frequency tokens, simple language",
    "avg_token_id": 85.5,
    "max_token_id": 350,
    "p95_token_id": 280.0,
    "reason": "avg=85.5 <= 1000, max=350 <= 2000, p95=280.0 <= 1500"
  }
}
```

## Classification Logic

The classifier uses three metrics to determine band:
1. **Average token ID**: Mean of all token IDs in the text
2. **Maximum token ID**: Highest token ID in the text
3. **95th percentile (p95)**: 95% threshold - 95% of token IDs are below this value

**What is p95?**
- **p95 = 95% threshold** (or 95% cutoff)
- Meaning: 95% of token IDs are below this value; only 5% are above it
- **Example**: If p95 = 8000, then 95% of tokens have IDs ≤ 8000, and 5% have IDs > 8000
- **Why it's useful**: Captures high-end outliers (rare tokens) that might push max ID very high, giving a more stable measure than just the maximum

A record is assigned to the **highest band** (most difficult) that it qualifies for based on these thresholds.

## Programmatic Usage

```python
from classify_curriculum_bands import CurriculumBandClassifier

# Initialize classifier with local tokenizer (recommended)
classifier = CurriculumBandClassifier(
    local_tokenizer_path="<path>\\tokenizer"
)

# Or with HuggingFace model
classifier = CurriculumBandClassifier(
    model_id="meta-llama/Llama-3.3-70B-Instruct"
)

# Classify a single record
record = {"text": "The cat sat on the mat."}
classified = classifier.process_record(record, text_field="text")
print(f"Band: {classified['curriculum_band']}")
print(f"Avg token ID: {classified['token_stats']['avg']}")

# Process entire dataset
band_counts = classifier.process_dataset(
    input_file="dataset.jsonl",
    output_file="classified.jsonl",
    text_field="text"
)
```

## Understanding the Results

### B0 (Nursery)
- Simple, common words
- Basic grammar and syntax
- High frequency tokens (low IDs)
- Example: "The cat sat on the mat."

### B1 (Primary)
- Everyday language
- Common knowledge
- Still high frequency, but slightly more diverse
- Example: "The weather is nice today. We went to the park."

### B2 (High School)
- Structured explanations
- Educational content
- Medium frequency tokens
- Example: "Photosynthesis is the process by which plants convert sunlight into energy."

### B3 (Undergraduate)
- Technical content
- Code and algorithms
- Lower frequency tokens
- Example: "The quicksort algorithm uses divide-and-conquer with O(n log n) average complexity."

### B4 (Graduate)
- Complex reasoning
- Research-level content
- Low frequency tokens
- Example: "The transformer architecture employs self-attention mechanisms to model long-range dependencies."

### B5 (PhD)
- Advanced/rare terms
- Specialized vocabulary
- Very low frequency tokens
- Example: "The antidisestablishmentarian framework utilizes zephyr-based optimization for quantum computing applications."

## Customization

### Calibrating Thresholds for Your Tokenizer

**Important**: Different tokenizers have different vocabulary sizes and token ID ranges. The default thresholds are calibrated for Llama 3.3. For other tokenizers, you should calibrate thresholds using the provided tool.

#### Automatic Calibration

Use `calibrate_thresholds.py` to automatically determine optimal thresholds:

```bash
# With local tokenizer (recommended)
python calibrate_thresholds.py --local-tokenizer <path>\tokenizer

# With HuggingFace model (alternative)
python calibrate_thresholds.py
```

This script will:
1. Test sample texts from each difficulty band (B0-B5)
2. Analyze actual token ID ranges for your tokenizer
3. Suggest optimal threshold values
4. Print updated `TOKEN_ID_THRESHOLDS` dictionary to copy into your code

**Example output:**
```
B0 Samples:
  Sample 1: Avg: 2345.2, Max: 5678, P95: 3456.1
  ...

SUGGESTED THRESHOLDS
TOKEN_ID_THRESHOLDS = {
    'B0': {
        'avg_max': 5000,
        'max_max': 10000,
        'p95_max': 8000,
        ...
    },
    ...
}
```

#### Manual Threshold Adjustment

If you prefer to adjust manually, edit the `TOKEN_ID_THRESHOLDS` dictionary in `classify_curriculum_bands.py`:

```python
TOKEN_ID_THRESHOLDS = {
    'B0': {
        'avg_max': 5000,    # Adjust based on your tokenizer
        'max_max': 10000,
        'p95_max': 8000,
        ...
    },
    ...
}
```

**Guidelines for threshold selection:**
- Test with known simple texts (should be B0)
- Test with known complex texts (should be B4-B5)
- Ensure thresholds create clear separation between bands
- Consider your tokenizer's vocabulary size (larger vocab = higher IDs)

### Using Different Tokenizers

#### Local Tokenizer (Recommended)
```python
classifier = CurriculumBandClassifier(
    local_tokenizer_path="<path>\\tokenizer"
)
```

#### HuggingFace Model (Alternative)
```python
classifier = CurriculumBandClassifier(
    model_id="meta-llama/Llama-3.3-70B-Instruct"  # Or any other model
)
```

**Note**: When switching tokenizers, always recalibrate thresholds using `calibrate_thresholds.py --local-tokenizer <path>\tokenizer`.

## Visualization

Visualize band distribution from classified datasets:

```bash
python visualize_band_distribution.py --input classified_dataset.jsonl --output band_chart.png
```

Or programmatically:
```python
classifier.visualize_band_distribution(
    classified_file="classified.jsonl",
    output_file="band_distribution.png"
)
```

See `VISUALIZATION_GUIDE.md` for more details.

## Notes

- **Tokenizer dependency**: Results depend on the specific tokenizer used. Different tokenizers will produce different token IDs for the same text. **Always calibrate thresholds when switching tokenizers.**
- **BPE assumption**: This approach assumes a BPE tokenizer where token IDs correlate with frequency. Other tokenization methods may not follow this pattern.
- **Context matters**: The classification is based purely on token frequency, not semantic complexity. A simple sentence with rare words might be classified as B5.
- **Batch processing**: For large datasets, consider processing in batches to manage memory.
- **Threshold calibration**: Default thresholds are for Llama 3.3. Use `calibrate_thresholds.py` for other tokenizers.

## Troubleshooting

### "Field 'text' not found in record"
- Use `--text-field` to specify the correct field name
- Check your input file format

### Out of memory errors
- Process smaller batches
- Use a smaller model for tokenization if available

### Slow processing
- Normal for large datasets
- Consider parallel processing for very large files

## License

This tool is provided as-is for curriculum learning research and LLM pretraining.
