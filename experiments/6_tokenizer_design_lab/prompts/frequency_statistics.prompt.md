# Prompt: Cross-Tokenizer Token Commonality Analysis

## Objective
Create a Jupyter notebook that analyzes English, Indic, Code, and JSON tokens across multiple LLM tokenizers to find common vocabulary patterns and token position consistency.

## Tokenizers to Analyze
Load the following tokenizers from JSON files in `data/` directory:
| Model Key | File | Description |
|-----------|------|-------------|
| ByteD | byted_tokenizer.json | ByteDance tokenizer |
| DeepSeek | ds_tokenizer.json | DeepSeek V3 tokenizer |
| DSCoder | dscode_tokenizer.json | DeepSeek Coder tokenizer |
| GPT-OSS | gptoss_tokenizer.json | OpenAI-style tokenizer |
| Mistral | mistral_tokenizer.json | Mistral AI tokenizer |
| OLMo | olmo_tokenizer.json | AI2 OLMo tokenizer |
| OLMoCode | olmocode_tokenizer.json | AI2 OLMo Code tokenizer |
| Qwen | qwen_tokenizer.json | Alibaba Qwen tokenizer |
| QwenCode | qwencode_tokenizer.json | Alibaba Qwen Code tokenizer |
| Gemma | gemma_tokenizer.json | Google Gemma tokenizer |

## Required Libraries
```python
import json
import re
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Notebook Structure

### 1. Imports and Setup
- Import required libraries
- Define paths: `DATA_DIR`, `OUTPUT_DIR`, `CHARTS_DIR`
- Create output directories if they don't exist
- Define tokenizer file mappings and `MODEL_ORDER` list

### 2. Helper Functions

#### Byte Decoder (GPT-2 Style)
```python
def create_byte_decoder():
    """Create GPT-2 style byte decoder mapping for BPE tokens."""
    # Maps encoded characters back to byte values
```

#### Token Decoding
```python
def decode_bpe_token(token):
    """Decode a BPE token to its actual string representation."""
```

#### Language Classification
Implement functions for:
- `is_english_token(text)` - Check if token is primarily ASCII English
- `classify_indic_script(text)` - Detect Indic scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Odia, Gurmukhi, Sinhala)
- `is_code_token(token)` - Detect programming-related tokens (keywords, operators, patterns like camelCase/snake_case)
- `is_json_token(token)` - Detect JSON structural tokens (`{}[],:`, literals like `true/false/null`)
- `normalize_token(token)` - Normalize for cross-tokenizer comparison (strip space markers, lowercase)

#### Unicode Ranges for Indic Scripts
```python
INDIC_RANGES = {
    'Devanagari': (0x0900, 0x097F),
    'Bengali': (0x0980, 0x09FF),
    'Gurmukhi': (0x0A00, 0x0A7F),
    'Gujarati': (0x0A80, 0x0AFF),
    'Odia': (0x0B00, 0x0B7F),
    'Tamil': (0x0B80, 0x0BFF),
    'Telugu': (0x0C00, 0x0C7F),
    'Kannada': (0x0C80, 0x0CFF),
    'Malayalam': (0x0D00, 0x0D7F),
    'Sinhala': (0x0D80, 0x0DFF),
}
```

### 3. Load All Tokenizers
- Load each tokenizer JSON file
- Extract vocabulary from `data['model']['vocab']`
- Track vocabulary sizes per model
- Print loading status for each tokenizer

### 4. Extract and Classify Tokens
- Build a **token registry**: `normalized_token -> {model: {raw_token, token_id, decoded, category, normalized_id}}`
- **Normalized ID** = `token_id / vocab_size` (0-1 scale for cross-model comparison)
- Track statistics per model: English count, Indic count, Code count, JSON count
- Skip very short tokens (< 2 characters)

### 5. Build Cross-Tokenizer Comparison Table
Create a DataFrame with columns:
- `normalized_token` - Normalized token string
- `model_count` - Number of models containing this token
- `category` - Primary category (English, Indic script name)
- `is_code`, `is_json` - Boolean flags
- `{model}_token`, `{model}_token_id`, `{model}_norm_id` - Per-model data
- `min_norm_id`, `max_norm_id`, `avg_norm_id`, `norm_id_range` - Position statistics

### 6. Sort by Commonality and Importance
Sort tokens by:
1. `model_count` (descending) - Tokens in more models are more common
2. `avg_norm_id` (ascending) - Lower IDs = more frequent in training
3. `norm_id_range` (ascending) - Consistent ranking across models

Add `importance_rank` column.

### 7. View Tokens by Category
Display:
- Category distribution
- Top 20 English tokens
- Top 20 Indic tokens
- Top 20 Code-related tokens
- Top 20 JSON-related tokens

### 8. Visualizations

#### Token Position Consistency (4-panel)
1. **Correlation heatmap** - Token position correlation across all models
2. **Histogram** - Distribution of `norm_id_range` (position variability)
3. **Scatter plot** - Compare two models' normalized IDs
4. **Box plots** - Token position distribution per model

Save to: `charts/token_position_consistency.png`

#### Pairwise Model Comparisons
- Grid of scatter plots for all model pairs
- Show correlation coefficient for each pair
- Save to: `charts/pairwise_comparisons.png`

#### Code Token Pairwise Analysis
- Filter to `is_code == True` tokens only
- Pairwise scatter plots with correlation
- Save to: `charts/code_token_pairwise.png`

#### JSON Token Pairwise Analysis
- Filter to `is_json == True` tokens only
- Pairwise scatter plots with correlation
- Save to: `charts/json_token_pairwise.png`

#### General-Purpose Tokenizer Correlation
- Exclude code-specific models: DSCoder, QwenCode, OLMoCode
- Analyze 7 general-purpose models: ByteD, DeepSeek, GPT-OSS, Mistral, OLMo, Qwen, Gemma
- Correlation heatmap + sample scatter plot
- Save to: `charts/general_tokenizer_correlation.png`

#### Common Tokens Correlation Analysis
**Part 1**: Tokens common to ALL general-purpose models
- Correlation heatmap
- Bar chart of pairwise correlations (highlight top 3)
- Save to: `charts/common_tokens_correlation.png`

**Part 2**: Top 3 most correlated model pairs
- Scatter plots for each top pair
- Save to: `charts/top3_pairs_correlation.png`

#### Parallel Coordinates Plot
- Top 200 tokens
- Color by consistency (green = consistent, red = variable)
- Save to: `charts/parallel_coordinates.png`

#### Analysis by Model Count
- Box plot of variability by model count
- Distribution of avg_norm_id by model count
- Category breakdown stacked bar chart
- Scatter: importance vs variability
- Save to: `charts/model_count_analysis.png`

### 9. Export Results
Export to CSV: `output/token_commonality_analysis.csv`

Columns:
- importance_rank, normalized_token, category, model_count, is_code, is_json
- Per-model: {model}_token, {model}_token_id, {model}_norm_id
- Summary: min_norm_id, max_norm_id, avg_norm_id, norm_id_range

### 10. Summary Statistics
Print comprehensive summary:
- Total unique tokens analyzed
- Tokens by model count
- Category breakdown with percentages
- Code & JSON token counts
- Vocabulary sizes per model
- Position consistency statistics
- List of output files

## Key Analysis Insights to Capture

1. **Cross-Tokenizer Overlap**: How many tokens are shared across all/most tokenizers
2. **Language Distribution**: English vs Indic token distribution
3. **Position Consistency**: Do common tokens have similar relative positions?
4. **Model Cohorts**: Identify which tokenizers are most similar (e.g., Qwen/GPT-OSS/OLMo vs Mistral/DeepSeek/ByteDance)
5. **Gemma Outlier**: Note SentencePiece-based Gemma shows lower correlation with BPE tokenizers

## Output Files
- `output/token_commonality_analysis.csv`
- `charts/token_position_consistency.png`
- `charts/pairwise_comparisons.png`
- `charts/code_token_pairwise.png`
- `charts/json_token_pairwise.png`
- `charts/general_tokenizer_correlation.png`
- `charts/common_tokens_correlation.png`
- `charts/top3_pairs_correlation.png`
- `charts/parallel_coordinates.png`
- `charts/model_count_analysis.png`

## Chart Styling
- Use `figsize` appropriate for each visualization type
- Use colormaps: `RdYlGn` for correlation, `Set3` for categories
- Include colorbar for heatmaps
- Add correlation values as text on heatmaps
- Set axis limits to [0, 1] for normalized ID scatter plots
- Add red dashed diagonal line for "perfect alignment" reference
- Save all charts at `dpi=150` with `bbox_inches='tight'`
