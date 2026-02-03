# Prompt: Create Tokenizer Statistics Analysis Notebook

## Objective

Create a Jupyter Notebook called `tokens_statistics.ipynb` that performs comprehensive analysis of LLM tokenizers to evaluate their vocabulary composition, token length distributions, language coverage, and specialized token capabilities.

## Context

You are analyzing tokenizer JSON files from various LLM models (e.g., DeepSeek, Qwen, Gemma, GPT-OSS, OLMo, Mistral, ByteDance Seed) stored in a `data/` directory. The JSON files follow HuggingFace tokenizer format with `model.vocab` and `added_tokens` sections.

## Requirements

### Setup and Structure

1. **Directory Structure**:
   - Input: `data/*_tokenizer.json` - tokenizer files to analyze
   - Output: `output/` - CSV result files
   - Charts: `charts/` - PNG visualization files

2. **Dependencies**:
   - `json`, `os`, `re` (standard library)
   - `collections.defaultdict`
   - `pathlib.Path`
   - `pandas`
   - `numpy`
   - `matplotlib`

### Core Helper Functions

Create the following helper functions:

1. **`load_tokenizer(file_path)`**: Load tokenizer JSON and extract vocabulary, added tokens, and tokenizer type.

2. **`create_byte_decoder()`**: Create GPT-2/byte-level BPE character-to-byte mapping for proper token decoding. This is critical for accurately decoding tokens.

3. **`decode_bpe_token(token)`**: Decode byte-level BPE tokens back to original text using the byte decoder.

4. **`get_token_length(token)`**: Get actual character length after decoding.

5. **`detect_language(token)`**: Detect language/script using Unicode code point ranges. Support:
   - CJK (Chinese, Japanese Hiragana/Katakana, Korean)
   - Cyrillic, Arabic, Hebrew, Greek, Thai
   - Indic scripts: Hindi (Devanagari), Bengali, Punjabi (Gurmukhi), Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala
   - Latin/ASCII, Numbers, Punctuation, Whitespace

6. **`is_code_token(token)`**: Detect programming-related tokens (keywords, operators, patterns like camelCase, snake_case).

7. **`is_json_token(token)`**: Detect JSON-related tokens.

8. **`is_function_call_token(token)`**: Detect function call related tokens.

### Analysis Sections

Create the following analysis cells, each with printed output and visualizations:

#### 1. Vocabulary Size Statistics
- Load all tokenizer files
- Calculate total vocab size, base vocab, added tokens
- Create bar chart of vocabulary sizes
- Save chart as `vocab_sizes.png`

#### 2. Largest Tokens Analysis
- Find top 10 largest tokens (by decoded character length) per tokenizer
- Display sample tokens with lengths
- Create bar chart of max token lengths
- Save chart as `max_token_lengths.png`

#### 3. Tokens Over 32 Characters
- Count tokens with length > 32 characters
- Calculate percentage of vocabulary
- Find tokens in the 33-40 character range
- Create side-by-side charts (count and percentage)
- Save chart as `tokens_over_32.png`

#### 4. Language/Script Distribution
- Analyze language distribution across all tokens
- Count tokens per language/script
- Create stacked bar chart showing distribution
- Save to `output/language_distribution.csv`
- Save chart as `language_distribution.png`

#### 5. Indic and English Coverage
- Focus analysis on Indic scripts: Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Odia, Punjabi
- Compare English vs Indic token counts and percentages
- Create 4-panel figure:
  - English vs Indic counts
  - English vs Indic percentages
  - Indic language breakdown (stacked)
  - Indic coverage ranking (horizontal bars)
- Save to `output/indic_english_coverage.csv`
- Save chart as `indic_english_coverage.png`

#### 6. Code Token Coverage
- Create `analyze_code_tokens()` function with categories:
  - keywords, operators, brackets, indentation, comments, sql, api, other_code
- Use regex patterns for detection
- Create 2-panel figure (total code tokens, category breakdown)
- Save to `output/code_token_coverage.csv`
- Save chart as `code_token_coverage.png`

#### 7. JSON Handling Capabilities
- Create `analyze_json_tokens()` function with categories:
  - structural (`{}[],:`)
  - literals (`true`, `false`, `null`)
  - quotes, json_keywords, escape_sequences, special_json
- Check added_tokens for JSON-related special tokens
- Create grouped bar chart
- Save to `output/json_token_coverage.csv`
- Save chart as `json_capabilities.png`

#### 8. Function Call Handling Capabilities
- Create `analyze_function_call_tokens()` function with categories:
  - function_keywords, call_keywords, parameter_keywords
  - async_keywords, parentheses
  - special_function_tokens, tool_tokens
- Check added_tokens for tool/function call special tokens (important for AI agents)
- Create 2-panel figure (total tokens, tool/action tokens ranking)
- Save to `output/function_call_coverage.csv`
- Save chart as `function_call_capabilities.png`

#### 9. Comprehensive Summary
- Build unified DataFrame with all metrics:
  - Vocab Size, Tokens > 32 chars, English %, Indic %
  - Individual Indic languages (Hindi, Bengali, Tamil, Telugu)
  - Code %, JSON Tokens, Function Tokens, Tool/Action Tokens
- Create 6-panel comprehensive comparison figure:
  - Vocabulary Size
  - Indic Coverage %
  - Code Coverage %
  - Tokens > 32 chars %
  - Indic Language Breakdown
  - Tool/Action Tokens
- Save to `output/tokenizer_comprehensive_summary.csv`
- Save chart as `comprehensive_comparison.png`

#### 10. Final Summary Display
- Display formatted summary table
- Highlight key insights with emoji indicators:
  - 🏆 Largest Vocabulary
  - 🇮🇳 Best Indic Coverage
  - 💻 Best Code Coverage
  - 🔧 Most Tool/Action Tokens
  - 📏 Most Long Tokens
- List all generated output files and charts

### Visualization Guidelines

- Use `matplotlib` with `figsize` appropriate for each chart
- Apply color maps: `viridis`, `plasma`, `coolwarm`, `Blues`, `Purples`, `YlOrRd`, `Set2`, `Set3`, `tab20`
- Add value labels on bar charts
- Rotate x-axis labels 45° for readability
- Use `tight_layout()` and save with `dpi=150, bbox_inches='tight'`
- Include legends with `bbox_to_anchor` for stacked charts

### Output Files

Generate the following CSV files in `output/`:
- `language_distribution.csv`
- `indic_english_coverage.csv`
- `code_token_coverage.csv`
- `json_token_coverage.csv`
- `function_call_coverage.csv`
- `tokenizer_comprehensive_summary.csv`

Generate the following PNG charts in `charts/`:
- `vocab_sizes.png`
- `max_token_lengths.png`
- `tokens_over_32.png`
- `language_distribution.png`
- `indic_english_coverage.png`
- `code_token_coverage.png`
- `json_capabilities.png`
- `function_call_capabilities.png`
- `comprehensive_comparison.png`

## Markdown Header

Include a markdown cell at the beginning with:
- Title: "Candidate Tokenizer Analysis"
- Description of the notebook's purpose
- Table listing candidate tokenizers with HuggingFace links

## Sample Tokenizer Models to Analyze

| Model | Example HuggingFace Link |
|-------|--------------------------|
| DeepSeek V3.2 | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 |
| DeepSeek Coder 33B | https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct |
| Qwen3 235B | https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 |
| Qwen3 Coder 480B | https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct |
| Gemma 2 27B | https://huggingface.co/google/gemma-2-27b |
| GPT-OSS 120B | https://huggingface.co/openai/gpt-oss-120b |
| OLMo 3 32B | https://huggingface.co/allenai/Olmo-3-1125-32B |
| Mistral Large 3 675B | https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512 |
| ByteDance Seed 36B | https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base |

## Key Technical Notes

1. **Byte-Level BPE Decoding**: Many tokenizers use byte-level BPE encoding. The `create_byte_decoder()` function creates a mapping from GPT-2 style encoded characters back to bytes, which is essential for correctly interpreting token content.

2. **Unicode Script Detection**: Use code point ranges to detect scripts. Handle edge cases like byte fragments (tokens with `�` characters) and whitespace/control characters.

3. **Added Tokens Analysis**: The `added_tokens` section often contains special tokens for function calling, tool use, and structured output. These are critical for evaluating a tokenizer's agent/tool-use capabilities.

4. **Percentage Calculations**: Always handle division by zero when calculating percentages.

5. **Token Length Analysis**: Analyze tokens with length > 32 characters as these may impact context window efficiency.
