# Validate Vocabulary Notebook

## Objective
Create a Jupyter notebook that validates the `unified_128k_tokenizer.json` by testing its performance on English, Indic languages, JSON structured data, and code. The notebook should measure tokenization efficiency and verify correct roundtrip encoding/decoding.

## Input Files
- `output/unified_128k_tokenizer.json` - The unified HuggingFace-compatible tokenizer to validate
- `output/unified_128k_vocab.json` - Vocabulary metadata with token classifications

## Output Files
- `charts/vocab_composition.png` - Bar chart of vocabulary distribution by category
- `charts/english_assessment.png` - English tokenization analysis charts
- `charts/hindi_assessment.png` - Hindi tokenization analysis charts
- `charts/code_assessment.png` - Code tokenization analysis charts
- `output/validation_summary.txt` - Text summary of validation results

## External Data Sources
- `wikimedia/wikipedia` dataset (20231101.en) - English Wikipedia for English assessment
- `wikimedia/wikipedia` dataset (20231101.hi) - Hindi Wikipedia for Indic assessment

## Notebook Structure

### Section 1: Imports and Setup (Cell 1)
**Purpose**: Set up the environment and configuration

**Requirements**:
- Import: json, re, time, pathlib.Path, collections.Counter/defaultdict, unicodedata
- Import: matplotlib.pyplot, numpy
- Configuration paths:
  - `BASE_PATH`, `OUTPUT_PATH`
  - `TOKENIZER_FILE` = output/unified_128k_tokenizer.json
  - `VOCAB_METADATA_FILE` = output/unified_128k_vocab.json
- Print header and configuration

### Section 2: Load Tokenizer and Vocabulary (Cell 2)
**Purpose**: Load the tokenizer and metadata

**Requirements**:
- Import `Tokenizer` from `tokenizers` library
- Load tokenizer using `Tokenizer.from_file()`
- Get vocabulary and vocab_size
- Load vocabulary metadata JSON
- Extract token_list from metadata
- Print loading status with statistics

### Section 3: Pre-tokenizer Diagnostic (Cell 3)
**Purpose**: Quick sanity check that tokenizer is working correctly

**Requirements**:
- Define test inputs covering various scenarios:
  - Single word: "hello"
  - Two words: "hello world"
  - With punctuation: "hello, world!"
  - JSON snippet: '{"key": "value"}'
  - Hindi word: "नमस्ते"
  - Hindi sentence: "यह एक परीक्षण है"
  - Code snippet: "def foo(): pass"
  - Numbers: "12345"
  - Mixed: "Hello नमस्ते 123"
- For each input:
  - Encode with tokenizer
  - Count tokens
  - Decode and check roundtrip
  - Print status (✓ or ❌)
- Print summary of pre-tokenizer behavior

### Section 4: Vocabulary Composition Analysis (Cell 4)
**Purpose**: Analyze token distribution by category from metadata

**Requirements**:
- Extract `category_breakdown` from vocab_metadata
- Print table with category, count, and percentage
- Create horizontal bar chart of top 12 categories
- Save chart to `charts/vocab_composition.png`

### Section 5: Language Creep Analysis (Cell 5)
**Purpose**: Check for unwanted language tokens (CJK, Arabic, etc.)

**Requirements**:
- Define `SCRIPT_RANGES` dict with Unicode ranges for:
  - Latin/ASCII, Devanagari, Bengali, Tamil, Telugu, Kannada
  - Malayalam, Gujarati, Gurmukhi, Odia
  - CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Korean
  - Japanese Hiragana, Japanese Katakana
- Function `get_token_script(token_str)` to classify token by primary script
- Analyze all tokens from token_list
- Print distribution table with samples
- Flag if any unwanted scripts have >100 tokens

### Section 6: English Assessment - Data Loading (Cell 6)
**Purpose**: Load and process English Wikipedia dataset

**Requirements**:
- Use `datasets.load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)`
- Process NUM_SAMPLES = 500 articles
- Track statistics:
  - total_chars, total_words, total_tokens
  - unique_tokens set
  - word_token_counts (tokens per word samples)
  - token_freq Counter
- For each article:
  - Tokenize full text
  - Sample word-level tokenization (every 100th word)
- Print progress every 100 articles

### Section 7: English Assessment - Results (Cell 7)
**Purpose**: Display English tokenization results

**Requirements**:
- Calculate and print:
  - Tokens per word
  - Characters per token
  - Vocabulary utilization (unique tokens / vocab_size)
- Show tokens per word distribution
- List top 15 most frequent tokens
- Assign rating (Excellent if <1.5 tokens/word, Good if <2.0)

### Section 8: English Assessment - Visualization (Cell 8)
**Purpose**: Create English assessment charts

**Requirements**:
- Create 2-subplot figure:
  - Left: Token distribution per word (bar chart)
  - Right: Top 10 most frequent tokens (pie chart)
- Save to `charts/english_assessment.png`

### Section 9: Hindi Assessment - Data Loading (Cell 9)
**Purpose**: Load and process Hindi Wikipedia dataset

**Requirements**:
- Use `datasets.load_dataset("wikimedia/wikipedia", "20231101.hi", streaming=True)`
- Process NUM_SAMPLES_HI = 300 articles
- Define DEVANAGARI_PATTERN regex for Hindi words: `[\u0900-\u097F]+`
- Track statistics similar to English
- Track hindi_words set and word_token_map

### Section 10: Hindi Assessment - Results (Cell 10)
**Purpose**: Display Hindi tokenization results

**Requirements**:
- Calculate and print:
  - Tokens per word
  - Characters per token
  - Single-token words percentage
- Show tokens per Hindi word distribution
- List sample Hindi words tokenized as single tokens
- Assign rating (Excellent if <2.0, Good if <3.0, Fair if <4.0)

### Section 11: Hindi Assessment - Visualization (Cell 11)
**Purpose**: Create Hindi assessment charts

**Requirements**:
- Create 2-subplot figure:
  - Left: Token distribution per Hindi word (bar chart)
  - Right: Comparison English vs Hindi tokens/word (bar chart)
- Save to `charts/hindi_assessment.png`

### Section 12: JSON Assessment (Cell 12)
**Purpose**: Test tokenizer on JSON structured data

**Requirements**:
- Define json_samples list with various JSON structures:
  - Simple object: `{"name": "John", "age": 30}`
  - Nested object
  - Array
  - Complex API response
  - Function calling format
  - Tool definition
- For each sample:
  - Encode and count tokens
  - Calculate chars per token
- Print table with results
- Check structural tokens (`{`, `}`, `[`, `]`, `:`, `,`) tokenize as single tokens
- Calculate and print average chars/token for JSON

### Section 13: JSON Token Breakdown (Cell 13)
**Purpose**: Show detailed token breakdown for a JSON example

**Requirements**:
- Take a complex JSON example
- Encode and print token-by-token breakdown
- Show token ID, raw token, and decoded form
- Verify roundtrip encoding

### Section 14: Code Assessment - Python & JavaScript (Cell 14)
**Purpose**: Test tokenizer on programming code

**Requirements**:
- Define python_samples list with:
  - Simple function
  - Class definition
  - Async function
  - Complex logic (fibonacci)
- Define js_samples list with:
  - Arrow function
  - Async/await
  - Class
- For each sample:
  - Count lines, chars, tokens
  - Calculate chars per token
- Print tables for Python and JavaScript separately
- Calculate averages

### Section 15: Code Keyword & Operator Analysis (Cell 15)
**Purpose**: Check if keywords and operators tokenize as single tokens

**Requirements**:
- Define python_keywords list (def, class, if, else, for, while, try, except, etc.)
- Define js_keywords list (function, const, let, var, async, await, class, etc.)
- Define operators list (==, !=, ===, !==, <=, >=, &&, ||, =>, ->, etc.)
- For each category:
  - Check how many tokenize as single tokens
  - Print percentage
  - List any multi-token keywords/operators

### Section 16: Code Assessment - Visualization (Cell 16)
**Purpose**: Create code assessment charts

**Requirements**:
- Create 2-subplot figure:
  - Left: Characters per token across all domains (English, Hindi, JSON, Python, JS)
  - Right: Single-token rate for Python keywords, JS keywords, Operators
- Save to `charts/code_assessment.png`

### Section 17: Comprehensive Summary (Cell 17)
**Purpose**: Print summary of all assessments

**Requirements**:
- Print vocabulary composition summary
- Print English assessment with rating
- Print Hindi assessment with rating
- Print JSON assessment with rating
- Print Code assessment with rating
- Assign overall validation status

### Section 18: Final Summary and Conclusions (Cell 18)
**Purpose**: Save final validation summary

**Requirements**:
- Print comprehensive summary including:
  - Vocabulary composition stats
  - Tokenization efficiency for each domain
  - Overall validation status
- Create summary_text string
- Save to `output/validation_summary.txt`

## Key Metrics to Evaluate

### English Assessment
- **Tokens per word**: Target < 1.5 (Excellent), < 2.0 (Good)
- **Characters per token**: Higher is better (~4-5 expected)
- **Vocabulary utilization**: Percentage of vocab used

### Hindi/Indic Assessment
- **Tokens per word**: Target < 2.5 (Excellent for non-Latin)
- **Single-token words**: Percentage of unique words as single tokens
- **Roundtrip accuracy**: Encode → Decode should match

### JSON Assessment
- **Structural tokens**: All should be single tokens
- **Characters per token**: Higher indicates efficient encoding
- **Roundtrip accuracy**: Must pass for structured output use

### Code Assessment
- **Keyword recognition**: 100% as single tokens ideal
- **Operator recognition**: 90%+ as single tokens
- **Characters per token**: Higher is better for code

## Rating Scale
- ✅ **Excellent**: Exceeds expectations
- ✅ **Good**: Meets requirements
- ⚠️ **Fair**: Acceptable but could improve
- ❌ **Poor**: Needs attention

## Dependencies
```python
# Required packages
tokenizers  # HuggingFace tokenizers
datasets    # HuggingFace datasets for Wikipedia
matplotlib  # Visualization
numpy       # Numerical operations
```
