# Unified Vocabulary Generation Notebook

## Objective
Create a Jupyter notebook that generates a unified 128K tokenizer from three source tokenizers (GPT-OSS, OLMo, Qwen), combining their vocabularies to optimize for English, Indic language support, and Code/JSON handling.

## Design Principles
1. **Position-based ordering**: Token position takes priority over model count
   - A token at position 300 (in 2 models) ranks higher than a token at position 10000 (in 3 models)
   - This preserves the natural ordering that tokenizers learned during training
2. **GPT-OSS as base ordering**: Use GPT-OSS token positions as the foundation
   - Tokens not in GPT-OSS are inserted based on their position in OLMo/Qwen
3. **Indic priority**: GPT-OSS has best Indic support; these tokens are preserved at their positions
4. **Code & JSON preserved**: Structural tokens and programming constructs must be maintained
5. **Merge consistency**: Only include BPE merges where both input tokens AND the merged result exist in the unified vocabulary

## Input Files
- `data/gptoss_tokenizer.json` - GPT-OSS tokenizer (best Indic support, ~200K tokens)
- `data/olmo_tokenizer.json` - OLMo tokenizer (open dataset, ~100K tokens)
- `data/qwen_tokenizer.json` - Qwen tokenizer (strong general vocab, ~150K tokens)

## Output Files
- `output/unified_128k_tokenizer.json` - Full HuggingFace-compatible tokenizer
- `output/unified_128k_vocab.json` - Detailed vocabulary with metadata
- `output/unified_128k_vocab_simple.json` - Simple token → id mapping

## Notebook Structure

### Section 1: Imports and Setup (Cell 1)
**Purpose**: Set up the environment and define configuration

**Requirements**:
- Import: json, os, copy, re, pathlib.Path, collections.defaultdict/Counter, numpy
- Configuration constants:
  - `TARGET_VOCAB_SIZE = 128000`
  - `BASE_PATH`, `DATA_PATH`, `OUTPUT_PATH`
  - `SOURCE_TOKENIZERS` dict mapping names to filenames
- GPT-2 style byte decoder/encoder functions:
  - `create_byte_decoder()` - Maps Unicode chars to byte values
  - `create_byte_encoder()` - Inverse mapping
  - `decode_bpe_token(token)` - Decode BPE token to actual string
  - `encode_to_bpe(text)` - Encode text to BPE representation
- Print configuration summary

### Section 2: Load Source Tokenizers (Cell 2)
**Purpose**: Load all three tokenizers and extract vocabularies and merges

**Requirements**:
- Iterate through `SOURCE_TOKENIZERS`
- Load JSON files from `data/` directory
- Extract `vocab` from `data['model']['vocab']`
- Extract `merges` from `data['model']['merges']`
- Track vocab sizes for each tokenizer
- Set GPT-OSS as `BASE_TOKENIZER` (best Indic support)
- Print loading status with token counts and merge counts

### Section 3: Token Classification Functions (Cell 3)
**Purpose**: Define comprehensive token classification functions

**Requirements**:
- Unicode ranges for Indic scripts:
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
- CJK Unicode ranges for Chinese token detection
- Functions to implement:
  - `get_script_type(char)` - Determine script type of a character (ASCII, Indic script, CJK, Other)
  - `classify_indic_script(text)` - Return specific Indic script name or None
  - `is_english_token(text)` - Check if token is primarily English (>70% ASCII alphanumeric)
  - `is_code_token(token)` - Detect programming-related tokens:
    - Keywords: def, class, function, if, else, for, while, try, except, return, etc.
    - Operators: =>, ->, ::, &&, ||, ==, !=, etc.
    - Brackets: {}, [], (), ;
    - camelCase and snake_case patterns
    - SQL keywords, HTTP methods
  - `is_json_token(token)` - Detect JSON-related tokens:
    - Structural: {, }, [, ], :, ,
    - Literals: true, false, null
    - JSON methods: stringify, parse, dumps, loads
  - `normalize_token(token)` - Normalize for cross-model comparison:
    - Decode BPE, strip whitespace
    - Remove leading space markers (Ġ, ▁, space)
    - Convert to lowercase
  - `classify_token(token)` - Comprehensive classification returning:
    - (category, is_indic, is_code, is_json, priority)
    - Priority values: 1=special, 2=code/json, 3=indic, 4=english, 5=other, 6=CJK

### Section 4: Build Token Registry (Cell 4)
**Purpose**: Create a cross-tokenizer registry tracking all tokens

**Requirements**:
- `token_registry`: normalized_token → {model: token_info}
- `raw_token_registry`: raw_token → {model: token_id}
- For each tokenizer, for each token:
  - Decode and normalize token
  - Classify token (category, is_indic, is_code, is_json, priority)
  - Calculate normalized_id = token_id / vocab_size
  - Store in registries
- Track per-model statistics: total, english, indic, code_json, cjk, other
- Print summary statistics per model

### Section 5: Analyze Token Distribution (Cell 5)
**Purpose**: Analyze token overlap and distribution

**Requirements**:
- Build analysis_rows list with entries containing:
  - normalized_token, raw_token
  - model_count (how many models have this token)
  - models list
  - category, is_indic, is_code, is_json
  - priority (minimum across models = highest priority)
  - avg_norm_id (mean of normalized IDs across models)
  - model_data (full data from each model)
- For raw_token: prefer GPT-OSS, then OLMo, then Qwen
- Sort by: model_count (desc), priority (asc), avg_norm_id (asc)
- Print distribution analysis:
  - Tokens by model count (3, 2, 1)
  - Category breakdown (top 15)
  - Indic tokens by source tokenizer

### Section 6: Token Selection Strategy (Cell 6)
**Purpose**: Select tokens for unified vocabulary using position-based strategy

**Algorithm**:
1. Use GPT-OSS as base ordering - tokens in GPT-OSS use their original position
2. For tokens NOT in GPT-OSS, calculate effective position from OLMo/Qwen (use the lower normalized position)
3. Sort all candidates by effective_position (ascending)
4. Exclude CJK tokens to make room for more useful tokens
5. Select top TARGET_VOCAB_SIZE tokens

**Key Insight**: Position takes priority over model count
- Token at position 300 (in 2 models) > Token at position 10000 (in 3 models)
- This preserves the natural ordering that tokenizers learned during training

**Requirements**:
- Calculate `effective_position` for each token:
  ```python
  if 'GPT-OSS' in model_data:
      effective_position = gptoss_token_id / gptoss_vocab_size
  else:
      # Use lowest normalized position from OLMo/Qwen
      effective_position = min(olmo_norm_id, qwen_norm_id)
  ```
- Track `source` (which tokenizer determined the position)
- Sort by: (effective_position, indic_bonus, -model_count)
  - indic_bonus = 0 for Indic tokens, 0.0001 for others (tie-breaker)
- Print position distribution analysis
- Print source distribution (how many from GPT-OSS vs OLMo vs Qwen order)
- Print model coverage stats (in 3, 2, 1 models)

### Section 7: Create Unified Vocabulary (Cell 7)
**Purpose**: Create final vocabulary mapping with metadata

**Requirements**:
- `unified_vocab`: raw_token → new_token_id
- `unified_metadata`: list of entries with:
  - token (decoded), raw_token, token_id (new)
  - normalized_form, category
  - is_indic, is_code, is_json
  - model_count, source_token_ids (per-model)
  - effective_position, position_source (new fields)
  - avg_position (rounded to 6 decimals)
- Calculate final statistics: total, common_3, indic, code, json
- Print composition and category breakdown

### Section 8: Generate Consistent Merges (Cell 8)
**Purpose**: Filter BPE merges for consistency

**Requirements**:
- Use GPT-OSS merges as base (best for Indic)
- `parse_merge(merge)` function - handle both list and string formats
- For each merge, check:
  - Both tokens exist in unified_vocab
  - Merged result (token1 + token2) exists in unified_vocab
- Track merge_stats: total, valid, missing_tokens, missing_result
- Keep only valid merges as `consistent_merges`

### Section 9: Supplement Merges (Cell 9)
**Purpose**: Add valid merges from OLMo and Qwen

**Requirements**:
- Track existing merges as set of (token1, token2) tuples
- For each supplementary tokenizer (OLMo, Qwen):
  - Find valid merges not already included
  - Apply same consistency checks
  - Add to additional_merges list
- Combine: `all_merges = consistent_merges + additional_merges`
- Print counts per source

### Section 10: Build Tokenizer Structure (Cell 10)
**Purpose**: Create complete HuggingFace-compatible tokenizer

**Requirements**:
- Deep copy base tokenizer structure
- Update `model.vocab` with unified_vocab
- Update `model.merges` with all_merges
- Filter `added_tokens` to only include tokens in unified_vocab
  - Update token IDs to new mapping
- Print structure summary

### Section 11: Save Tokenizer (Cell 11)
**Purpose**: Save unified tokenizer in HuggingFace format

**Output**: `output/unified_128k_tokenizer.json`
- Full tokenizer structure with vocab and merges
- Print file path and size

### Section 12: Save Vocabulary with Metadata (Cell 12)
**Purpose**: Save detailed vocabulary with full metadata

**Output**: `output/unified_128k_vocab.json`
- Structure:
  ```json
  {
    "name": "Unified 128K Vocabulary",
    "description": "...",
    "version": "1.0",
    "vocab_size": 128000,
    "source_tokenizers": ["GPT-OSS", "OLMo", "Qwen"],
    "base_tokenizer": "GPT-OSS",
    "statistics": {...},
    "category_breakdown": {...},
    "tokens": [unified_metadata list]
  }
  ```
- Print file path and size

### Section 13: Save Simple Mapping (Cell 13)
**Purpose**: Save simple token → id mapping

**Output**: `output/unified_128k_vocab_simple.json`
- Simple dict: decoded_token → token_id
- Print file path and size

### Section 14: Verification (Cell 14)
**Purpose**: Verify generated tokenizer integrity

**Requirements**:
- Reload tokenizer from file
- Verify vocab size and merge count
- Merge consistency check (sample first 1000):
  - Both tokens exist in vocab
  - Merged result exists in vocab
- Print sample tokens at various indices (0, 1, 2, 100, 1000, 10000, 50000, 100000, last)
- Show token string, category, and model count

### Section 15: Final Summary (Cell 15)
**Purpose**: Display comprehensive generation summary

**Requirements**:
- Print source tokenizer info (name, size, filename)
- Unified tokenizer stats (vocab size, merges, base)
- Token composition (common, indic, code, json)
- Output file paths with descriptions
- File sizes for all outputs
- Success message

## Expected Output Summary
```
Source Tokenizers:
  • GPT-OSS: ~200K tokens
  • OLMo: ~100K tokens
  • Qwen: ~150K tokens

Unified Tokenizer:
  • Vocabulary size: 128,000
  • Consistent merges: ~19K
  • Base structure: GPT-OSS

Source Distribution:
  • From GPT-OSS order: ~115K
  • From OLMo order: ~10
  • From Qwen order: ~12.5K

Model Coverage:
  • In all 3 models: ~44K
  • In 2 models: ~21K
  • In 1 model: ~63K

Token Composition:
  • Indic tokens: ~11K
  • Code tokens: ~1.6K
  • JSON tokens: ~54
```

## Technical Notes
- Uses GPT-2 style byte encoding/decoding for BPE tokens
- Normalized token ID = token_id / vocab_size (0-1 scale for cross-model comparison)
- **Position-based ordering**: Tokens at lower positions are prioritized regardless of model count
- GPT-OSS order is used as the base; tokens not in GPT-OSS use their OLMo/Qwen position
- Indic tokens get a small tie-breaker bonus to ensure preservation
- Merge consistency is critical for proper tokenization behavior
- HuggingFace format compatibility maintained for direct use with transformers library
