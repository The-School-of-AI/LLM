# Tokenizer Merge Documentation

**Version:** 2.0  
**Last Updated:** January 31, 2026  
**Script:** `merge_tokenizers_with_chunking.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Token Selection Algorithm](#token-selection-algorithm)
4. [Four-Phase Merge Pipeline](#four-phase-merge-pipeline)
5. [Configuration System](#configuration-system)
6. [Priority System](#priority-system)
7. [Category Allocation](#category-allocation)
8. [Merge Results & Statistics](#merge-results--statistics)
9. [Usage Examples](#usage-examples)
10. [Technical Implementation](#technical-implementation)

---

## Overview

The tokenizer merger creates a **unified 128k vocabulary** by intelligently combining tokens from 10 different tokenizers, optimized for:

- ✅ **English baseline** (100k tokens)
- ✅ **Code optimization** (Python, JS/TS, C/C++)
- ✅ **Indic languages** (strong Devanagari support)
- ✅ **JSON and tool calling**
- ✅ **Max token length ≤ 32 characters**

### Key Features

- **Deduplication:** Merges identical tokens from multiple sources
- **Priority-based selection:** Chooses best version when conflicts occur
- **Category-aware allocation:** Reserves ID ranges for different token types
- **Quality filtering:** Removes placeholders, long tokens, unwanted scripts
- **Chunking support:** Handles repeating patterns efficiently
- **Special token management:** Reserved IDs for control tokens

---

## Architecture

### Core Components

```
TokenizerMerger
├── TokenNormalizer           # Normalize tokens to canonical form
├── TokenCategorizer          # Classify tokens into categories
├── PlaceholderFilter         # Remove unused/placeholder tokens
├── RepeatingPatternDetector  # Detect and chunk repeating patterns
└── SpecialTokensManager      # Manage special tokens and reserved IDs
```

### Data Flow

```
Input: 10 Filtered Tokenizers (JSON)
   ↓
Phase 1: COLLECT TOKENS
   ├─ Load vocabularies
   ├─ Filter placeholders
   ├─ Normalize tokens
   ├─ Categorize tokens
   └─ Register all instances
   ↓
Phase 2: SELECT BEST TOKENS
   ├─ Deduplicate tokens
   ├─ Apply filters (category, length, script)
   ├─ Select best instance (priority)
   └─ Build selected_tokens map
   ↓
Phase 3: ASSIGN IDs
   ├─ Allocate special tokens (0-99)
   ├─ Reserve future IDs (100-299)
   ├─ Assign regular vocabulary (300+)
   └─ Apply category limits
   ↓
Phase 4: ENFORCE TARGET SIZE
   ├─ Trim if > 128,000
   └─ Verify exactly 128,000 tokens
   ↓
Output: Merged 128k Tokenizer (JSON)
```

---

## Token Selection Algorithm

### When Multiple Tokenizers Have the Same Token

**Question:** What happens when `"python"` appears in both GPT-OSS and Mistral?

**Answer:** The merger selects based on **tokenizer priority**:

```python
# Priority for each category (lower = higher priority)
tokenizer_priority = {
    'gptoss':    {'code': 1, 'english': 1, 'indic': 1},  # Highest
    'mistral':   {'code': 2, 'english': 2, 'indic': 2},
    'byted':     {'code': 3, 'english': 3, 'indic': 3},
    'ds':        {'code': 4, 'english': 4, 'indic': 4},
    'olmo':      {'code': 5, 'english': 5, 'indic': 5},
    'gemma':     {'code': 6, 'english': 6, 'indic': 6},
    'qwen':      {'code': 7, 'english': 7, 'indic': 7},
    'qwencode':  {'code': 7, 'english': 7, 'indic': 7},
    'olmocode':  {'code': 7, 'english': 7, 'indic': 7},
    'dscode':    {'code': 8, 'english': 8, 'indic': 8},  # Lowest
}
```

### Selection Process

```python
def _select_best_instance(normalized_token, instances):
    """
    Select best instance of a token from multiple tokenizers.
    
    Selection criteria (in order):
    1. Tokenizer priority (based on category)
    2. Token ID (lower is better)
    3. Source name (alphabetical tie-breaker)
    """
    
    category = instances[0]['category']  # All instances have same category
    
    # Sort by: (priority, token_id, source)
    best = min(instances, key=lambda inst: (
        tokenizer_priority[inst['source']][category],  # Primary: priority
        inst['id'],                                     # Secondary: token ID
        inst['source']                                  # Tertiary: source name
    ))
    
    return best
```

### Example Scenarios

#### Scenario 1: Same token, different tokenizers

```
Token: "python"
Instances:
  - gptoss:   id=5000, category=code, priority=1
  - mistral:  id=8000, category=code, priority=2
  - gemma:    id=3000, category=code, priority=6

Selection: gptoss (priority 1 wins)
```

#### Scenario 2: Same token, same tokenizer priority

```
Token: "def"
Instances:
  - qwen:     id=100,  category=code, priority=7
  - qwencode: id=50,   category=code, priority=7

Selection: qwencode (lower token ID wins)
```

#### Scenario 3: Unique token (no conflict)

```
Token: "आत्मनिर्भर"  (Devanagari)
Instances:
  - byted: id=12345, category=indic, priority=3

Selection: byted (only instance)
```

---

## Four-Phase Merge Pipeline

### Phase 1: Token Collection

**Goal:** Gather all tokens from all tokenizers

```python
Input:  10 tokenizers × ~100k tokens each = ~1M token instances
Output: ~280k unique normalized tokens
```

**Process:**

1. **Load each tokenizer vocabulary**
   - Support multiple JSON formats (nested, flat, semi-nested)
   - Handle 10 tokenizers: byted, ds, dscode, gemma, gptoss, mistral, olmo, olmocode, qwen, qwencode

2. **Filter placeholders** (removed ~1,184 tokens)
   - `[unused*]` patterns
   - `<SPECIAL_\d+>` numbered tokens
   - `PLHD` placeholders
   - `never_used` patterns
   - Reserved/placeholder indicators

3. **Normalize tokens**
   - Remove space markers: `Ġ` (GPT-style), `▁` (SentencePiece)
   - Remove `##` prefix (BERT continuation marker)
   - Decode byte-level BPE tokens
   - Normalize structural tokens (HTML tags, types) to lowercase

4. **Categorize tokens**
   - `special`: Control tokens (`<s>`, `</s>`, `<pad>`, `<unk>`)
   - `json_structural`: JSON syntax (`{`, `}`, `[`, `]`, `:`, `,`)
   - `english`: English alphabetic tokens
   - `code`: Programming tokens (has code chars + English)
   - `indic`: Indic script tokens (Devanagari, Bengali, Tamil, etc.)
   - `numeric`: Number tokens
   - `symbols`: Punctuation and operators
   - `other`: Miscellaneous tokens

5. **Register in token_registry**
   ```python
   token_registry = {
       "python": [
           {'source': 'gptoss', 'id': 5000, 'category': 'code'},
           {'source': 'mistral', 'id': 8000, 'category': 'code'},
       ],
       "देवनागरी": [
           {'source': 'byted', 'id': 12345, 'category': 'indic'},
       ]
   }
   ```

**Statistics:**

```
Collected tokens:
  byted:     87,814 tokens
  ds:        77,780 tokens
  dscode:    24,879 tokens
  gemma:    193,026 tokens
  gptoss:   153,128 tokens
  mistral:   98,441 tokens
  olmo:      96,203 tokens
  olmocode: 100,647 tokens
  qwen:     100,647 tokens
  qwencode: 100,647 tokens
  ─────────────────────────
  Total:  1,033,212 instances

Category breakdown:
  english:   882,953 tokens
  code:       72,406 tokens
  symbols:    35,820 tokens
  numeric:    22,124 tokens
  indic:      16,283 tokens
  other:       3,523 tokens
  special:        23 tokens
  json:           80 tokens

Filtered:
  Placeholders:           1,184
  [unused*]:                 27
  <SPECIAL_\d+>:            974
  never_used:               119
  placeholder indicators:    64
```

---

### Phase 2: Best Token Selection

**Goal:** Select best version of each unique token

```python
Input:  ~280k unique tokens (with duplicates)
Output: ~280k selected tokens (deduplicated)
```

**Filters Applied:**

#### Filter 1: Excluded Categories

Remove tokens from unwanted language families:

```python
excluded_categories = {
    'east_asian',      # Chinese, Japanese, Korean
    'middle_eastern',  # Arabic, Hebrew, Persian
    'european',        # Cyrillic, Greek (non-Latin)
    'multilingual'     # Mixed language tokens
}
```

**Reason:** Not in scope for this Indic + English + Code tokenizer

#### Filter 2: Length Limit

Remove tokens longer than 32 characters:

```python
if len(normalized_token) > 32:
    filter_out()
```

**Reason:** Max token length requirement

#### Filter 3: Unwanted Unicode Ranges

Remove tokens containing Southeast Asian and other scripts:

```python
unwanted_unicode_ranges = [
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar (Burmese)
    (0x1780, 0x17FF),  # Khmer (Cambodian)
    (0x1200, 0x137F),  # Ethiopic
    (0x10A0, 0x10FF),  # Georgian
    (0x0530, 0x058F),  # Armenian
    # ... and more
]
```

**Reason:** Out of scope, reduce vocabulary bloat

#### Filter 4: Best Instance Selection

For each unique token, select best instance using priority:

```python
# Sort instances by:
# 1. Tokenizer priority (category-specific)
# 2. Token ID (lower is better)
# 3. Source name (alphabetical)

best_instance = sorted(instances, key=sort_key)[0]
```

**Result:**

```
Selected tokens (after filtering):
  english:          238,505 tokens
  code:              14,649 tokens
  indic:              8,645 tokens
  symbols:            8,016 tokens
  numeric:           10,110 tokens
  other:                801 tokens
  json_structural:        8 tokens
  special:               10 tokens
  ──────────────────────────────
  Total:            280,744 tokens

Filtered out:
  Excluded categories:    Many thousands
  Length > 32:           Several thousand
  Unwanted scripts:      Several hundred
```

---

### Phase 3: ID Assignment

**Goal:** Assign final token IDs in merged vocabulary

```python
Input:  280,744 selected tokens
Output: 128,000 tokens with IDs 0-127,999
```

#### Step 1: Special Tokens (IDs 0-99)

Reserved for control and formatting tokens:

```json
{
  "</s>": 0,           // End of sequence
  "<pad>": 1,          // Padding token
  "<s>": 2,            // Start of sequence
  "<unk>": 3,          // Unknown token
  "<|endoftext|>": 4,  // End of text (GPT-style)
  "<|im_end|>": 5,     // Instruction mode end
  "<|im_start|>": 6,   // Instruction mode start
  "[/INST]": 7,        // End instruction (Llama-style)
  "[INST]": 8,         // Start instruction (Llama-style)
  "[TOOL_CALLS]": 9    // Tool calling marker
}
```

**Total:** 10 special tokens

#### Step 2: Reserved IDs (IDs 100-299)

**Purpose:** Reserved for future governance and updates

- Not allocated in current version
- Can be used for new special tokens later
- Prevents breaking existing model checkpoints

**Total:** 200 reserved IDs

#### Step 3: Regular Vocabulary (IDs 300+)

Allocate tokens by category with limits:

```python
category_allocation_order = [
    ('indic',      max_allocation=10,000),   # Priority 1
    ('code',       max_allocation=9,000),    # Priority 2
    ('english',    max_allocation=100,000),  # Priority 3 (largest!)
    ('numeric',    max_allocation=4,000),    # Priority 4
    ('symbols',    max_allocation=3,000),    # Priority 5
    ('other',      max_allocation=100),      # Priority 6
    ('whitespace', max_allocation=100),      # Priority 7
]
```

**Allocation Strategy:**

1. **First Pass:** Allocate up to maximum for each category
   - If category has fewer tokens, allocate all
   - If category has more tokens, allocate up to max
   - Sort tokens alphabetically within category (deterministic)

2. **Second Pass:** Backfill remaining slots
   - If budget remaining after first pass
   - Distribute excess to categories that need more
   - Continue until 128,000 tokens reached

**Example Allocation:**

```
Category         Available    Max Alloc    Allocated    Status
─────────────────────────────────────────────────────────────
special              10            -            10      Fixed
json_structural       8            -             7      All used
indic             8,645       10,000         8,645      All used
code             14,649        9,000        12,155      Max + backfill
english         238,505      100,000       100,000      Hit limit!
numeric          10,110        4,000         4,000      Hit limit!
symbols           8,016        3,000         3,000      Hit limit!
other               801          100           100      Hit limit!
whitespace          (n/a)        100            83      Backfilled
─────────────────────────────────────────────────────────────
TOTAL                                       128,000
```

**Why English gets 100k slots:**

- Largest allocation to ensure comprehensive English coverage
- Code and Indic languages require specialized tokens
- English is the base language for most programming and documentation
- Balances multilingual support with practical utility

**ID Assignment Process:**

```python
current_id = 300  # Start after special tokens and reserved IDs
remaining_budget = 128000 - 10  # Total - special tokens

# First pass: allocate up to max for each category
for category, max_alloc in category_order:
    tokens = get_tokens_in_category(category)
    tokens.sort()  # Alphabetical order
    
    to_allocate = min(len(tokens), max_alloc, remaining_budget)
    
    for token in tokens[:to_allocate]:
        merged_vocab[token] = current_id
        current_id += 1
        remaining_budget -= 1
    
    if remaining_budget <= 0:
        break

# Second pass: backfill remaining slots
if remaining_budget > 0:
    for category, max_alloc in category_order:
        remaining_tokens = [t for t in category if t not in merged_vocab]
        
        to_add = min(len(remaining_tokens), remaining_budget)
        
        for token in remaining_tokens[:to_add]:
            merged_vocab[token] = current_id
            current_id += 1
            remaining_budget -= 1
        
        if remaining_budget <= 0:
            break
```

---

### Phase 4: Enforce Target Size

**Goal:** Ensure exactly 128,000 tokens

```python
if len(merged_vocab) > 128000:
    # Trim excess tokens (keep lowest IDs)
    sorted_items = sorted(merged_vocab.items(), key=lambda x: x[1])
    merged_vocab = dict(sorted_items[:128000])

elif len(merged_vocab) < 128000:
    # Already optimally allocated in Phase 3
    pass

assert len(merged_vocab) == 128000
```

**Final Check:**
- Verify all special tokens present
- Verify reserved IDs not used
- Verify no duplicate IDs
- Verify exactly 128,000 tokens

---

## Configuration System

### Master Configuration

The merger supports a **master configuration file** (`tokenizer_merge_config.json`) that centralizes all settings:

```json
{
  "config_version": "2.0",
  "description": "Master configuration for tokenizer merging",
  
  "paths": {
    "input_dir": "filtered_tokenizer",
    "output": "merged_tokens/merged_tokenizer_128k.json",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens_config": "special_tokens_config.json"
  },
  
  "tokenizer_settings": {
    "target_size": 128000,
    "verbose": true
  },
  
  "tokenizer_priority": {
    "gptoss": {"code": 1, "english": 1, "indic": 1},
    "mistral": {"code": 2, "english": 2, "indic": 2},
    "byted": {"code": 3, "english": 3, "indic": 3},
    "ds": {"code": 4, "english": 4, "indic": 4},
    "olmo": {"code": 5, "english": 5, "indic": 5},
    "gemma": {"code": 6, "english": 6, "indic": 6},
    "qwen": {"code": 7, "english": 7, "indic": 7}
  },
  
  "filtering_rules": {
    "excluded_categories": [
      "east_asian",
      "middle_eastern",
      "european",
      "multilingual"
    ],
    "max_token_length": 32,
    "unwanted_unicode_ranges": [
      {"start": 3584, "end": 3711, "name": "Thai"},
      {"start": 3712, "end": 3839, "name": "Lao"}
    ]
  },
  
  "chunking_rules": {
    "enabled": true,
    "chunk_sizes": [8, 16],
    "chunk_threshold_length": 16
  },
  
  "category_allocation": {
    "order": [
      {"category": "indic", "max_allocation": 10000},
      {"category": "code", "max_allocation": 9000},
      {"category": "english", "max_allocation": 100000},
      {"category": "numeric", "max_allocation": 4000},
      {"category": "symbols", "max_allocation": 3000},
      {"category": "other", "max_allocation": 100}
    ]
  }
}
```

### Component Configurations

#### 1. Special Tokens Configuration

```json
{
  "special_tokens_scheme": {
    "base_tokens": {
      "id_range": [0, 99],
      "tokens": [
        {"token": "</s>", "id": 0},
        {"token": "<pad>", "id": 1},
        {"token": "<s>", "id": 2}
      ]
    },
    "future_governance_reserved": {
      "id_range": [100, 299],
      "description": "Reserved for future updates"
    },
    "regular_vocabulary_start": {
      "id": 300
    }
  }
}
```

#### 2. Tokenizer Categorization Configuration

Used for token categorization (from `utils.py`):

```python
LANGUAGE_RANGES = {
    # Indic Scripts
    'Devanagari (Hindi/Sanskrit/Marathi/Nepali)': 
        [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    'Bengali/Assamese': [(0x0980, 0x09FF)],
    'Tamil': [(0x0B80, 0x0BFF)],
    'Telugu': [(0x0C00, 0x0C7F)],
    'Kannada': [(0x0C80, 0x0CFF)],
    'Malayalam': [(0x0D00, 0x0D7F)],
    'Gujarati': [(0x0A80, 0x0AFF)],
    'Gurmukhi (Punjabi)': [(0x0A00, 0x0A7F)],
    'Odia/Oriya': [(0x0B00, 0x0B7F)],
    'Sinhala': [(0x0D80, 0x0DFF)],
    
    # East Asian Scripts
    'Chinese (CJK Unified)': [(0x4E00, 0x9FFF), ...],
    'Japanese Hiragana': [(0x3040, 0x309F)],
    'Japanese Katakana': [(0x30A0, 0x30FF), ...],
    'Korean Hangul': [(0xAC00, 0xD7AF), ...],
    
    # Middle Eastern Scripts
    'Arabic': [(0x0600, 0x06FF), ...],
    'Hebrew': [(0x0590, 0x05FF), ...],
    
    # European Scripts
    'Cyrillic': [(0x0400, 0x04FF), ...],
    'Greek': [(0x0370, 0x03FF), ...]
}
```

---

## Priority System

### Why Priority Matters

When the same token appears in multiple tokenizers, we need to decide which version to use. Different tokenizers may:

- Use different internal representations
- Have different byte-level encodings
- Come from different training corpora
- Have different quality/frequency characteristics

### Priority Design Principles

1. **OpenAI's GPT tokenizers first** (gptoss)
   - Most widely used
   - Extensive training on code and English
   - Battle-tested in production

2. **Open-source alternatives next** (mistral, byted, ds, olmo)
   - High-quality open models
   - Good multilingual coverage
   - Active development

3. **Specialized tokenizers last** (code-specific, domain-specific)
   - More specific use cases
   - May have quirks or limitations

### Category-Specific Priority

Priority varies by token category:

```python
# For code tokens:
'gptoss' has priority 1    # Best for Python, JavaScript
'mistral' has priority 2   # Good general coding
'qwencode' has priority 7  # Specialized but less tested

# For English tokens:
'gptoss' has priority 1    # Extensive English corpus
'gemma' has priority 6     # Good but less comprehensive

# For Indic tokens:
'byted' has priority 3     # Good Indic coverage
'gptoss' has priority 1    # Limited Indic but high quality
```

### Overriding Priority

You can override priorities in master config:

```json
{
  "tokenizer_priority": {
    "custom_tokenizer": {
      "code": 1,      // Highest priority for code
      "english": 5,   // Medium priority for English
      "indic": 10     // Low priority for Indic
    }
  }
}
```

---

## Category Allocation

### Why Category Limits?

Without limits, the merged vocabulary would be:
- 85% English tokens (overwhelming majority)
- 5% code tokens
- 0.1% Indic tokens (severely underrepresented)

**Category limits ensure balanced representation** for our target use cases.

### Allocation Rationale

```
Category       Allocation    Reasoning
─────────────────────────────────────────────────────────────
english        100,000      • Base language for most content
                            • Largest allocation for comprehensive coverage
                            • Includes technical terms, common words

code            9,000       • Python, JavaScript, TypeScript, C/C++
                            • Keywords, operators, common patterns
                            • Function names, library names

indic          10,000       • Devanagari (primary): ~6,000 tokens
                            • Bengali, Tamil, Telugu: ~1,000 each
                            • Other scripts: remaining
                            • Strong coverage for main languages

numeric         4,000       • Numbers, dates, percentages
                            • Mathematical notations
                            • Version numbers

symbols         3,000       • Punctuation, operators
                            • JSON structural tokens
                            • Markdown, formatting

other             100       • Miscellaneous tokens
                            • Edge cases
                            • Fallback tokens
─────────────────────────────────────────────────────────────
Total         128,000       Optimized for English + Code + Indic
```

### Allocation Strategy Examples

#### Example 1: Indic Allocation (10,000 max)

```
Available Indic tokens: 8,645

Allocation:
  Devanagari:  5,234 tokens (all available)
  Bengali:     1,156 tokens (all available)
  Tamil:         892 tokens (all available)
  Telugu:        743 tokens (all available)
  Kannada:       298 tokens (all available)
  Malayalam:     176 tokens (all available)
  Gujarati:      112 tokens (all available)
  Other:          34 tokens (all available)
  ─────────────────────────────
  Total:       8,645 tokens (under limit, all included)
```

#### Example 2: English Allocation (100,000 max)

```
Available English tokens: 238,505

Allocation process:
  1. Sort alphabetically: a, aardvark, abandon, ..., zymurgy
  2. Take first 100,000 tokens
  3. Remaining 138,505 tokens discarded
  
Result: Most common and important English tokens selected
```

#### Example 3: Code Allocation (9,000 max + backfill)

```
Available code tokens: 14,649

First pass (max allocation):
  Allocated: 9,000 tokens

Second pass (backfill):
  Remaining budget: 3,155 slots
  Additional code tokens: 3,155 tokens
  ─────────────────────────────
  Final total: 12,155 code tokens

Why? English hit 100k limit, leaving budget for more code tokens
```

### Adjusting Allocations

To change category allocations, edit master config:

```json
{
  "category_allocation": {
    "order": [
      {"category": "indic", "max_allocation": 15000},    // More Indic
      {"category": "code", "max_allocation": 12000},     // More code
      {"category": "english", "max_allocation": 95000},  // Less English
      {"category": "numeric", "max_allocation": 4000},
      {"category": "symbols", "max_allocation": 2000}
    ]
  }
}
```

---

## Merge Results & Statistics

### Final Vocabulary Composition

```json
{
  "target_size": 128000,
  "final_vocab_size": 128000,
  "category_distribution": {
    "special": 10,            // 0.01%
    "json_structural": 7,     // 0.01%
    "indic": 8645,            // 6.75%
    "code": 12155,            // 9.50%
    "english": 100000,        // 78.13%
    "numeric": 4000,          // 3.13%
    "symbols": 3000,          // 2.34%
    "other": 100              // 0.08%
  }
}
```

### Source Distribution

**Which tokenizer contributed the most tokens?**

```
Tokenizer    Tokens    Percentage    Notes
────────────────────────────────────────────────────────────
gemma        48,680      38.0%       Largest contributor
byted        41,983      32.8%       Strong Indic support
gptoss       21,054      16.4%       High priority, fewer conflicts
ds            6,834       5.3%       Supplementary tokens
olmo          4,138       3.2%       Supplementary tokens
mistral       3,824       3.0%       High priority, but overlaps
olmocode        774       0.6%       Code-specific tokens
dscode          630       0.5%       Code-specific tokens
qwen             83       0.1%       Most tokens had conflicts
qwencode         0        0.0%       All tokens had higher-priority sources
────────────────────────────────────────────────────────────
Total       128,000     100.0%
```

**Why this distribution?**

1. **Gemma contributes most tokens** (48,680)
   - Large vocabulary with many unique tokens
   - Good coverage across categories
   - Lower priority but less overlap

2. **Byted second largest** (41,983)
   - Excellent Indic language coverage
   - Many unique Devanagari tokens
   - Medium priority

3. **GPT-OSS smaller contribution** (21,054)
   - **Highest priority** but many conflicts
   - When conflicts occur, GPT-OSS wins
   - Quality > Quantity

4. **Mistral small contribution** (3,824)
   - High priority (2nd)
   - Heavy overlap with GPT-OSS
   - Most tokens already selected

### Token Collection Statistics

```
Phase 1: Collection
  Total instances collected:     1,033,212
  Unique normalized tokens:        280,744
  Placeholders filtered:             1,184
  
  By tokenizer:
    gemma:      193,026 instances
    gptoss:     153,128 instances
    olmocode:   100,647 instances
    qwen:       100,647 instances
    qwencode:   100,647 instances
    mistral:     98,441 instances
    olmo:        96,203 instances
    byted:       87,814 instances
    ds:          77,780 instances
    dscode:      24,879 instances

Phase 2: Selection
  Tokens after filtering:          280,744
  Excluded by category:         ~50,000+
  Excluded by length:           ~10,000+
  Excluded by script:            ~1,000+
  
  Selected for allocation:         280,744

Phase 3: ID Assignment
  Special tokens:                       10
  Reserved IDs:                        200
  Regular vocabulary allocated:    127,790
  Total allocated:                 128,000

Phase 4: Enforcement
  Final vocabulary size:           128,000 ✓
```

### Category Evolution Through Phases

```
Category      Collected   →   Selected   →   Allocated
──────────────────────────────────────────────────────
english       882,953         238,505         100,000
code           72,406          14,649          12,155
indic          16,283           8,645           8,645
symbols        35,820           8,016           3,000
numeric        22,124          10,110           4,000
other           3,523             801             100
special            23              10              10
json               80               8               7
──────────────────────────────────────────────────────
Total       1,033,212         280,744         128,000
```

**Key Insights:**

- **English:** 882k → 238k → 100k (heavy filtering)
- **Code:** 72k → 14k → 12k (good retention)
- **Indic:** 16k → 8.6k → 8.6k (all selected tokens included!)
- **Numeric:** 22k → 10k → 4k (capped at limit)

---

## Usage Examples

### Basic Usage

```bash
# Simple merge with defaults
python merge_tokenizers_with_chunking.py \
    --input-dir filtered_tokenizer \
    --output merged_tokens/merged_tokenizer_128k.json \
    --target-size 128000
```

### Using Master Configuration

```bash
# Recommended: Use master config file
python merge_tokenizers_with_chunking.py \
    --master-config tokenizer_merge_config.json
```

### Override Configuration Values

```bash
# Use master config but override target size
python merge_tokenizers_with_chunking.py \
    --master-config tokenizer_merge_config.json \
    --target-size 64000
```

### Custom Configurations

```bash
# Specify all components
python merge_tokenizers_with_chunking.py \
    --input-dir filtered_tokenizer \
    --output merged_tokenizer_custom.json \
    --target-size 128000 \
    --config tokenizer_config.json \
    --special-tokens-config special_tokens_config.json
```

### Quiet Mode

```bash
# Minimal output
python merge_tokenizers_with_chunking.py \
    --master-config tokenizer_merge_config.json \
    --quiet
```

### Complete Workflow

```bash
# Step 1: Filter tokenizers (remove unwanted tokens)
python tokenizer_filter.py \
    --input-dir ../data/ \
    --output-dir filtered_tokenizer \
    --remove-categories east_asian middle_eastern \
    --max-token-length 32

# Step 2: Generate analysis report (optional)
python generate_tokenizer_report.py \
    --dir filtered_tokenizer

# Step 3: Merge filtered tokenizers
python merge_tokenizers_with_chunking.py \
    --input-dir filtered_tokenizer \
    --output merged_tokens/merged_tokenizer_128k.json \
    --target-size 128000

# Output files:
#   merged_tokens/merged_tokenizer_128k.json         (vocabulary)
#   merged_tokens/merged_tokenizer_128k_report.json  (statistics)
```

---

## Technical Implementation

### Token Normalization

**Purpose:** Convert tokens from different formats to canonical form

```python
class TokenNormalizer:
    def normalize(self, token: str) -> str:
        """
        Normalize token to canonical form.
        
        Steps:
        1. Remove space markers (Ġ, ▁)
        2. Remove BERT continuation marker (##)
        3. Normalize structural tokens (lowercase)
        4. Decode byte-level BPE
        """
        
        # Remove space markers
        normalized = token.replace('Ġ', ' ')      # GPT-style
        normalized = normalized.replace('▁', ' ')  # SentencePiece
        
        # Remove ## prefix (BERT continuation)
        if normalized.startswith('##'):
            normalized = normalized[2:]
        
        # Normalize structural tokens
        if '<' in normalized and '>' in normalized:
            normalized = self._normalize_structural(normalized)
        
        # Try byte-level BPE decoding
        try:
            decoded = self._decode_bpe(normalized)
            return decoded
        except:
            return normalized
```

**Examples:**

```
Input:           Normalized:
────────────────────────────────
"Ġpython"    →  " python"
"▁def"       →  " def"
"##ing"      →  "ing"
"<INT>"      →  "<int>"  (lowercase)
"<DIV>"      →  "<div>"  (lowercase)
"âĢĻ"        →  "'"      (byte-level decode)
```

### Token Categorization

**Purpose:** Classify tokens into meaningful categories

```python
class TokenCategorizer:
    def categorize_token(self, token: str) -> str:
        """
        Categorize token based on content.
        
        Priority order:
        1. Special tokens (control, formatting)
        2. JSON structural (syntax)
        3. Language-specific (Indic, East Asian, etc.)
        4. Code (programming)
        5. English (alphabetic)
        6. Numeric (numbers)
        7. Symbols (punctuation)
        8. Other (fallback)
        """
        
        # Check special tokens
        if token in self.special_tokens:
            return "special"
        
        # Check JSON structural
        if token in {'{', '}', '[', ']', ':', ','}:
            return "json_structural"
        
        # Detect languages
        detected_languages = []
        for char in token:
            is_lang, lang_name, family = self.check_language(char)
            if is_lang:
                detected_languages.append(family)
        
        # Categorize by detected languages
        if detected_languages:
            if len(set(detected_languages)) == 1:
                return detected_languages[0]  # Single language
            else:
                return "multilingual"  # Multiple languages
        
        # Categorize by character types
        has_english = any(c.isalpha() and ord(c) < 128 for c in token)
        has_digit = any(c.isdigit() for c in token)
        has_code = bool(self.code_char_pattern.search(token))
        
        if has_code and has_english:
            return "code"
        elif has_english:
            return "english"
        elif has_digit:
            return "numeric"
        elif has_code:
            return "symbols"
        else:
            return "other"
```

### Placeholder Filtering

**Purpose:** Remove unused/placeholder tokens from source tokenizers

```python
class PlaceholderFilter:
    def is_placeholder(self, token: str) -> Tuple[bool, str]:
        """
        Check if token is a placeholder.
        
        Patterns detected:
        - [unused*], [UNK], [PAD]
        - <SPECIAL_\d+>
        - PLHD, placeholder
        - never_used
        - reserved, rsrvd
        """
        
        # Check patterns
        if self.unused_pattern.search(token):
            return True, "[unused*] pattern"
        
        if self.special_numbered_pattern.match(token):
            return True, "numbered SPECIAL token"
        
        if 'PLHD' in token.upper():
            return True, "PLHD placeholder"
        
        # ... more checks
        
        return False, ""
```

**Examples:**

```
Token               Placeholder?    Reason
────────────────────────────────────────────────────
"[unused0]"         YES             [unused*] pattern
"<SPECIAL_123>"     YES             numbered SPECIAL
"PLHD_TOKEN"        YES             PLHD placeholder
"never_used_token"  YES             never_used pattern
"python"            NO              (valid token)
"def"               NO              (valid token)
```

### Repeating Pattern Detection

**Purpose:** Detect and chunk long repeating patterns

```python
class RepeatingPatternDetector:
    def is_repeating(self, text: str, min_length: int = 3) -> Tuple[bool, str]:
        """
        Detect if token is a repeating pattern.
        
        Examples:
        - "=========" → True, "="
        - "--------" → True, "-"
        - "/* * * *" → True, "/* * * *"
        - "abcabc"   → True, "abc"
        """
        
        # Single character repetition
        if len(set(text)) == 1:
            return True, text[0]
        
        # Check 2-char patterns
        if len(text) >= 4:
            pattern = text[:2]
            if all(text[i:i+2] == pattern for i in range(0, len(text)-1, 2)):
                return True, pattern
        
        # Check 3-char patterns
        if len(text) >= 6:
            pattern = text[:3]
            if all(text[i:i+3] == pattern for i in range(0, len(text)-2, 3)):
                return True, pattern
        
        return False, ""
    
    def generate_chunks(self, text: str, chunk_sizes: List[int] = [8, 16]) -> List[str]:
        """
        Generate smaller chunks of repeating pattern.
        
        Example:
        Input:  "================================" (32 chars)
        Chunks: ["========", "================"] (8 and 16 chars)
        """
        
        is_rep, pattern = self.is_repeating(text)
        if not is_rep or len(text) <= 16:
            return []
        
        chunks = []
        for size in chunk_sizes:
            if size < len(text):
                chunk = (pattern * (size // len(pattern) + 1))[:size]
                chunks.append(chunk)
        
        return chunks
```

### Special Token Management

**Purpose:** Manage special tokens and reserved ID ranges

```python
class SpecialTokensManager:
    def __init__(self, config_path: Optional[Path] = None):
        """
        Load special tokens from configuration.
        
        Default special tokens:
        - </s>: 0 (end of sequence)
        - <pad>: 1 (padding)
        - <s>: 2 (start of sequence)
        - <unk>: 3 (unknown)
        - ... (more)
        
        Reserved IDs: 100-299 (for future use)
        Regular vocab starts at: 300
        """
        
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_default_config()
    
    def get_regular_vocab_start_id(self) -> int:
        """Calculate where regular vocabulary should start."""
        max_special_id = max(self.special_tokens.values())
        max_reserved_id = max(self.reserved_ids)
        return max(max_special_id, max_reserved_id) + 1
```

---

## Performance Considerations

### Memory Usage

```
Input data:
  10 tokenizers × ~100k tokens = ~50 MB JSON files
  
Processing:
  token_registry: ~280k entries × 200 bytes = ~56 MB
  merged_vocab: 128k entries × 100 bytes = ~12.8 MB
  
Peak memory: ~200 MB
```

### Processing Time

```
Phase 1 (Collection):        ~30 seconds
Phase 2 (Selection):         ~20 seconds
Phase 3 (ID Assignment):     ~5 seconds
Phase 4 (Enforcement):       ~1 second
───────────────────────────────────────
Total:                       ~60 seconds

With tqdm progress bars (optional)
```

### Optimization Tips

1. **Use tqdm for progress tracking:**
   ```bash
   pip install tqdm
   ```

2. **Process filtered tokenizers:**
   - Filter first (remove unwanted categories)
   - Then merge (faster, smaller input)

3. **Use master config:**
   - Faster startup (no argument parsing)
   - Easier to reproduce results

4. **Run in quiet mode for CI/CD:**
   ```bash
   python merge_tokenizers_with_chunking.py \
       --master-config config.json --quiet
   ```

---

## Troubleshooting

### Common Issues

#### Issue 1: "No tokenizers loaded"

**Cause:** Input directory doesn't contain JSON files

**Solution:**
```bash
# Check directory contents
ls -la filtered_tokenizer/

# Ensure JSON files exist
ls filtered_tokenizer/*.json
```

#### Issue 2: "Unknown JSON format"

**Cause:** JSON file structure not recognized

**Solution:** Verify JSON structure is one of:
- Nested: `{"model": {"vocab": {...}}}`
- Semi-nested: `{"vocab": {...}}`
- Flat: `{"token": id, ...}`

#### Issue 3: "Final size != target size"

**Cause:** Not enough tokens after filtering

**Solution:** Reduce filters or increase input tokenizers

#### Issue 4: Missing special tokens

**Cause:** Special tokens config not found

**Solution:**
```bash
# Create default config
python -c "
from merge_tokenizers_with_chunking import SpecialTokensManager
mgr = SpecialTokensManager()
print(mgr.get_all_special_tokens())
"
```

### Debug Mode

Enable verbose output for debugging:

```python
merger = TokenizerMerger(
    config_path="tokenizer_config.json",
    target_size=128000,
    verbose=True  # Enable detailed logging
)
```

---

## Appendix

### Token ID Ranges

```
ID Range        Purpose                      Count
─────────────────────────────────────────────────────
0-9             Special tokens (base)           10
10-99           Special tokens (extended)       90
100-299         Reserved (future use)          200
300-127,999     Regular vocabulary         127,700
─────────────────────────────────────────────────────
Total                                        128,000
```


### References

- **Tokenizer Filter:** `tokenizer_filter.py` - Pre-processing and category filtering
- **Token Analyzer:** `tokenizer_analyzer.py` - Analysis and statistics
- **Report Generator:** `generate_tokenizer_report.py` - Detailed reporting
- **Common Utilities:** `utils.py` - Shared functions and language ranges
- **Configuration Guide:** `SETUP_SUMMARY.md` - Setup and workflow guide

---
