# Hindi Dataset Generation Approach

## Overview

This document describes the approach for generating a Hindi language curriculum dataset similar to the English `group1` dataset. The goal is to create 200,000 question-answer pairs in Devanagari script, with each data point containing at least 512 tokens.

**Purpose**: Generate Hindi Q&A pairs for language and literacy training  
**Scope**: All 10 statement types adapted for Hindi/Devanagari script  
**Output**: Single TXT file `output/group1_hindi.txt`

## Vocabulary Sources & Verification

### Word Collection

- **Total unique Hindi words**: 335 verified words (after deduplication)
- **Sources**: Shabdkosh, HindiPod101, government resources
- **Word distribution by difficulty**:
  - Easy words (2-4 chars): 168 unique words
  - Medium words (5-6 chars): 215 unique words
  - Hard words (7+ chars): 80 unique words

### Categories

Words are organized into categories:
- Animals (जानवर)
- Objects & Places (वस्तुएं)
- Body Parts (शरीर के अंग)
- Colors (रंग)
- Nature (प्रकृति)
- People & Family (लोग)
- Food (खाना)
- Professions (व्यवसाय)
- Vehicles
- Household Items
- Abstract Concepts
- Days of Week
- Months
- Numbers (1-100)

### Sufficiency Analysis

The 335 unique words are sufficient for generating 200,000+ pairs through:
- **Semantic variations**: 10-15+ question templates per word
- **Position variations**: 10 positions for letter-based questions
- **Word pair combinations**: 335 × 334 = 111,890 possible pairs for comparison questions
- **Reuse across statement types**: Same words used in different contexts
- **Number range**: 1-100 (100 values) for number spelling questions

## Format Specifications

### Pattern

- **Single pair**: `Q? A।`
- **Multiple pairs**: `Q? A। Q? A। Q? A। ...`
- **Spacing**: Exactly one space after `?` and exactly one space after `।`
- **No line breaks**: All pairs on same line, separated by `। ` (purna-viraam + space)

### Critical Format Rules

1. **ALL queries MUST end with "?"** - This ensures LLM understands it's a query during training
2. **NEVER use "।" in queries** - "।" should only appear after answers
3. **ALL answers MUST end with "।"** (purna-viraam)

### Examples

✅ **Correct**:
```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र।
```

❌ **Wrong**:
```
"कमल" का वर्तनी बताइए। क, म, ल।  (query has "।" instead of "?")
```

## Statement Types Breakdown

### Statement 1: Spelling (वर्तनी) - 28,600 pairs (14.3%)

**Question patterns**:
- `"कमल" की वर्तनी क्या है?` → `क, म, ल।`
- `"घर" की वर्तनी क्या है?` → `घ, र।`
- `"पानी" की वर्तनी क्या है?` → `प, ा, न, ी।`
- `"मुर्गी" के अक्षर क्या हैं?` → `म, ु, र, ्, ग, ी।`

**Answer format**: Comma-separated Unicode characters ending with `।`

**Character splitting**: Uses detailed Unicode character split (each Unicode codepoint = 1 character)
- Example: "मुर्गी" → म, ु, र, ्, ग, ी (6 Unicode characters)

**Semantic variations**: 15+ templates per word (all use Hindi words only, no English transliterations)

### Statement 2: Letter at Position (अक्षर स्थिति) - 25,800 pairs (12.9%)

**Question patterns**:
- `"मुर्गी" का पहला अक्षर क्या है?` → `मु।`
- `"मुर्गी" का दूसरा अक्षर क्या है?` → `र्गी।`
- `"कमल" का पहला अक्षर क्या है?` → `क।`

**Answer format**: Grapheme cluster ending with `।`

**Character splitting**: Uses grapheme clusters (user-perceived characters)
- Example: "मुर्गी" → मु, र्गी (2 grapheme clusters)
- Example: "कमल" → क, म, ल (3 grapheme clusters)

**Position variations**: पहला, दूसरा, तीसरा, चौथा, पांचवां, छठा, सातवां, आठवां, नौवां, दसवां

### Statement 3: Sound Matching (ध्वनि मिलान) - 20,000 pairs (10%)

**Question patterns**:
- `कौन सा शब्द "/क/" ध्वनि से शुरू होता है, "कुत्ता" या "बिल्ली"?` → `कुत्ता।`

**Answer format**: Selected word ending with `।`

**Approach**: Multiple-choice format for clarity

### Statement 4: Letter Count (अक्षर गिनती) - 25,800 pairs (12.9%)

**Question patterns**:
- `"मुर्गी" में कितने अक्षर हैं?` → `2।`
- `"कमल" में कितने अक्षर हैं?` → `3।`
- `"विद्यालय" में कितने अक्षर हैं?` → `4।`

**Answer format**: Numeric value ending with `।`

**Character counting**: Uses grapheme cluster count (not Unicode character count)
- Example: "मुर्गी" → 2 अक्षर (मु, र्गी)
- Example: "पानी" → 2 अक्षर (पा, नी)

### Statement 5: Rhyming (तुकबंदी) - 20,000 pairs (10%)

**Question patterns** (Multiple Choice):
- `"कमल" से तुकबंदी करने वाला शब्द कौन सा है, "जमल" या "बिल्ली"?` → `जमल।`

**Answer format**: Rhyming word ending with `।`

**Approach**: Multiple-choice format recommended for validation

### Statement 6: Classification (वर्गीकरण) - 20,000 pairs (10%)

**Question patterns**:
- `"कुत्ता" एक व्यक्ति, जानवर या वस्तु है?` → `जानवर।`
- `"शिक्षक" एक व्यक्ति, जानवर या वस्तु है?` → `व्यक्ति।`

**Answer format**: Category ending with `।`

**Categories**: जानवर (animal), व्यक्ति (person), वस्तु (object)

### Statement 7: Position of Letter (अक्षर की स्थिति) - 17,200 pairs (8.6%)

**Question patterns**:
- `"मुर्गी" में "मु" अक्षर किस स्थान पर है?` → `पहला।` or `1।`
- `"मुर्गी" में "र्गी" अक्षर किस स्थान पर है?` → `दूसरा।` or `2।`

**Answer format**: Position (word or numeric) ending with `।`

**Character reference**: Questions ask for position of grapheme clusters (not individual Unicode characters)
- Example: "मुर्गी" में "मु" → पहला (position of grapheme cluster "मु")

### Statement 8: Number Spelling (संख्या वर्तनी) - 10,000 pairs (5%)

**Question patterns**:
- `11 की वर्तनी क्या है?` → `ग्यारह।`
- `"पचास" की वर्तनी क्या है?` → `प, च, ा, स।`

**Answer format**: Number name or spelling ending with `।`

**Character splitting**: Uses detailed Unicode character split (same as S1)
- Example: "पचास" → प, च, ा, स (4 Unicode characters)

**Range**: Numbers 1-100

**Language**: All templates use Hindi words only (वर्तनी), no English transliterations

### Statement 9: Last Letter (अंतिम अक्षर) - 17,200 pairs (8.6%)

**Question patterns**:
- `"मुर्गी" का अंतिम अक्षर क्या है?` → `र्गी।`
- `"कमल" का अंतिम अक्षर क्या है?` → `ल।`
- `"घर" किस अक्षर से समाप्त होता है?` → `र।`

**Answer format**: Last grapheme cluster ending with `।`

**Character reference**: Returns last grapheme cluster (not last Unicode character)
- Example: "मुर्गी" → र्गी (last grapheme cluster)
- Example: "पानी" → नी (last grapheme cluster)

### Statement 10: Word Comparison (शब्द तुलना) - 11,000 pairs (5.5%)

**Question patterns**:
- `कौन सा शब्द लंबा है, "मुर्गी" या "विद्यालय"?` → `विद्यालय।` (4 clusters > 2 clusters)
- `"सेब" और "समानता" में से कौन सा शब्द छोटा है?` → `सेब।` (2 clusters < 4 clusters)

**Answer format**: Longer/shorter word ending with `।`

**Comparison method**: Uses grapheme cluster count (not Unicode character count)
- Example: "मुर्गी" (2 clusters) vs "विद्यालय" (4 clusters) → विद्यालय is longer
- Example: "सेब" (2 clusters) vs "समानता" (4 clusters) → सेब is shorter

**Important**: Equal-length pairs are skipped (cannot compare when both words have same grapheme cluster count)

## Character Splitting Methodology

### Two Different Approaches

The dataset uses **two different character splitting methods** depending on the statement type:

#### 1. Detailed Unicode Character Split (for Spelling: S1, S8)

**Function**: `get_hindi_characters(word: str) -> list[str]`

**Method**: Each Unicode codepoint is a separate character
- Used for: Spelling questions (S1, S8)
- Example: "मुर्गी" → ['म', 'ु', 'र', '्', 'ग', 'ी'] (6 Unicode characters)
- Example: "पानी" → ['प', 'ा', 'न', 'ी'] (4 Unicode characters)

**Rationale**: Spelling requires showing every Unicode character separately, including matras, halant, and combining marks.

#### 2. Grapheme Cluster Split (for Counting/Position: S2, S4, S7, S9, S10)

**Function**: `get_hindi_grapheme_clusters(word: str) -> list[str]`

**Method**: Uses `regex` library's `\X` pattern (Unicode UAX#29 compliant)
- Used for: Counting, length, and position questions (S2, S4, S7, S9, S10)
- Example: "मुर्गी" → ['मु', 'र्गी'] (2 grapheme clusters)
- Example: "पानी" → ['पा', 'नी'] (2 grapheme clusters)
- Example: "कमल" → ['क', 'म', 'ल'] (3 grapheme clusters)

**Rationale**: Grapheme clusters represent user-perceived characters (consonant + matra, conjuncts, etc.), which is appropriate for counting and position questions.

**Key Insight**: In Hindi, "अक्षर" (akshar) means different things:
- **Spelling context**: Each Unicode character = 1 अक्षर
- **Counting/Position context**: Each grapheme cluster = 1 अक्षर

## Token Counting Methodology

### Function

- **Script**: `curriculum_training_data/prompt_utils.py`
- **Function**: `count_tokens(text: str) -> int`
- **Description**: "Count tokens using LLM-like tokenization"

### How It Works

- **For Devanagari/Hindi**: Each Unicode character counts as 1 token (matches spelling format)
- **For other scripts**: Word units (sequences of letters/digits) count as 1 token
- **Symbol units**: Punctuation, quotes, symbols = 1 token each
- **Whitespace**: Skipped (not counted)

### Devanagari Handling

- Each Devanagari Unicode character counts as 1 token
- This ensures consistency with the detailed spelling format
- Punctuation (`।`, `?`, `,`) counts as separate tokens

### Examples

- `"कमल" की वर्तनी क्या है?` = 16 tokens (each Devanagari char = 1 token)
- `क, म, ल।` = 6 tokens
- `"कमल" की वर्तनी क्या है? क, म, ल।` = 22 tokens

### Minimum Token Requirement

- Each data point must have **minimum 512 tokens**
- Achieved by combining multiple Q&A pairs using `combine_qa_pairs_to_reach_min_tokens_hindi()`

## Implementation Strategy

### File Structure

```
curriculum_training_data/
├── group1_hindi/
│   ├── generate_group1_hindi_dataset.py  # Main generator
│   ├── generate_s1_spelling.py           # Statement 1
│   ├── generate_s2_letter_position.py    # Statement 2
│   ├── generate_s3_sound.py              # Statement 3
│   ├── generate_s4_count.py              # Statement 4
│   ├── generate_s5_rhyme.py              # Statement 5
│   ├── generate_s6_classify.py           # Statement 6
│   ├── generate_s7_position.py           # Statement 7
│   ├── generate_s8_numbers.py            # Statement 8
│   ├── generate_s9_last.py               # Statement 9
│   ├── generate_s10_compare.py           # Statement 10
│   ├── hindi_vocabulary.py               # Hindi word lists
│   └── HINDI_DATASET_APPROACH.md         # This document
└── output/
    └── group1_hindi.txt                    # Final output TXT
```

### Generation Approach

1. **Individual Statement Generators**: Each statement type has its own generator script
2. **Vocabulary Module**: Centralized word lists in `hindi_vocabulary.py`
3. **Main Generator**: Combines all statements and creates final dataset
4. **Formatting Utilities**: Hindi-specific formatting in `prompt_utils.py`

### Performance Optimizations

- **Pre-computed word groups**: Sound matching uses pre-computed groups by first sound
- **Cached character breakdowns**: Word lengths cached to avoid repeated computation
- **Efficient data structures**: Use sets and dictionaries for O(1) lookups
- **Batch processing**: Generate unique combinations first, then sample with replacement

### Semantic Variation Strategy

- **Multiple templates**: 10-15+ question templates per statement type
- **Word reuse**: Same words used across different contexts and statement types
- **Position variations**: 10 positions for letter-based questions
- **Template combinations**: Different templates × different words = high diversity

## Validation Criteria

### Format Validation

- [ ] ALL queries end with "?" (no "।" in queries)
- [ ] ALL answers end with "।" (purna-viraam)
- [ ] Proper spacing: `Q? A। Q? A।`
- [ ] No line breaks between pairs

### Word Authenticity

- [x] All words verified from trusted sources
- [x] Proper Devanagari script formatting
- [x] Correct handling of matras and halant
- [x] **Only Hindi words used** - No English transliterations (e.g., "स्पेलिंग" removed, use "वर्तनी" instead)

### Token Count Verification

- [x] Minimum 512 tokens per data point
- [x] Token counting verified with Hindi examples
- [x] Proper character counting (handling matras, halant)
- [x] **Grapheme cluster detection** implemented using `regex` library's `\X` pattern
- [x] **Two splitting methods**: Unicode characters for spelling, grapheme clusters for counting/position

### Quality Checks

- [x] Natural Hindi question phrasing
- [x] Correct answer formats for each statement type
- [x] No duplicate Q&A pairs (within same data point)
- [x] Proper distribution across statement types
- [x] **S10 comparison**: Equal-length pairs skipped (cannot compare when grapheme cluster counts are equal)

## Output Format

### File Specifications

- **Filename**: `group1_hindi.txt`
- **Location**: `curriculum_training_data/output/group1_hindi.txt`
- **Format**: Continuous Q?A pairs on single lines
- **Encoding**: UTF-8

### Example Output

```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र। "मुर्गी" के अक्षर क्या हैं? म, ु, र, ्, ग, ी। "मुर्गी" का पहला अक्षर क्या है? मु। "मुर्गी" में कितने अक्षर हैं? 2। "सेब" और "समानता" में से कौन सा शब्द छोटा है? सेब।
```

**Note**: 
- Spelling (S1) shows detailed Unicode characters: "मुर्गी" → म, ु, र, ्, ग, ी
- Counting/Position (S2, S4) use grapheme clusters: "मुर्गी" → 2 अक्षर (मु, र्गी)

### Comparison with English Format

- **English**: Uses period (`.`) after answers
- **Hindi**: Uses purna-viraam (`।`) after answers
- **Both**: Queries end with `?`, continuous format, minimum token requirement

## Distribution Summary

| Statement | Pairs | Percentage |
|-----------|-------|------------|
| S1: Spelling | 28,600 | 14.3% |
| S2: Letter Position | 25,800 | 12.9% |
| S3: Sound Matching | 20,000 | 10% |
| S4: Letter Count | 25,800 | 12.9% |
| S5: Rhyming | 20,000 | 10% |
| S6: Classification | 20,000 | 10% |
| S7: Position of Letter | 17,200 | 8.6% |
| S8: Number Spelling | 10,000 | 5% |
| S9: Last Letter | 17,200 | 8.6% |
| S10: Word Comparison | 11,000 | 5.5% |
| **Total** | **200,000** | **100%** |

## Implementation Details

### Grapheme Cluster Detection

- **Library**: `regex` (Python package)
- **Pattern**: `\X` (Unicode UAX#29 compliant grapheme cluster)
- **Function**: `get_hindi_grapheme_clusters(word)` in `generate_s1_spelling.py`
- **Usage**: Imported by S2, S4, S7, S9, S10 generators

### Language Purity

- **No English transliterations**: All templates use Hindi words only
- **Removed**: "स्पेलिंग" (spelling) → replaced with "वर्तनी"
- **Removed**: "स्पेल करते हैं" → replaced with "वर्तनी करते हैं" or "लिखते हैं"
- **Loanwords**: Common loanwords like "कंप्यूटर", "फोन" are acceptable (commonly used in Hindi)

### Statement-Specific Logic

| Statement | Character Split Method | Example |
|-----------|------------------------|---------|
| S1 (Spelling) | Unicode characters | "मुर्गी" → म, ु, र, ्, ग, ी (6 chars) |
| S2 (Position) | Grapheme clusters | "मुर्गी" → मु, र्गी (2 clusters) |
| S4 (Count) | Grapheme clusters | "मुर्गी" → 2 अक्षर |
| S7 (Position of) | Grapheme clusters | "मुर्गी" में "मु" → पहला |
| S8 (Number Spelling) | Unicode characters | "पचास" → प, च, ा, स (4 chars) |
| S9 (Last) | Grapheme clusters | "मुर्गी" → र्गी |
| S10 (Compare) | Grapheme clusters | Compares cluster counts, skips equal pairs |

## Notes

- All scripts use UTF-8 encoding for proper Devanagari handling
- Character breakdown handles matras (vowel marks) and halant correctly
- Performance optimizations ensure fast generation (sub-second for most statements)
- The dataset follows the same structure as English `group1` but adapted for Hindi language and script
- **Grapheme cluster detection** ensures accurate counting and positioning for user-perceived characters
- **Language purity** maintained by using only Hindi words (no English transliterations)
