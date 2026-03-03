# Marathi Dataset Generation Approach

## Overview

This document describes the approach for generating a Marathi language curriculum dataset similar to the English `group1` dataset. The goal is to create 200,000 question-answer pairs in Devanagari script, with each data point containing at least 512 tokens.

**Purpose**: Generate Marathi Q&A pairs for language and literacy training  
**Scope**: All 10 statement types adapted for Marathi/Devanagari script  
**Output**: Single TXT file `output/group1_marathi.txt`

## Vocabulary Sources & Verification

### Word Collection

- **Total unique Marathi words**: 335 verified words (after deduplication)
- **Sources**: Marathi dictionaries, educational resources
- **Word distribution by difficulty**:
  - Easy words (2-4 chars): 168 unique words
  - Medium words (5-6 chars): 215 unique words
  - Hard words (7+ chars): 80 unique words

### Categories

Words are organized into categories:
- Animals (प्राणी)
- Objects & Places (वस्तू)
- Body Parts (शरीराचे अवयव)
- Colors (रंग)
- Nature (निसर्ग)
- People & Family (लोक)
- Food (खाणे)
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
- **Number range**: 1-100 (100 values) for number वर्तनी questions

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
"कमळ" ची वर्तनी काय आहे? क, म, ळ. "घर" ची वर्तनी काय आहे? घ, र.
```

❌ **Wrong**:
```
"कमळ" ची वर्तनी सांगा। क, म, ळ।  (query has "।" instead of "?")
```

## Statement Types Breakdown

### Statement 1: वर्तनी (Spelling) - 28,600 pairs (14.3%)

**Question patterns**:
- `"कमळ" ची वर्तनी काय आहे?` → `क, म, ळ.`
- `"घर" ची वर्तनी काय आहे?` → `घ, र.`
- `"पाणी" ची वर्तनी काय आहे?` → `प, ा, ण, ी.`
- `"कोंबडी" ची अक्षरे काय आहेत?` → `कों, ब, डी.`

**Answer format**: Comma-separated Unicode characters (consonants, vowels, matras, anusvaras) ending with `.`

**Character splitting**: Uses Unicode characters
- Example: "कोंबडी" → क, ो, ं, ब, ड, ी (6 Unicode characters)
- Example: "ज्वालामुखी" → ज, ्, व, ा, ल, ा, म, ु, ख, ी (10 Unicode characters)

**Semantic variations**: 15+ templates per word (all use Marathi words only)

### Statement 2: Letter at Position (अक्षर स्थिती) - 25,800 pairs (12.9%)

**Question patterns**:
- `"कोंबडी" चे पहिले अक्षर काय आहे?` → `को.`
- `"कोंबडी" मध्ये "ब" कोणत्या स्थानावर आहे?` → `२.`
- `"पाणी" शब्दातील "पा" चे स्थान काय आहे?` → `१.`

**Answer format**: Number ending with `.`

**Character splitting**: Uses grapheme clusters (user-perceived characters)
- Example: "कोंबडी" → को, ंब, डी (3 grapheme clusters)
- Example: "कमळ" → क, म, ळ (3 grapheme clusters)

**Position variations**: पहिले, दुसरे, तिसरे, चौथे, पाचवे, सहावे, सातवे, आठवे, नववे, दहावे

### Statement 3: Sound Matching (ध्वनी जुळणी) - 20,000 pairs (10%)

**Question patterns**:
- `कोणता शब्द "/क/" ध्वनीने सुरू होतो, "कमळ" किंवा "घर"?` → `कमळ.`
- `"/प/" ध्वनीने सुरू होणारा शब्द कोणता आहे, "पाणी" किंवा "आकाश"?` → `पाणी.`

**Answer format**: The correct word ending with `.`

**Approach**: Multiple-choice format for clarity

### Statement 4: Letter Count (अक्षर गणना) - 25,800 pairs (12.9%)

**Question patterns**:
- `"कमळ" मध्ये किती अक्षरे आहेत?` → `३.`
- `"घर" मध्ये किती अक्षरे आहेत?` → `२.`
- `"कोंबडी" मध्ये किती अक्षरे आहेत?` → `३.`

**Answer format**: Number (in Marathi script) ending with `.`

**Character counting**: Uses grapheme cluster count (not Unicode character count)
- Example: "कोंbडी" → 3 अक्षरे (को, ंब, डी)
- Example: "पाणी" → 2 अक्षरे (पा, णी)

### Statement 5: Rhyming (यमक) - 20,000 pairs (10%)

**Question patterns** (Multiple Choice):
- `"कमळ" शी यमक करणारा शब्द कोणता आहे, "जमळ" किंवा "घर"?` → `जमळ.`
- `कोणता शब्द "घर" शी यमक करतो, "कर" किंवा "पाणी"?` → `कर.`

**Answer format**: The rhyming word ending with `.`

**Approach**: Multiple-choice format recommended for validation

### Statement 6: Classification (वर्गीकरण) - 20,000 pairs (10%)

**Question patterns**:
- `"कुत्रा" काय आहे, प्राणी किंवा वस्तू?` → `प्राणी.`
- `"कमळ" हे काय आहे, प्राणी किंवा वस्तू?` → `वस्तू.`

**Answer format**: Category name ending with `.`

**Categories**: प्राणी (animal), व्यक्ती (person), वस्तू (object)

### Statement 7: Position of Letter (अक्षराची स्थिती) - 17,200 pairs (8.6%)

**Question patterns**:
- `"कोंबडी" मध्ये "को" अक्षर कोणत्या स्थानावर आहे?` → `पहिले.` or `1.`
- `"कोंबडी" मध्ये "ंब" अक्षर कोणत्या स्थानावर आहे?` → `दुसरे.` or `2.`

**Answer format**: Position (word or numeric) ending with `.`

**Character reference**: Questions ask for position of grapheme clusters (not individual Unicode characters)
- Example: "कोंबडी" मध्ये "को" → पहिले (position of grapheme cluster "को")

### Statement 8: संख्या वर्तनी (Number Spelling) - 10,000 pairs (5%)

**Question patterns**:
- `11 ची वर्तनी काय आहे?` → `अ, क, रा.`
- `"पन्नास" ची वर्तनी काय आहे?` → `प, न्ना, स.`

**Answer format**: Number name or वर्तनी (as syllables) ending with `.`

**Character splitting**: Uses grapheme clusters (syllables)
- Example: "पन्नास" → पन, ्ना, स (3 grapheme clusters)

**Range**: Numbers 1-100

**Language**: All templates use Marathi words only, no English transliterations

### Statement 9: Last Letter (शेवटचे अक्षर) - 17,200 pairs (8.6%)

**Question patterns**:
- `"कोंबडी" चे शेवटचे अक्षर काय आहे?` → `डी।`
- `"कमळ" चे शेवटचे अक्षर काय आहे?` → `ळ।`
- `"घर" कोणत्या अक्षराने संपते?` → `र।`

**Answer format**: Last grapheme cluster ending with `।`

**Character reference**: Returns last grapheme cluster (not last Unicode character)
- Example: "कोंबडी" → डी (last grapheme cluster)
- Example: "पाणी" → णी (last grapheme cluster)

### Statement 10: Word Comparison (शब्द तुलना) - 11,000 pairs (5.5%)

**Question patterns**:
- `कोणता शब्द लांब आहे, "कोंबडी" किंवा "शाळा"?` → `कोंबडी।` (3 clusters > 2 clusters)
- `"सफरचंद" आणि "घर" यांपैकी कोणता शब्द लहान आहे?` → `घर।` (2 clusters < 4 clusters)

**Answer format**: Longer/shorter word ending with `।`

**Comparison method**: Uses grapheme cluster count (not Unicode character count)
- Example: "कोंबडी" (3 clusters) vs "शाळा" (2 clusters) → कोंबडी is longer
- Example: "सफरचंद" (4 clusters) vs "घर" (2 clusters) → घर is shorter

**Important**: Equal-length pairs are skipped (cannot compare when both words have same grapheme cluster count)

## Character Splitting Methodology

### Two Different Approaches

The dataset uses **two different character splitting methods** depending on the statement type:

#### 1. Grapheme Cluster Split (for All Statements)

**Function**: `get_marathi_grapheme_clusters(word: str) -> list[str]`

**Method**: Uses `regex` library's `\X` pattern (Unicode UAX#29 compliant)
- Used for: All statement types including Spelling (S1, S8)
- Example: "कोंबडी" → ['कों', 'ब', 'डी'] (3 grapheme clusters)
- Example: "पाणी" → ['पा', 'णी'] (2 grapheme clusters)
- Example: "ज्वालामुखी" → ['ज्वा', 'ला', 'मु', 'खी'] (4 grapheme clusters)

**Rationale**: Since spelling is now requested at the syllable (grapheme cluster) level, all statements now use the same high-level splitting methodology.

#### 2. Grapheme Cluster Split (for Counting/Position: S2, S4, S7, S9, S10)

**Function**: `get_marathi_grapheme_clusters(word: str) -> list[str]`

**Method**: Uses `regex` library's `\X` pattern (Unicode UAX#29 compliant)
- Used for: Counting, length, and position questions (S2, S4, S7, S9, S10)
- Example: "कोंबडी" → ['को', 'ंब', 'डी'] (3 grapheme clusters)
- Example: "पाणी" → ['पा', 'णी'] (2 grapheme clusters)
- Example: "कमळ" → ['क', 'म', 'ळ'] (3 grapheme clusters)

**Rationale**: Grapheme clusters represent user-perceived characters (consonant + matra, conjuncts, etc.), which is appropriate for counting and position questions.

**Key Insight**: In Marathi, "अक्षर" (akshar) means different things:
- **Spelling context**: Each Unicode character = 1 अक्षर
- **Counting/Position context**: Each grapheme cluster = 1 अक्षर

## Token Counting Methodology

### Function

- **Script**: `curriculum_training_data/prompt_utils.py`
- **Function**: `count_tokens(text: str) -> int`
- **Description**: "Count tokens using LLM-like tokenization"

### How It Works

- **For Devanagari/Marathi**: Each Unicode character counts as 1 token (matches spelling format)
- **For other scripts**: Word units (sequences of letters/digits) count as 1 token
- **Symbol units**: Punctuation, quotes, symbols = 1 token each
- **Whitespace**: Skipped (not counted)

### Devanagari Handling

- Each Devanagari Unicode character counts as 1 token
- This ensures consistency with the detailed spelling format
- Punctuation (`।`, `?`, `,`) counts as separate tokens

### Examples

- `"कमळ" ची वर्तनी काय आहे?` = 16 tokens (each Devanagari char = 1 token)
- `क, म, ळ।` = 6 tokens
- `"कमळ" ची वर्तनी काय आहे? क, म, ळ।` = 22 tokens

### Minimum Token Requirement

- Each data point must have **minimum 512 tokens**
- Achieved by combining multiple Q&A pairs using `combine_qa_pairs_to_reach_min_tokens_hindi()`

## Implementation Strategy

### File Structure

```
curriculum_training_data/
├── group1_marathi/
│   ├── generate_group1_marathi_dataset.py  # Main generator
│   ├── generate_s1_spelling.py             # Statement 1
│   ├── generate_s2_letter_position.py      # Statement 2
│   ├── generate_s3_sound.py                # Statement 3
│   ├── generate_s4_count.py                # Statement 4
│   ├── generate_s5_rhyme.py                # Statement 5
│   ├── generate_s6_classify.py             # Statement 6
│   ├── generate_s7_position.py             # Statement 7
│   ├── generate_s8_numbers.py              # Statement 8
│   ├── generate_s9_last.py                 # Statement 9
│   ├── generate_s10_compare.py             # Statement 10
│   ├── marathi_vocabulary.py               # Marathi word lists
│   └── MARATHI_DATASET_APPROACH.md         # This document
└── output/
    └── group1_marathi.txt                  # Final output TXT
```

### Generation Approach

1. **Individual Statement Generators**: Each statement type has its own generator script
2. **Vocabulary Module**: Centralized word lists in `marathi_vocabulary.py`
3. **Main Generator**: Combines all statements and creates final dataset
4. **Formatting Utilities**: Marathi-specific formatting in `prompt_utils.py`

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
- [x] Correct handling of matras and anusvara
- [x] **Only Marathi words used** - No English transliterations

### Token Count Verification

- [x] Minimum 512 tokens per data point
- [x] Token counting verified with Marathi examples
- [x] Proper character counting (handling matras, anusvara)
- [x] **Grapheme cluster detection** implemented using `regex` library's `\X` pattern
- [x] **Two splitting methods**: Unicode characters for spelling, grapheme clusters for counting/position

### Quality Checks

- [x] Natural Marathi question phrasing
- [x] Correct answer formats for each statement type
- [x] No duplicate Q&A pairs (within same data point)
- [x] Proper distribution across statement types
- [x] **S10 comparison**: Equal-length pairs skipped (cannot compare when grapheme cluster counts are equal)

## Output Format

### File Specifications

- **Filename**: `group1_marathi.txt`
- **Location**: `curriculum_training_data/output/group1_marathi.txt`
- **Format**: Continuous Q?A pairs on single lines
- **Encoding**: UTF-8

### Example Output

```
"कमळ" ची वर्तनी काय आहे? क, म, ळ। "घर" ची वर्तनी काय आहे? घ, र। "कोंबडी" ची अक्षरे काय आहेत? क, ो, ं, ब, ड, ी। "कोंबडी" चे पहिले अक्षर काय आहे? को। "कोंबडी" मध्ये किती अक्षरे आहेत? 3। "सफरचंद" आणि "घर" यांपैकी कोणता शब्द लहान आहे? घर।
```

**Note**: 
- All Statements use grapheme clusters: "कोंबडी" → कों, ब, डी
- "ज्वालामुखी" → ज्वा, ला, मु, खी

### Comparison with English Format

- **English**: Uses period (`.`) after answers
- **Marathi**: Uses purna-viraam (`।`) after answers
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
- **Function**: `get_marathi_grapheme_clusters(word)` in `generate_s1_spelling.py`
- **Usage**: Imported by S2, S4, S7, S9, S10 generators

### Language Purity

- **No English transliterations**: All templates use Marathi words only
- **Loanwords**: Common loanwords like "संगणक", "फोन" are acceptable (commonly used in Marathi)

### Statement-Specific Logic

| Statement | Character Split Method | Example |
|-----------|--------------------------|---------|
| S1 (Spelling) | Unicode characters | "कोंबडी" → क, ो, ं, ब, ड, ी (6 chars) |
| S2 (Position) | Grapheme clusters | "कोंबडी" → को, ंब, डी (3 clusters) |
| S4 (Count) | Grapheme clusters | "कोंबडी" → 3 अक्षरे |
| S7 (Position of) | Grapheme clusters | "कोंबडी" मध्ये "को" → पहिले |
| S8 (Number Spelling) | Unicode characters | "पन्नास" → प, न, ्, न, ा, स (6 chars) |
| S9 (Last) | Grapheme clusters | "कोंबडी" → डी |
| S10 (Compare) | Grapheme clusters | Compares cluster counts, skips equal pairs |

## Notes

- All scripts use UTF-8 encoding for proper Devanagari handling
- Character breakdown handles matras (vowel marks) and anusvara correctly
- Performance optimizations ensure fast generation (sub-second for most statements)
- The dataset follows the same structure as English `group1` but adapted for Marathi language and script
- **Grapheme cluster detection** ensures accurate counting and positioning for user-perceived characters
- **Language purity** maintained by using only Marathi words
