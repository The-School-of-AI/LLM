# Curriculum Training Data

A comprehensive dataset generation system for AI curriculum learning, focusing on foundational skills in language, mathematics, visual perception, and pattern recognition.

## Purpose

This repository contains scripts and data for generating high-quality training datasets across multiple curriculum groups and languages:

- **Group 1 (English)**: Language and Literacy (~690,400 samples)
- **Group 1 (Indian languages)**: Hindi, Punjabi, Marathi, Kannada, Telugu, Assamese — each ~200,000 samples
- **Group 2**: Math and Numbers (~382,886 samples)
- **Group 3**: Shapes, Colors & Patterns (~128,302 samples)

These datasets are designed for curriculum learning in AI models, following educational progression from foundational concepts to advanced skills.

## Dataset Overview

### Group 1: Language and Literacy (English)
**Target**: 70,000 samples (currently 690,400 generated)

Focuses on foundational language skills across 10 statement types:
- **S1: Spelling** (100,000 samples) - Letter-by-letter spelling with 15+ templates
- **S2: Letter at Position** (90,000 samples) - Identifying letters at specific positions
- **S3: Sound Matching** (70,000 samples) - Phonetic sound recognition and matching
- **S4: Letter Count** (90,000 samples) - Counting letters in words
- **S5: Rhyming** (70,000 samples) - Word rhyming with 24 templates
- **S6: Classification** (70,000 samples) - Categorizing words as person/animal/thing
- **S7: Position of Letter** (60,000 samples) - Finding position of specific letters
- **S8: Number Spelling** (50,000 samples) - Spelling numbers 1-100
- **S9: Last Letter** (60,000 samples) - Identifying last letters in words
- **S10: Word Comparison** (30,000 samples) - Comparing word lengths

**Key Features**:
- **7,031 unique words** across difficulty levels (EASY: 1,077, MEDIUM: 3,006, HARD: 4,138)
- **10 distinct statement types** with semantic variations
- **Template-based generation** with 10-24 templates per statement type
- **Difficulty distribution**: 30% easy, 50% medium, 20% hard
- **Generation strategies**: Random selection (high capacity) and systematic enumeration (tight capacity)
- **Output format**: Q&A pairs combined to reach minimum 512 tokens per data point

**Architecture**:
- Main generator: `group1/generate_group1_dataset.py`
- Individual statement generators: `generate_s1_spelling.py` through `generate_s10_compare.py`
- Shared utilities: `prompt_utils.py` for token counting and Q&A combination
- Documentation: `group1/group1_plan.md` with detailed breakdown

**Files**:
- `group1/generate_group1_dataset.py` - Main generation script
- `group1/generate_s1_spelling.py` through `generate_s10_compare.py` - Statement generators
- `group1/combine_and_fill.py` - Q&A combination utilities
- `output/group1_part1.txt` - Final output dataset part 1 (15.8MB, ~345,200 samples)
- `output/group1_part2.txt` - Final output dataset part 2 (17.3MB, ~345,200 samples)
- **Note**: Split into two parts for GitHub compatibility (each < 25MB)

### Group 2: Math and Numbers (English)
**Target**: 600,000 samples (currently 382,886 generated)

Covers mathematical concepts across 6 statement types:
- **S1: Counting** (60,000 samples) - Counting sequences from 1 to N, custom ranges
- **S2: Before/After** (80,000 samples) - Number sequences, what comes before/after
- **S3: Word Problems** (120,000 samples) - Real-world math problems with objects
- **S4: Comparisons** (100,000 samples) - Comparing numbers (bigger/smaller/higher/lower)
- **S5: Direct Math** (150,000 samples) - Arithmetic operations (addition, subtraction, multiplication)
- **S6: Word-Based Math** (90,000 samples) - Math operations using number words

**Key Features**:
- **Number range**: 1-1000+ for various operations
- **25+ templates per statement type** (expanded from 5-12 templates)
- **Proper pluralization** handling for word problems
- **Number-word variations**: Supports both numeric (5) and spelled-out (five) formats
- **Curriculum-aligned progression**: Foundation → Advanced skills
- **Comprehensive validation**: 80+ detection patterns for accurate categorization

**Architecture**:
- Main generator: `group2/generate_group2_dataset.py`
- Validation: `group2/validate_group2.py`
- Documentation: `group2/DISTRIBUTION_JUSTIFICATION.md`, `group2/TECHNICAL_SUMMARY_final.md`
- Output format: Q&A pairs combined to reach minimum 512 tokens per data point

**Files**:
- `group2/generate_group2_dataset.py` - Main generation script
- `group2/validate_group2.py` - Validation and distribution analysis
- `output/group2.txt` - Final output dataset (16MB, 382,886 samples)

### Group 3: Shapes, Colors & Patterns (English)
**Target**: 150,000 samples (currently 128,302 generated)

Visual perception and pattern recognition across 4 statement types with 35 sub-generators:
- **Statement 1: Color Perception** (55,000 samples) - 6 sub-generators:
  - 1A: Object Color ID (30,000)
  - 1B: Reverse Color ID (5,000)
  - 1C: Color Verification (8,000)
  - 1D: Color Multiple Choice (5,000)
  - 1E: Color Mixing (2,000)
  - 1F: Color Associations (2,500)

- **Statement 2: Shape Perception** (40,000 samples) - 6 sub-generators:
  - 2A: Object Shape ID (25,000)
  - 2B: Reverse Shape ID (5,000)
  - 2C: Shape Verification (5,000)
  - 2D: Shape Multiple Choice (4,000)
  - 2E: 2D vs 3D Distinction (1,000)
  - 2F: 2D-3D Relationship (500)

- **Statement 3: Geometric Concepts** (10,000 samples) - 12 sub-generators:
  - Sides, vertices, faces, angles, symmetry, perimeter, area, volume, etc.

- **Statement 4: Pattern Recognition** (45,000 samples) - 11 sub-generators:
  - Number sequences, shape patterns, color patterns, growing patterns, etc.

**Key Features**:
- **Quality-focused generation**: Revised from 250,000 to 150,000 based on combinatorial analysis
- **200+ color-object mappings**: 11 colors × 15-25 objects per color
- **150+ shape-object mappings**: 2D and 3D shapes
- **Mathematically accurate**: All geometric facts verified
- **High diversity**: 8-10 templates per sub-generator
- **Parametric expansions**: Adjectives, contexts, states for increased variety
- **Curriculum progression**: Patterns → Colors → Shapes → Geometry

**Architecture**:
- Main generator: `group3/generate_group3_dataset.py` (~3,000 lines)
- Validation: Built-in `validate_distribution()` function
- Documentation: `group3/group3_approach.md` with combinatorial analysis
- Output format: Q&A pairs combined to reach minimum 512 tokens per data point

**Files**:
- `group3/generate_group3_dataset.py` - Main generation script
- `group3/group3_approach.md` - Technical documentation and target revision justification
- `output/group3.txt` - Final output dataset (4.6MB, 128,302 samples)

## Project Structure

```
curriculum_training_data/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── prompt_utils.py                     # Shared utilities (token counting, Q&A combination)
├── combine_group1_parts.py             # Utility to combine group1_part1.txt + group1_part2.txt
├── CURRICULUM_LANGUAGE_DATA_GUIDE.md   # Curriculum language data guide
├── analyze_group1_queries.py          # Analysis utilities
├── calculate_sample_distribution.py   # Sample distribution calculator
├── curriculum_distribution.json       # Distribution configuration
├── validate_determinism.py            # Validates distribution stability across runs
│
├── output/                             # Generated dataset outputs
│   ├── group1_part1.txt                # Group 1 English output part 1 (~345K samples)
│   ├── group1_part2.txt                # Group 1 English output part 2 (~345K samples)
│   ├── group1_hindi.txt                # Group 1 Hindi output (200,000 samples)
│   ├── group1_punjabi.txt              # Group 1 Punjabi (200,000 samples)
│   ├── group1_marathi.txt              # Group 1 Marathi (200,000 samples)
│   ├── group1_kannada.txt              # Group 1 Kannada (200,000 samples)
│   ├── group1_telugu.txt               # Group 1 Telugu (200,000 samples)
│   ├── group1_assamese.txt             # Group 1 Assamese (200,000 samples)
│   ├── group2.txt                     # Group 2 English output (382,886 samples)
│   └── group3.txt                     # Group 3 English output (128,302 samples)
│
├── group1/                             # Group 1 English generators
│   ├── generate_group1_dataset.py      # Main generator
│   ├── generate_s1_spelling.py         # Statement 1: Spelling
│   ├── generate_s2_letter_position.py  # Statement 2: Letter at Position
│   ├── generate_s3_sound.py            # Statement 3: Sound Matching
│   ├── generate_s4_count.py            # Statement 4: Letter Count
│   ├── generate_s5_rhyme.py            # Statement 5: Rhyming
│   ├── generate_s6_classify.py         # Statement 6: Classification
│   ├── generate_s7_position.py         # Statement 7: Position of Letter
│   ├── generate_s8_numbers.py          # Statement 8: Number Spelling
│   ├── generate_s9_last.py            # Statement 9: Last Letter
│   ├── generate_s10_compare.py        # Statement 10: Word Comparison
│   ├── combine_and_fill.py             # Q&A combination utilities
│   ├── group1_plan.md                  # Detailed documentation
│   ├── final_updated_analysis_group1.md
│   └── group1_s1.json … group1_s10.json  # Statement outputs (JSON)
│
├── group1_hindi/                       # Group 1 Hindi (Devanagari)
│   ├── HINDI_DATASET_APPROACH.md       # Approach document
│   ├── generate_group1_hindi_dataset.py
│   ├── generate_s1_spelling.py … generate_s10_compare.py
│   ├── hindi_vocabulary.py             # 335 verified words
│   ├── validate_dataset.py
│   └── group1_s1.txt … group1_s10.txt  # Intermediate outputs
│
├── group1_punjabi/                     # Group 1 Punjabi (Gurmukhi)
│   ├── PUNJABI_DATASET_APPROACH.md
│   ├── generate_group1_punjabi_dataset.py
│   ├── generate_s1_spelling.py … generate_s10_compare.py
│   ├── punjabi_vocabulary.py           # ~450 verified words
│   └── group1_s1.txt … group1_s10.txt
│
├── group1_marathi/                     # Group 1 Marathi (Devanagari)
│   ├── MARATHI_DATASET_APPROACH.md
│   ├── generate_group1_marathi_dataset.py
│   ├── generate_s1_spelling.py … generate_s10_compare.py
│   ├── marathi_vocabulary.py            # 335 verified words
│   └── group1_s1.txt … group1_s10.txt
│
├── group1_kannada/                     # Group 1 Kannada (Kannada script)
│   ├── KANNADA_DATASET_APPROACH.md
│   ├── generate_group1_kannada_dataset.py
│   ├── generate_s1_spelling.py … generate_s11_ottakshara.py  # S11: Ottakshara
│   ├── kannada_vocabulary.py, kannada_expanded_vocabulary.py
│   ├── kannada_grammar.py
│   └── group1_s1.txt … group1_s11.txt
│
├── group1_telugu/                      # Group 1 Telugu (Telugu script)
│   ├── TELUGU_DATASET_APPROACH.md, TELUGU_DATASET_VALIDATION.md
│   ├── generate_group1_telugu_dataset.py
│   ├── generate_s1_spelling.py … generate_s11_ottulu_gunintalu.py  # S11: Ottulu/Gunintalu
│   ├── telugu_vocabulary.py, telugu_grammar.py
│   ├── prompt_utils_telugu.py
│   ├── tests/                          # Unit tests
│   └── group1_s1.txt … group1_s11.txt
│
├── group1_assamese/                    # Group 1 Assamese (Bengali-Assamese script)
│   ├── ASSAMESE_DATASET_APPROACH.md
│   ├── generate_group1_assamese_dataset.py
│   ├── generate_s1_spelling.py … generate_s10_semantics.py  # Custom statement mapping
│   ├── assamese_vocabulary.py, assamese_vocabulary_expanded.py
│   └── group1_s1.txt … group1_s10.txt
│
├── group2/                             # Group 2 English (Math and Numbers)
│   ├── generate_group2_dataset.py
│   ├── validate_group2.py
│   ├── group2_curriculum_distribution.md
│   ├── DISTRIBUTION_JUSTIFICATION.md
│   └── TECHNICAL_SUMMARY_final.md
│
├── group3/                             # Group 3 English (Shapes, Colors & Patterns)
│   ├── generate_group3_dataset.py     # ~3,000 lines, 35 sub-generators
│   ├── validate_data_quality.py
│   └── group3_approach.md
│
└── prompts/
    ├── group1prompt.txt
    ├── group2prompt.txt
    └── group3prompt.txt
```

## Installation

### Prerequisites

- Python 3.12
- Standard library only (no external dependencies required)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd curriculum_training_data
   ```

2. **Install dependencies** (optional - all scripts use standard library):
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Python version**:
   ```bash
   python3 --version  # Should be 3.12+
   ```

## Usage

### Generating Group 1 English Dataset

```bash
# From curriculum_training_data directory
python group1/generate_group1_dataset.py

# Output: curriculum_training_data/output/group1_part1.txt and group1_part2.txt
```

**Expected Output**:
- ~690,400 samples (exceeds original 70,000 target)
- Distribution validation report showing counts per statement type
- Two TXT files with Q&A pairs combined to reach minimum 512 tokens per data point
- Format: `Q1? A1. Q2? A2. Q3? A3. ...` (continuous format)
- Files split into two parts for GitHub compatibility (each < 25MB):
  - `output/group1_part1.txt` (~15.8MB, ~345,200 samples)
  - `output/group1_part2.txt` (~17.3MB, ~345,200 samples)

**Generation Process**:
1. Each of 10 statement generators creates Q&A pairs
2. Pairs are deduplicated and validated
3. Pairs are combined using `combine_qa_pairs_to_reach_min_tokens()` to reach 512+ tokens per sample
4. Final output written to `output/group1.txt`, then split into two parts for GitHub compatibility

**Note**: The dataset is split into two parts (`group1_part1.txt` and `group1_part2.txt`) 
to comply with GitHub's file size recommendations (< 25MB per file). To combine them back 
into a single file, use: `python combine_group1_parts.py`

### Generating Group 1 Hindi Dataset

```bash
# From curriculum_training_data directory
python group1_hindi/generate_group1_hindi_dataset.py

# Output: curriculum_training_data/output/group1_hindi.txt
```

**Expected Output**:
- 200,000 samples (exactly as specified)
- Distribution: S1: 28,600, S2: 25,800, S3: 20,000, S4: 25,800, S5: 20,000, S6: 20,000, S7: 17,200, S8: 10,000, S9: 17,200, S10: 11,000
- TXT file with Hindi Q&A pairs in Devanagari script
- Format: `Q? A। Q? A। Q? A। ...` (purna-viraam separator)
- Minimum 512 tokens per data point

**Key Features**:
- Uses 335 verified Hindi words from trusted sources
- Two character splitting methods: Unicode characters (spelling) and grapheme clusters (counting/position)
- All queries end with "?" and answers end with "।" (purna-viraam)
- See `group1_hindi/HINDI_DATASET_APPROACH.md` for comprehensive documentation

### Generating Group 1 Punjabi Dataset

```bash
python group1_punjabi/generate_group1_punjabi_dataset.py
# Output: output/group1_punjabi.txt (200,000 samples, Gurmukhi script)
```

### Generating Group 1 Marathi Dataset

```bash
python group1_marathi/generate_group1_marathi_dataset.py
# Output: output/group1_marathi.txt (200,000 samples, Devanagari script)
```

### Generating Group 1 Kannada Dataset

```bash
python group1_kannada/generate_group1_kannada_dataset.py
# Output: output/group1_kannada.txt (200,000 samples, S1–S11 including Ottakshara)
```

### Generating Group 1 Telugu Dataset

```bash
python group1_telugu/generate_group1_telugu_dataset.py
# Output: output/group1_telugu.txt (200,000 samples, S1–S11 including Ottulu/Gunintalu)
```

### Generating Group 1 Assamese Dataset

```bash
python group1_assamese/generate_group1_assamese_dataset.py
# Output: output/group1_assamese.txt (200,000 samples, Bengali-Assamese script)
# Note: Assamese uses custom statement mapping (S2+S7 merged, S9 Morphology, S10 Semantics)
```

### Generating Group 2 English Dataset

```bash
# From curriculum_training_data directory
cd group2
python generate_group2_dataset.py

# Output: ../output/group2.txt
```

**Expected Output**:
- ~382,886 samples (target: 600,000, 64% complete)
- Distribution validation report with statement type breakdown
- TXT file with Q&A pairs combined to reach minimum 512 tokens per data point
- Format: `Q1? A1. Q2? A2. Q3? A3. ...` (continuous format)

**Key Improvements Applied**:
- Fixed pluralization bugs (candieses → candies)
- Expanded templates from 5-12 to 25+ per statement type
- Added number-word variations (5 vs five)
- Comprehensive validation with 80+ detection patterns

### Generating Group 3 English Dataset

```bash
# From curriculum_training_data directory
cd group3
python generate_group3_dataset.py

# Output: ../output/group3.txt
```

**Expected Output**:
- ~128,302 samples (target: 150,000, 86% complete)
- Distribution validation report across 35 sub-generators
- TXT file with Q&A pairs combined to reach minimum 512 tokens per data point
- Format: `Q1? A1. Q2? A2. Q3? A3. ...` (continuous format)

**Note**: Group 3 targets were revised from 250,000 to 150,000 based on rigorous combinatorial analysis. See `group3/group3_approach.md` for technical justification. The current 128,302 samples represent high-quality, mathematically verified data.

## Output Format

All datasets use a **continuous Q&A format** with minimum 512 tokens per data point:

### English Format
```
Q1? A1. Q2? A2. Q3? A3. Q4? A4. ...
```

**Example**:
```
What is the spelling of 'cat'? c, a, t. How many letters are in 'dog'? 3. What rhymes with 'cat'? bat.
```

### Hindi Format
```
Q? A। Q? A। Q? A। ...
```

**Example**:
```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र। "मुर्गी" में कितने अक्षर हैं? 2।
```

**Structure**:
- **Format**: Continuous Q&A pairs on single lines
- **Encoding**: UTF-8
- **Minimum tokens**: 512 tokens per data point (enforced by `combine_qa_pairs_to_reach_min_tokens()`)
- **Separators**: 
  - English, Kannada, Telugu: Period (`.`) after answers
  - Hindi, Punjabi, Marathi, Assamese: Purna-viraam (`।`) after answers
- **Query format**: All queries end with `?`
- **Answer format**: All answers end with `.` or `।` as above

## Dataset Statistics

### Current Sample Counts

| Group | Language | Target | Generated | Status |
|-------|----------|--------|-----------|--------|
| Group 1 | English | 70,000 | 690,400 | ✅ Exceeded |
| Group 1 | Hindi | 200,000 | 200,000 | ✅ Complete |
| Group 1 | Punjabi | 200,000 | 200,000 | ✅ Complete |
| Group 1 | Marathi | 200,000 | 200,000 | ✅ Complete |
| Group 1 | Kannada | 200,000 | 200,000 | ✅ Complete |
| Group 1 | Telugu | 200,000 | 200,000 | ✅ Complete |
| Group 1 | Assamese | 200,000 | 200,000 | ✅ Complete |
| Group 2 | English | 600,000 | 382,886 | ⚠️ 64% complete |
| Group 3 | English | 150,000 | 128,302 | ⚠️ 86% complete |

### Quality Metrics

- **Uniqueness**: Zero duplicates (dictionary-based deduplication)
- **Diversity**: High template variation across all groups
- **Accuracy**: All facts verified (especially geometric data)
- **Coverage**: Comprehensive concept space exploration

## Validation

Each generator includes built-in validation:

1. **Distribution Validation**: Verifies sample counts match expected targets
2. **Category Classification**: Categorizes queries by statement type
3. **Tolerance Thresholds**: 
   - ✓ OK: ±5% deviation
   - ⚠️ WARNING: ±5-10% deviation
   - ✗ ERROR: >10% deviation

### Running Validation

Validation runs automatically during generation. To analyze existing datasets:

```bash
# Analyze Group 1
python analyze_group1_queries.py
```

## Reproducibility & Determinism

### Non-Deterministic Content, Deterministic Distribution

The generation scripts **do not fix random seeds** (`random.seed()` is never called), so each run produces **different content** — different specific QA pairs are selected. However, the **distribution of samples per statement type is deterministic** across runs.

This was validated empirically by running generators multiple times and comparing results:

| Aspect | Deterministic? | Explanation |
|--------|---------------|-------------|
| **Per-statement QA pair counts** | **Yes — exactly identical** | Targets are hardcoded constants. Combinatorial generators (Group 1, parts of Group 3) always produce `min(combinatorial_space, target)` samples regardless of random ordering. Attempt-based generators (Group 2, parts of Group 3) use high attempt multipliers (100×–200× target) ensuring targets are reliably reached. |
| **Content (specific QA pairs)** | **No** | Different runs select different subsets from the combinatorial space. Empirically, content overlap between runs is 0–8%, confirming high diversity. |
| **Combined sample count (output lines)** | **Approximately — within ~1%** | The `combine_qa_pairs_to_reach_min_tokens()` function is itself deterministic (sequential packing, no randomness), but different QA pair selections across runs produce slightly different average token lengths, leading to minor packing variation (~0.8%). |

### Why Distribution is Stable

1. **Hardcoded targets**: Every generator has explicit `num_samples` parameters (e.g., `generate_s1_spelling(num_samples=100000)`).

2. **Combinatorial generation pattern** (Group 1, parts of Group 3):
   - All possible combinations are enumerated (e.g., `words × templates`)
   - The list is shuffled (`random.shuffle`)
   - Iteration stops at `num_samples` — the count is always `min(len(all_combinations), num_samples)`
   - Shuffling changes *which* samples are selected, not *how many*

3. **Attempt-based generation pattern** (Group 2, parts of Group 3):
   - `while len(samples) < num_samples and attempt < max_attempts` with `max_attempts = 100×–200× target`
   - Large combinatorial spaces relative to targets ensure the target is reached consistently

4. **Validation enforcement**: Built-in `validate_distribution()` checks actual vs expected counts with ±5% OK / ±10% warning / >10% error thresholds.

### Empirical Validation

A validation script (`validate_determinism.py`) confirms these properties across 3 independent runs:

```
Group 1 (combinatorial generators):
  S1 Spelling:        5,000 / 5,000 / 5,000  — IDENTICAL
  S2 Letter Position: 5,000 / 5,000 / 5,000  — IDENTICAL
  S4 Letter Count:    5,000 / 5,000 / 5,000  — IDENTICAL
  S6 Classification:  5,000 / 5,000 / 5,000  — IDENTICAL
  S9 Last Letter:     5,000 / 5,000 / 5,000  — IDENTICAL
  S10 Word Comparison:5,000 / 5,000 / 5,000  — IDENTICAL
  Content overlap: 0.0%–8.4% (confirming different content per run)

Group 2 (attempt-based generators):
  S1 Counting:    1,150 / 1,150 / 1,150  — IDENTICAL
  S2 Before/After:2,000 / 2,000 / 2,000  — IDENTICAL
  S4 Comparisons: 2,000 / 2,000 / 2,000  — IDENTICAL

Combine step: 123 / 123 / 122 combined samples (0.81% variation)
```

## Technical Details

### Generation Strategies

1. **Random Selection** (High Capacity)
   - Used when: `max_capacity >> target_samples`
   - Approach: High max_attempts multiplier (100× to 200×)
   - Examples: Group 1 (spelling), Group 2 (arithmetic), Group 3 (color/shape ID)

2. **Systematic Enumeration** (Tight Capacity)
   - Used when: `max_capacity < target_samples × 2`
   - Approach: Enumerate all combinations, shuffle, select first N
   - Examples: Group 3 (geometric facts, color mixing)

3. **Parametric Variations** (Expansion)
   - Adjectives, contexts, states
   - Increases combinatorial space without compromising quality
   - Examples: Group 3 (color/shape queries with modifiers)

### Quality Assurance

- **No Artificial Expansion**: Rejected far-fetched associations for quality
- **Mathematically Accurate**: All geometric facts verified
- **Educationally Appropriate**: Curriculum-aligned scope (foundation to hard level)
- **Combinatorial Analysis**: Targets based on mathematical limits

## Documentation

### English Datasets
- **Group 1**: `group1/group1_plan.md` - Detailed breakdown of 10 statement types, word pools, templates
- **Group 2**: `group2/DISTRIBUTION_JUSTIFICATION.md`, `group2/TECHNICAL_SUMMARY_final.md` - Technical rationale and curriculum distribution
- **Group 3**: `group3/group3_approach.md` - Comprehensive approach, combinatorial analysis, target revision justification

### Indian Language Datasets
- **Hindi**: `group1_hindi/HINDI_DATASET_APPROACH.md` - Devanagari, 335 words, 10 statement types
- **Punjabi**: `group1_punjabi/PUNJABI_DATASET_APPROACH.md` - Gurmukhi, ~450 words
- **Marathi**: `group1_marathi/MARATHI_DATASET_APPROACH.md` - Devanagari, 335 words
- **Kannada**: `group1_kannada/KANNADA_DATASET_APPROACH.md` - Kannada script, akshara segmentation, S1–S11
- **Telugu**: `group1_telugu/TELUGU_DATASET_APPROACH.md`, `TELUGU_DATASET_VALIDATION.md` - Telugu script, akshara segmentation, S1–S11
- **Assamese**: `group1_assamese/ASSAMESE_DATASET_APPROACH.md` - Bengali-Assamese script, custom statement mapping

## Contributing

When adding new generators or modifying existing ones:

1. **Follow the architecture**: Data modules → Templates → Generators → Validation
2. **Maintain quality**: Prioritize accuracy over quantity
3. **Document changes**: Update relevant markdown files
4. **Validate output**: Ensure distribution matches targets
5. **Test thoroughly**: Verify no duplicates and correct format

## Troubleshooting

### Common Issues

1. **Low sample counts**
   - Check combinatorial limits (see Group 3 approach doc)
   - Verify data module completeness
   - Review template variations

2. **High uncategorized samples**
   - Update validation patterns in `validate_distribution()`
   - Check query format matches expected patterns

3. **Generation taking too long**
   - Check for infinite loops (impossible targets)
   - Consider enumeration approach for tight capacity generators

4. **Memory issues**
   - Large datasets (600K+ samples) may require sufficient RAM
   - Consider generating in batches for very large targets

## License

[Add your license information here]

## Contact

[Add contact information or issue tracker links]

---

**Last Updated**: 2026-03-01  
**Version**: 1.3  
**Status**: Active Development

---

## Language Support

### English Datasets
- **Group 1**: Language and Literacy (690,400 samples)
- **Group 2**: Math and Numbers (382,886 samples)
- **Group 3**: Shapes, Colors & Patterns (128,302 samples)

### Indian Language Datasets (Group 1)
| Language | Script | Target | Key Features |
|----------|--------|--------|--------------|
| Hindi | Devanagari | 200,000 | 335 words, purna-viraam (।) separator |
| Punjabi | Gurmukhi | 200,000 | ~450 words, purna-viraam separator |
| Marathi | Devanagari | 200,000 | 335 words, purna-viraam separator |
| Kannada | Kannada | 200,000 | Akshara segmentation, S11 Ottakshara |
| Telugu | Telugu | 200,000 | Akshara segmentation, S11 Ottulu/Gunintalu |
| Assamese | Bengali-Assamese | 200,000 | Custom statement mapping (morphology, semantics) |
