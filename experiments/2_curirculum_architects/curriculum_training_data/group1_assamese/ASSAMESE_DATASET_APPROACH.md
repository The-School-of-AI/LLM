# Assamese Dataset Generation Approach

## Overview

This document describes the approach for generating an Assamese language curriculum dataset similar to the Hindi `group1_hindi` dataset. The goal is to create 200,000 question-answer pairs in Bengali-Assamese script, with each data point containing at least 512 tokens.

**Purpose**: Generate Assamese Q&A pairs for language and literacy training  
**Scope**: All 10 statement types adapted for Assamese/Bengali-Assamese script  
**Output**: Single TXT file `output/group1_assamese.txt`

## Script & Vocabulary

### Bengali-Assamese Script

- Assamese uses the **Bengali-Assamese script** (Unicode U+0980 to U+09FF)
- Same script family as Bengali; Assamese has some unique characters (e.g., ৰ, ৱ)
- Token counting in `prompt_utils.py` supports Bengali-Assamese: each Unicode character = 1 token

### Vocabulary Structure

- **assamese_vocabulary.py**: Core word lists organized by difficulty and category (Animals, Objects, etc.)
- **assamese_vocabulary_expanded.py**: Extended vocabulary (~1,200 words) added to support high-volume generation for S4, S1, and S9. Includes:
  - **Yuktakshars**: 300+ complex conjunct words
  - **Polysyllabic Words**: 200+ long words (5+ aksharas)
  - **Inflected Verbs**: 200+ verb forms (roots + suffixes)
  - **Specialized Nouns**: 200+ administrative, technical, and abstract terms

## Format Specifications

### Pattern

- **Single pair**: `Q? A।`
- **Multiple pairs**: `Q? A। Q? A। Q? A। ...`
- **Punctuation**: Queries end with `?`; answers end with `।` (purna-viraam)

### Example

```
"ঘৰ"ৰ বানান কি? ঘ, ৰ। "পানী"ৰ বানান কি? প,া, ন, ী।
```

## Statement Types (Revised for Assamese)

| Statement | Skill | Target Pairs | Focus |
|-----------|-------|--------------|-------|
| S1 | Spelling (বানান) | 25,000 | Orthography, Yuktakshars (conjuncts) |
| S2 | Positional Analysis | 20,000 | Merged S2+S7 (Letter at index X) |
| S3 | Phonetic Matching | 15,000 | Sibilants (শ/ষ/স), Wa/Ba distinction |
| S4 | Akshara Count | 15,000 | Visual vs phonetic units (uses expanded vocab) |
| S5 | Rhyming | 15,000 | Verb endings, standard rhymes |
| S6 | Classification | 20,000 | Semantic categories, Action vs Object |
| S7 | Numeric Mastery | 15,000 | Numbers, Ordinals |
| S8 | Word Boundaries | 15,000 | First/Last letter, Prefix |
| S9 | Morphology | 30,000 | Roots + Suffixes (Bibhakti) |
| S10 | Semantics | 30,000 | Synonyms, Antonyms |
| **Total** | | **200,000** | |

## File Structure

```
curriculum_training_data/
├── group1_assamese/
│   ├── assamese_vocabulary.py          # Core Word lists
│   ├── assamese_vocabulary_expanded.py # Extended Word lists
│   ├── generate_group1_assamese_dataset.py  # Main aggregator
│   ├── generate_s1_spelling.py         # S1: Spelling
│   ├── generate_s2_position.py         # S2: Positional Analysis
│   ├── generate_s3_phonetic.py         # S3: Phonetic Matching
│   ├── generate_s4_count.py            # S4: Character Count
│   ├── generate_s5_rhyme.py            # S5: Rhyming
│   ├── generate_s6_classify.py         # S6: Classification
│   ├── generate_s7_numbers.py          # S7: Numeric Mastery
│   ├── generate_s8_boundaries.py       # S8: Word Boundaries
│   ├── generate_s9_morphology.py       # S9: Morphology
│   ├── generate_s10_semantics.py       # S10: Semantics
│   └── ASSAMESE_DATASET_APPROACH.md    # This document
└── output/
    └── group1_assamese.txt             # Final output (200k pairs)
```

## Character Splitting

- **Spelling (S1, S8)**: Unicode character split (each codepoint separate)
- **Counting/Position (S2, S4, S7, S9, S10)**: Grapheme clusters via `\X` pattern

## How to Run

1. **Statement Generators**: The scripts `generate_s1_spelling.py` through `generate_s10_semantics.py` are configured with Assamese logic and templates.
2. **Grapheme Clusters**: The scripts utilize `regex.findall(r"\X", word)` for counting and positional analysis to correctly handle Assamese conjuncts.
3. **Execution**: 
   - First, execute each `generate_s*.py` script to create the individual statement files.
   - Then, run `generate_group1_assamese_dataset.py` to aggregate them into the final dataset.

