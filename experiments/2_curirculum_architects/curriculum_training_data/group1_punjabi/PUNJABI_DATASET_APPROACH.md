# Punjabi Dataset Generation Approach

## Overview

This document describes the approach for generating a Punjabi language curriculum dataset similar to the English `group1` dataset. The goal is to create 200,000 question-answer pairs in Gurmukhi script, with each data point containing at least 512 tokens.

**Purpose**: Generate Punjabi Q&A pairs for language and literacy training  
**Scope**: All 10 statement types adapted for Punjabi/Gurmukhi script  
**Output**: Single TXT file `output/group1_punjabi.txt`

## Vocabulary Sources & Verification

### Word Collection

- **Total unique Punjabi words**: ~350 verified words (after deduplication)
- **Sources**: Learnpunjabi.org, PunjabiCharm, basic vocabulary lists
- **Word distribution by difficulty**:
  - Easy words (2-4 characters): ~150 unique words
  - Medium words (5-6 characters): ~100 unique words
  - Hard words (7+ characters): ~50 unique words

### Categories

Words are organized into categories:
- Animals (ਜਾਨਵਰ)
- Objects (ਵਸਤੂ)
- Body Parts (ਸਰੀਰ ਦੇ ਅੰਗ)
- Colors (ਰੰਗ)
- Nature (ਕੁਦਰਤ)
- People (ਵਿਅਕਤੀ)
- Food (ਖਾਣਾ)
- Professions (ਕਿੱਤੇ)
- Numbers (1-100)

## Format Specifications

### Pattern

- **Single pair**: `Q? A।`
- **Multiple pairs**: `Q? A। Q? A। Q? A। ...`
- **Spacing**: Exactly one space after `?` and exactly one space after `।`
- **No line breaks**: All pairs on same line, separated by `। ` (danda + space)

### Critical Format Rules

1. **ALL queries MUST end with "?"**
2. **NEVER use "।" in queries**
3. **ALL answers MUST end with "।"**

## Statement Types Breakdown

### Statement 1: Spelling (ਵਰਤਨੀ) - 28,600 pairs (14.3%)
- `"ਘਰ" ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?` → `ਘ, ਰ।`
- Uses detailed Unicode character split.

### Statement 2: Letter at Position (ਅੱਖਰ ਸਥਿਤੀ) - 25,800 pairs (12.9%)
- `"ਕਿਤਾਬ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ?` → `ਕਿ।`
- Uses grapheme clusters.

### Statement 3: Sound Matching (ਧੁਨੀ ਮਿਲਾਨ) - 20,000 pairs (10%)
- `ਕਿਹੜਾ ਸ਼ਬਦ "/ਕ/" ਧੁਨੀ ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ, "ਕੁੱਤਾ" ਜਾਂ "ਬਿੱਲੀ"?` → `ਕੁੱਤਾ।`

### Statement 4: Letter Count (ਅੱਖਰ ਗਿਣਤੀ) - 25,800 pairs (12.9%)
- `"ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?` → `2।`

### Statement 5: Rhyming (ਤੁਕਬੰਦੀ) - 20,000 pairs (10%)
- `"{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਨ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{rhyme}" ਜਾਂ "{non_rhyme}"?`

### Statement 6: Classification (ਸ਼੍ਰੇਣੀਬੱਧਤਾ) - 20,000 pairs (10%)
- `"{word}" ਇੱਕ ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ ਹੈ?`

### Statement 7: Position of Letter (ਅੱਖਰ ਦੀ ਸਥਿਤੀ) - 17,200 pairs (8.6%)
- `"{word}" ਵਿੱਚ "{char}" ਅੱਖਰ ਕਿਸ ਸਥਾਨ ਤੇ ਹੈ?`

### Statement 8: Number Spelling (ਸੰਖਿਆ ਵਰਤਨੀ) - 10,000 pairs (5%)
- `11 ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?` → `ਗਿਆਰਾਂ।`

### Statement 9: Last Letter (ਆਖਰੀ ਅੱਖਰ) - 17,200 pairs (8.6%)
- `"{word}" ਦਾ ਆਖਰੀ ਅੱਖਰ ਕੀ ਹੈ?`

### Statement 10: Word Comparison (ਸ਼ਬਦ ਤੁਲਨਾ) - 11,000 pairs (5.5%)
- `ਕਿਹੜਾ ਸ਼ਬਦ ਲੰਬਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?`

## Character Splitting Methodology

1. **Detailed Unicode Character Split** (S1, S8): Each Unicode codepoint is a separate character.
2. **Grapheme Cluster Split** (S2, S4, S7, S9, S10): Uses `regex` library's `\X` pattern for user-perceived characters.

## Implementation Details

- All scripts use UTF-8 encoding.
- Dataset follows the structure of English `group1` but adapted for Punjabi.
- Minimum 512 tokens per data point achieved by concatenation.
