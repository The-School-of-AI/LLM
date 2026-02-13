# Kannada Dataset Generation Approach

## Overview

This document describes the approach for generating a **Kannada** language curriculum dataset, mirroring the Hindi (`group1_hindi`) implementation. The goal is to create 200,000 question-answer pairs in Kannada script, with each data point containing at least 512 tokens.

**Purpose**: Generate Kannada Q&A pairs for language and literacy training  
**Scope**: Statement types S1–S11; S1–S4 and S3 use user-specified question templates; S11 = Ottakshara & Kagunita  
**Output**: Single TXT file `output/group1_kannada.txt`

## Relationship to Hindi Implementation

The Kannada pipeline is a **direct port** of the Hindi design:

- Same **statement types** (S1–S10) and **target counts** (same distribution).
- Same **format**: queries end with `?`, answers end with `।` (purna-viraam); spacing `Q? A।`.
- Same **token rule**: minimum 512 tokens per data point; `prompt_utils.count_tokens` supports Kannada (U+0C80–U+0CFF) so each Kannada character counts as 1 token.
- Same **character logic**: Unicode characters for spelling (S1, S8); grapheme clusters (`regex` `\X`) for counting/position (S2, S4, S7, S9, S10).
- **Shared utilities**: `format_qa_pair_hindi` and `combine_qa_pairs_to_reach_min_tokens_hindi` are reused (Kannada also uses `।`).

## Vocabulary and Categories

- **Kannada vocabulary** is in `kannada_vocabulary.py`, aligned with Hindi categories:
  - Animals (ಪ್ರಾಣಿ), Objects & Places, Body Parts (ದೇಹ ಭಾಗಗಳು), Colors (ಬಣ್ಣ), Nature (ಪ್ರಕೃತಿ), People & Family, Food (ಆಹಾರ), Professions (ವೃತ್ತಿ), Vehicles, Household, Abstract, Days, Months, Numbers 1–100.
- **Rhyming pairs** (`RHYMING_PAIRS`) and **classification categories** (ವ್ಯಕ್ತಿ / ಪ್ರಾಣಿ / ವಸ್ತು) are defined for Kannada.
- **Numbers 1–100** are given in Kannada words (ಒಂದು, ಎರಡು, … ನೂರು).

## Format Specifications

- **Pattern**: `Q? A।` per pair; multiple pairs on one line separated by `। `.
- **Rules**: All queries end with `?`; all answers end with `।`; no `।` inside queries.
- **Encoding**: UTF-8.

## Statement Types (Kannada Terminology)

**Correct Kannada terms (no "ವರ್ತನಿ")**:
- **ಕಾಗುಣಿತ** (kāguṇita) = spelling (e.g. ಕಾಗುಣಿತ ಪರೀಕ್ಷೆ = spelling test)
- **ಪದಬರಿಗೆ** = spelling (alternative)
- **ಅಕ್ಷರ** = letter/character; **ಅಕ್ಷರಮಾಲೆ** = alphabet
- **ವರ್ಣ** = letter (also used in grammar)
- **ಪ್ರಾಸ** = rhyme (not ತುಕಬಂದಿ)

**Genitive suffix (ನ, ಯ, ದ, ರ)** — see `kannada_grammar.get_genitive_suffix(word)`:
- Words ending in ಇ, ಈ, ಎ, ಏ (ಿ, ೀ, ೆ, ೇ) → **ಯ** (e.g. ಗುಲಾಬಿ ಯ, ಕರುಣೆ ಯ)
- Words ending in ಉ, ಊ, ಐ, ಓ, etc. (ು, ೂ, ೈ, ೊ, ೋ, ೌ), consonant ಯ, or **್ (halant)** → **ನ** (e.g. ಜುಲೈ ನ, ಉಪಾಧ್ಯಾಯ ನಲ್ಲಿ, **ಫೋನ್ ನ**, ಬಸ್ ನ)
- Words ending in ಅ, ಆ or other consonant → **ದ** (e.g. ನಕ್ಷತ್ರಮಂಡಲ ದ)
- Numerals (72, 17) → **ರ** (72 ರ ಹೆಸರು)

**Vocabulary**: Only common, everyday Kannada words are used. Avoid non-existent or very complex compounds (e.g. ತರಕಾರಿವಿಕ್ರೇತ, ಕುಂಜಡಿಗ removed; ಅಂಗಡಿದಾರ used for shopkeeper).

**Question patterns**: Each statement type uses multiple phrasings so questions are not repetitive (e.g. different openings, "ತಿಳಿಸಿ" vs "ಏನು?", "ಪದದ" vs "{suffix}ಲ್ಲಿ", etc.).

| Statement | Kannada focus        | Question idea (Kannada) |
|-----------|----------------------|--------------------------|
| S1        | Spelling + Listing   | "X" ಪದದ ಸ್ಪೆಲ್ಲಿಂಗ್ ಏನು? / ಅಕ್ಷರಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ? → comma-separated chars |
| S2        | Letter position      | "X" ಪದದ ಮೊದಲ/ಕೊನೆಯ/ಮೂರನೇ/ಮಧ್ಯದ ಅಕ್ಷರ? ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ? ಕೊನೆಯಲ್ಲಿದೆಯೇ? → cluster or ಹೌದು/ಇಲ್ಲ |
| S3        | Sound matching       | ಪ್ರಾಸಬದ್ಧ ಪದ, ಅಕ್ಷರದಿಂದ ಪ್ರಾರಂಭ, ಪ್ರಾಸವಾಗುತ್ತವೆಯೇ?, ಧ್ವನಿ/ಉಚ್ಚಾರಣೆ, ಪ್ರಾಣಿ ಹೆಸರು, ಮೊದಲ ಧ್ವನಿ → word or ಹೌದು/ಇಲ್ಲ |
| S4        | Letter count         | "X" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರ/ವರ್ಣ? ಎಣಿಸಿ, ಲೆಕ್ಕ ಕೊಡಿ; ಎರಡು ಅಕ್ಷರದ ಪದವೇ? → number or ಹೌದು/ಇಲ್ಲ |
| S5        | Rhyming (ಪ್ರಾಸ)      | "X" ಪದವಿಗೆ ಪ್ರಾಸ ಪದ ಯಾವುದು? → rhyme word |
| S6        | Classification       | "X" ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು? → category |
| S7        | Position of letter   | "X" ನಲ್ಲಿ "Y" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನ? → position |
| S8        | Number name/spelling | 11 ನ ಹೆಸರು ಏನು? / "ಹನ್ನೊಂದು" ನ ಅಕ್ಷರಗಳು? → name or spelling |
| S9        | Last letter          | "X" ನ ಕೊನೆಯ ಅಕ್ಷರ ಏನು? → last grapheme cluster |
| S10       | Word comparison      | ಯಾವ ಪದ ಉದ್ದ/ಕಿರಿದು? → longer/shorter word |
| S11       | Ottakshara & Kagunita| ಒತ್ತಕ್ಷರ, ಋ-ಕಾರ, ಸಂಯುಕ್ತಾಕ್ಷರ, ಓ/ಐ/ಔ-ತ್ವ, ಗುಣಿತಾಕ್ಷರ, ಷ vs ಶ, ಕ್ಷ ಸ್ವರ/ವ್ಯಂಜನ? → language-specific answers |

Position names used: ಮೊದಲನೇ, ಎರಡನೇ, ಮೂರನೇ, ನಾಲ್ಕನೇ, ಐದನೇ, ಆರನೇ, ಏಳನೇ, ಎಂಟನೇ, ಒಂಬತ್ತನೇ, ಹತ್ತನೇ.

## Character Splitting (Akshara-Level)

- **Spelling (S1, S8)**: Uses aksharas for listing; `get_kannada_characters(word)` for raw Unicode.
- **Counting/position (S2, S4, S7, S9, S10)**: `get_kannada_aksharas(word)` — syllabic units per Kannada linguistics:
  - Ottakshara (conjuncts like ಸ್ಪ, ತ್ರೆ) = 1 unit
  - Anusvara (ಂ) = part of preceding letter
  - E.g. ಆಸ್ಪತ್ರೆ = 3 aksharas (ಆ, ಸ್ಪ, ತ್ರೆ); ಪುಸ್ತಕ = 3 (ಪು, ಸ್ತ, ಕ)

## File Structure

```
curriculum_training_data/
├── group1_kannada/
│   ├── generate_group1_kannada_dataset.py
│   ├── generate_s1_spelling.py .. generate_s10_compare.py, generate_s11_ottakshara.py
│   ├── kannada_vocabulary.py
│   └── KANNADA_DATASET_APPROACH.md
├── prompt_utils.py   # count_tokens supports Kannada; format/combine shared
└── output/
    └── group1_kannada.txt
```

## How to Run

1. **Generate per-statement files** (run from `curriculum_training_data/` so imports resolve):
   ```bash
   cd curriculum_training_data
   python group1_kannada/generate_s1_spelling.py
   python group1_kannada/generate_s2_letter_position.py
   python group1_kannada/generate_s3_sound.py
   python group1_kannada/generate_s4_count.py
   python group1_kannada/generate_s5_rhyme.py
   python group1_kannada/generate_s6_classify.py
   python group1_kannada/generate_s7_position.py
   python group1_kannada/generate_s8_numbers.py
   python group1_kannada/generate_s9_last.py
   python group1_kannada/generate_s10_compare.py
   python group1_kannada/generate_s11_ottakshara.py
   ```
2. **Combine into final dataset**:
   ```bash
   python group1_kannada/generate_group1_kannada_dataset.py
   ```
   This reads `group1_s1.txt` … `group1_s11.txt` from `group1_kannada/`, combines to ≥512 tokens per line, and writes `output/group1_kannada.txt`.

## Distribution Summary

| Statement | Pairs  | Percentage |
|-----------|--------|------------|
| S1        | 28,600 | 13.6%      |
| S2        | 25,800 | 12.3%      |
| S3        | 20,000 | 9.5%       |
| S4        | 25,800 | 12.3%      |
| S5        | 20,000 | 9.5%       |
| S6        | 20,000 | 9.5%       |
| S7        | 17,200 | 8.2%       |
| S8        | 10,000 | 4.8%       |
| S9        | 17,200 | 8.2%       |
| S10       | 11,000 | 5.2%       |
| S11       | 10,000 | 4.8%       |
| **Total** | **210,000** | **100%** |

## Token Counting (Kannada)

In `prompt_utils.count_tokens`, characters in the Kannada Unicode block (U+0C80–U+0CFF) are counted as **1 token each**, consistent with the spelling format and with the Hindi/Devanagari treatment.
