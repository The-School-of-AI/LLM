# Kannada Curriculum Dataset — Technical Approach

## Overview

A curriculum dataset of ~200,000 Kannada question-answer pairs for language and literacy training. Each data point contains at least 512 tokens, combining multiple Q&A pairs per line.

**Output**: `curriculum_training_data/output/group1_kannada.txt`

---

## Format Specifications

- **Q&A format**: `Q? A.` — query ends with `?`, answer ends with period `.`
- **Combining**: Multiple pairs per line joined with spaces; each pair is `Q? A.`
- **Minimum tokens**: 512 per line (Kannada U+0C80–U+0CFF counted as 1 token each via `prompt_utils.count_tokens`)
- **Encoding**: UTF-8

**Utilities** (`prompt_utils.py`):
- `format_qa_pair_kannada(query, answer)` — formats one pair as `Q? A.`
- `combine_qa_pairs_to_reach_min_tokens_kannada(qa_pairs, min_tokens=512)` — combines pairs into lines of ≥512 tokens

---

## Akshara-Level Segmentation

Kannada uses syllabic units (aksharas), not raw Unicode graphemes. `kannada_grammar.get_kannada_aksharas(word)` segments words per Kannada linguistics:

- **Ottakshara** (conjuncts like ಸ್ಪ, ತ್ರೆ) = 1 unit
- **Anusvara** (ಂ) = part of preceding letter
- **Example**: ಆಸ್ಪತ್ರೆ → ಆ, ಸ್ಪ, ತ್ರೆ (3 aksharas); ಪುಸ್ತಕ → ಪು, ಸ್ತ, ಕ (3 aksharas)

All counting, position, spelling-listing, and last-letter logic use aksharas (S1, S2, S4, S7, S8, S9, S10).

---

## Vocabulary and Rhyme Logic

**`kannada_vocabulary.py`**:
- Word lists by category (Animals, Objects, Body, Colors, Nature, People, Food, Professions, etc.)
- ~950+ unique words in `ALL_WORDS_UNIQUE`
- `RHYMING_PAIRS` built via `build_real_rhyming_pairs()` — only real words; grouped by last akshara, paired cyclically for variety
- Classification categories: ವ್ಯಕ್ತಿ (person), ಪ್ರಾಣಿ (animal), ವಸ್ತು (object)
- Kannada number names 1–100 (ಒಂದು … ನೂರು)

**Grammar** (`kannada_grammar.get_genitive_suffix(word)`):
- Words ending in ಇ, ಈ, ಎ, ಏ → **ಯ** (e.g. ಗುಲಾಬಿ ಯ)
- Words ending in ಉ, ಊ, ಐ, ಓ, halant (್) → **ನ** (e.g. ಫೋನ್ ನ, ಬಸ್ ನ)
- Words ending in ಅ, ಆ or consonant → **ದ** (e.g. ನಕ್ಷತ್ರಮಂಡಲ ದ)
- Numerals → **ರ** (e.g. 72 ರ ಹೆಸರು)

For rhyme questions: use **ಪದವಿಗೆ** (e.g. "ನೀರು" ಪದವಿಗೆ ಪ್ರಾಸ ಪದ ಯಾವುದು?).

---

## Statement Types (S1–S11)

| Statement | Focus | Examples |
|-----------|-------|----------|
| S1 | Spelling + listing | "X" ಪದದ ಸ್ಪೆಲ್ಲಿಂಗ್ ಏನು? → comma-separated aksharas |
| S2 | Letter position | ಮೊದಲ/ಕೊನೆಯ/ಮಧ್ಯದ ಅಕ್ಷರ, ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ? |
| S3 | Sound matching | ಪ್ರಾಸಬದ್ಧ ಪದ, ಮೊದಲ ಧ್ವನಿ, ಪ್ರಾಸವಾಗುತ್ತವೆಯೇ? |
| S4 | Letter count | ಎಷ್ಟು ಅಕ್ಷರ? ಎರಡು ಅಕ್ಷರದ ಪದವೇ? |
| S5 | Rhyme | "X" ಪದವಿಗೆ ಪ್ರಾಸ ಪದ ಯಾವುದು? |
| S6 | Classification | ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು? |
| S7 | Position of letter | "X" ನಲ್ಲಿ "Y" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನ? |
| S8 | Number name/spelling | 11 ನ ಹೆಸರು? / "ಹನ್ನೊಂದು" ನ ಅಕ್ಷರಗಳು? |
| S9 | Last letter | "X" ನ ಕೊನೆಯ ಅಕ್ಷರ ಏನು? |
| S10 | Word comparison | ಯಾವ ಪದ ಉದ್ದ/ಕಿರಿದು? |
| S11 | Ottakshara & Kagunita | ಸಂಯುಕ್ತಾಕ್ಷರ, ಷ vs ಶ, ಅನುನಾಸಿಕ ಧ್ವನಿ, etc. |

---

## File Structure

```
curriculum_training_data/
├── group1_kannada/
│   ├── generate_group1_kannada_dataset.py
│   ├── generate_s1_spelling.py … generate_s11_ottakshara.py
│   ├── kannada_vocabulary.py
│   ├── kannada_grammar.py
│   └── KANNADA_DATASET_APPROACH.md
├── prompt_utils.py
└── output/
    └── group1_kannada.txt
```

---

## How to Run

From `curriculum_training_data/`:

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
python group1_kannada/generate_group1_kannada_dataset.py
```

The final script reads `group1_s1.txt` … `group1_s11.txt` from `group1_kannada/`, combines Q&A pairs to ≥512 tokens per line, and writes `output/group1_kannada.txt`.

---

## Target Distribution

| Statement | Target pairs |
|-----------|--------------|
| S1 | 28,600 |
| S2 | 25,800 |
| S3 | 20,000 |
| S4 | 25,800 |
| S5 | 20,000 |
| S6 | 20,000 |
| S7 | 17,200 |
| S8 | 10,000 |
| S9 | 17,200 |
| S10 | 11,000 |
| S11 | 10,000 |
| **Total** | **210,000** |
