# Group1 Marathi Dataset Generation

This folder contains Marathi translations of the Hindi group1 curriculum dataset generation system.

## Overview
- **Purpose**: Generate 200,000 Marathi Q&A pairs for language and literacy training

**Command to generate all files**:

```sh
python3 generate_s1_spelling.py && python3 generate_s2_letter_position.py && python3 generate_s3_sound.py && python3 generate_s4_count.py && python3 generate_s5_rhyme.py && python3 generate_s6_classify.py && python3 generate_s7_position.py && python3 generate_s8_numbers.py && python3 generate_s9_last.py && python3 generate_s10_compare.py && python3 generate_group1_marathi_dataset.py
```
- **Output**: `output/group1_marathi.txt`
- **Format**: Continuous Q?A pairs with purna-viram (।)

## Files Included

### Complete Files:
1. `marathi_vocabulary.py` - Marathi word lists (translated from Hindi)
2. `generate_s1_spelling.py` - Sample spelling generator

### Files to Complete:
You need to create these files by translating from group1_hindi:
- `generate_s2_letter_position.py` - Letter position questions
- `generate_s3_sound.py` - Sound matching
- `generate_s4_count.py` - Letter counting
- `generate_s5_rhyme.py` - Rhyming words
- `generate_s6_classify.py` - Word classification
- `generate_s7_position.py` - Position of letter
- `generate_s8_numbers.py` - Number spelling
- `generate_s9_last.py` - Last letter
- `generate_s10_compare.py` - Word comparison
- `generate_group1_marathi_dataset.py` - Main generator
- `MARATHI_DATASET_APPROACH.md` - Documentation

## Key Marathi Terminology
- Hindi: वर्तनी → Marathi: शब्दलेखन (spelling)
- Hindi: की वर्तनी क्या है? → Marathi: चे शब्दलेखन काय आहे?
- Hindi: पहला, दूसरा, तीसरा → Marathi: पहिला, दुसरा, तिसरा
- Hindi: अक्षर → Marathi: अक्षर (same)
- Hindi: में कितने अक्षर हैं? → Marathi: मध्ये किती अक्षरे आहेत?

## Format Requirements
- ALL queries MUST end with "?"
- ALL answers MUST end with "।" (purna-viram)
- Pattern: `Q? A। Q? A।`
- Minimum 512 tokens per datapoint

## Sample Vocabulary Translations

### Animals:
- कुत्ता → कुत्रा (dog)
- बिल्ली → मांजर (cat)
- मुर्गी → कोंबडी (chicken)

### Common Words:
- पानी → पाणी (water)
- घर → घर (home - same)
- किताब → पुस्तक (book)

### Body Parts:
- हाथ → हात (hand)
- पैर → पाय (foot)
- आंख → डोळा (eye)

## How to Complete

1. **Translate Vocabulary**: Expand `marathi_vocabulary.py` with all categories
2. **Translate Question Templates**: For each generator (S2-S10), translate Hindi templates to Marathi
3. **Test Each Generator**: Run individually to verify output
4. **Create Main Generator**: Combine all 10 statement types
5. **Validate Output**: Ensure format correctness and token count

## Reference
See the original `group1_hindi` folder for complete implementation.

## Note
This is a SAMPLE package with core vocabulary and one generator. Complete the remaining generators by following the Hindi patterns and translating to natural Marathi.
