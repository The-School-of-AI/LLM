# Telugu Curriculum Dataset (Group 1) — Validation Report

**Date:** 2026-02-13
**Branch:** `p02/feat/curriculum_data`
**Output:** `output/group1_telugu.txt` (27 MB)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Q&A pairs | **208,000** |
| Statement types | 11 (S1–S11) |
| Final data points (combined lines) | **17,738** |
| Min tokens per data point | **512** |
| Max tokens per data point | **791** |
| Avg tokens per data point | **535.5** |
| Lines below 512 tokens | **0** |
| Vocabulary (unique words) | **958** |
| Output file size | **27 MB** |

---

## 2. Statement-Level Counts

All 11 generators produce the exact target count:

| Statement | Description | Target | Actual | Status |
|-----------|-------------|--------|--------|--------|
| S1 | Spelling (అక్షరక్రమం) | 30,000 | 30,000 | PASS |
| S2 | Letter Position (అక్షర స్థానం) | 26,000 | 26,000 | PASS |
| S3 | Sound Matching (ధ్వని) | 20,000 | 20,000 | PASS |
| S4 | Letter Count (అక్షర గణన) | 26,000 | 26,000 | PASS |
| S5 | Rhyming (ప్రాస) | 20,000 | 20,000 | PASS |
| S6 | Classification (వర్గీకరణ) | 20,000 | 20,000 | PASS |
| S7 | Position of Letter (అక్షరం స్థానం) | 18,000 | 18,000 | PASS |
| S8 | Number Spelling (సంఖ్య అక్షరక్రమం) | 12,000 | 12,000 | PASS |
| S9 | Last Letter (చివరి అక్షరం) | 18,000 | 18,000 | PASS |
| S10 | Word Comparison (పద పోలిక) | 10,000 | 10,000 | PASS |
| S11 | Ottulu & Gunintalu (ఒత్తులు & గుణింతాలు) | 8,000 | 8,000 | PASS |
| **Total** | | **208,000** | **208,000** | **PASS** |

---

## 3. Akshara Segmentation Tests

Core segmentation function `get_telugu_aksharas()` tested with 8 cases covering simple words, conjuncts, anusvara, and complex clusters:

| Word | Expected Aksharas | Count | Result | Status |
|------|-------------------|-------|--------|--------|
| పుస్తకం | పు, స్త, కం | 3 | పు, స్త, కం | PASS |
| అమ్మ | అ, మ్మ | 2 | అ, మ్మ | PASS |
| నీరు | నీ, రు | 2 | నీ, రు | PASS |
| విద్యార్థి | వి, ద్యా, ర్థి | 3 | వి, ద్యా, ర్థి | PASS |
| జ్ఞానం | జ్ఞా, నం | 2 | జ్ఞా, నం | PASS |
| కుక్క | కు, క్క | 2 | కు, క్క | PASS |
| బడి | బ, డి | 2 | బ, డి | PASS |
| విద్యాలయం | వి, ద్యా, ల, యం | 4 | వి, ద్యా, ల, యం | PASS |

Algorithm: `regex.findall(r"\X", word)` + virama (్ U+0C4D) merging to form conjunct aksharas.

---

## 4. Vocabulary Validation

| Category | Unique Count |
|----------|-------------|
| Easy words | 278 |
| Medium words | 354 |
| Hard words | 79 |
| **Total unique** | **958** |
| Rhyming pairs | 834 |
| Number words (1–100) | 100 |
| Varga consonant groups | 7 |
| Classification categories | 3 |
| Classification words | 365 |

Categories breakdown:
- **Easy**: Animals (44), Objects (69), Body Parts (39), Colors (18), Nature (45), People (30), Food (40)
- **Medium**: Animals (30), Objects (64), Professions (45), Nature (60), Vehicles (35), Food (80), Household (48)
- **Hard**: Complex Nouns (30), Abstract (30), Days (7), Months (12)
- **Additional**: ~250 supplementary words

Classification categories: జంతువు (animal), వ్యక్తి (person), వస్తువు (object)

---

## 5. Cross-Script Leakage Check

| Script | Characters Found | Status |
|--------|-----------------|--------|
| Kannada (U+0C80–U+0CFF) | **0** | PASS |
| Hindi/Devanagari (U+0900–U+097F) | **0** | PASS |

No cross-script contamination in the final output.

---

## 6. Token Count Verification

Every line in the final output meets the 512-token minimum:

| Metric | Value |
|--------|-------|
| Total data points | 17,738 |
| Min tokens | 512 |
| Max tokens | 791 |
| Avg tokens | 535.5 |
| Lines < 512 tokens | **0** |

Token counting uses Telugu-aware `count_tokens_telugu()` where each Telugu Unicode character (U+0C00–U+0C7F) = 1 token.

---

## 7. Sample Output (Spot Check)

### S1: Spelling
```
"సూది" పదంలోని అక్షరాలను ఒక్కొక్కటిగా చెప్పండి? సూ, ది.
```

### S2: Letter Position
```
"పొట్ట" పదంలో మధ్య అక్షరం ఏమిటి? ట్ట.
```

### S3: Sound Matching
```
"విటమిన్" పదానికి సమానమైన ధ్వని ఉన్న పదం ఏది? ప్రోటీన్.
```

### S4: Letter Count
```
"నీటిగుర్రం" పదంలోని అక్షరాల సంఖ్య ఎంత? 4 అక్షరాలు.
```

### S5: Rhyming
```
"గ్యాస్" పదానికి ప్రాస పదం ఏది, "సైనికుడు" లేదా "చెస్"? చెస్.
```

### S6: Classification
```
"తాడు" ఏమిటి, వ్యక్తి, జంతువు లేదా వస్తువు? వస్తువు.
```

### S7: Position of Letter
```
"సూర్యాస్తమయం" పదంలో "స్త" ఎంతవ స్థానంలో వస్తుంది? మూడవ.
```

### S8: Number Spelling
```
"నలభై రెండు" పదంలోని అక్షరాలు ఏమిటి? న, ల, భై, రెం, డు.
```

### S9: Last Letter
```
"కొంగ" లో చివరి అక్షరం ఏది? గ.
```

### S10: Word Comparison
```
"మైసూర్ పాక్" మరియు "పెట్రోలు" లలో చిన్న పదం ఏది? పెట్రోలు.
```

### S11: Ottulu & Gunintalu
```
"ఢ" యొక్క గుణింతాలు చెప్పండి? ఢ, ఢా, ఢి, ఢీ, ఢు, ఢూ, ఢృ, ఢె, ఢే, ఢై, ఢొ, ఢో, ఢౌ.
"క్ష" లో ఏయే వ్యంజనాలు కలిసి ఉన్నాయి? క, ష.
"గో" అక్షరంలో ఉన్న మూల హల్లు ఏమిటి? గ.
"సమాజం" పదంలో ఒత్తు ఉందా? లేదు.
"వాతావరణశాస్త్రం" పదంలో ఒత్తు ఏమిటి? స్త్రం.
```

---

## 8. S11 Design Rationale

### Why S11 was added

S1–S10 treat aksharas as **atomic units** — the model learns to spell, count, position, and compare aksharas but never understands what's *inside* them.

S11 teaches the **compositional structure** of Telugu script:

| Concept | What it teaches | Example |
|---------|----------------|---------|
| **Gunintalu (గుణింతాలు)** | Consonant + vowel sign = combined form | క + ఆ-కారం = కా |
| **Identify base consonant** | Extract root consonant from combined form | కీ → క |
| **Identify vowel sign** | Extract vowel sign from combined form | కీ → ఈ-కారం |
| **Gunintam chart** | Full vowel sign series for a consonant | క, కా, కి, కీ, కు, కూ, ... |
| **Ottulu (ఒత్తులు)** | Identify conjuncts in words | అమ్మ → మ్మ |
| **Conjunct decomposition** | Break conjunct into component consonants | స్త → స, త |
| **Conjunct detection** | Does a word contain conjuncts? | సమాజం → లేదు |
| **Vowel/consonant classification** | Is a character స్వరం or వ్యంజనం? | క → వ్యంజనం |

### S11 pair count: 8,000

| Component | Seed pairs | Fill | Allocation |
|-----------|-----------|------|------------|
| Gunintalu combinations (37 consonants × 16 vowels) | ~592 | ~2,200 | ~35% |
| Base consonant identification | ~148 | ~800 | ~12% |
| Vowel sign identification | ~148 | ~800 | ~12% |
| Gunintam charts (full series) | ~37 | ~1,200 | ~15% |
| Ottulu in words | ~400 | ~700 | ~14% |
| Conjunct decomposition | ~100 | ~200 | ~4% |
| Conjunct detection (yes/no) | ~300 | ~200 | ~6% |
| Vowel/consonant classification | ~53 | ~150 | ~2% |

**Consonants (37)**: Full traditional alphabet including all vargas (క–ఙ, చ–ఞ, ట–ణ, త–న, ప–మ), semi-vowels/sibilants (య, ర, ల, వ, శ, ష, స, హ, ళ), plus ఱ (hard ra) and క్ష (traditionally taught as alphabet entry).

**Vowel signs (16)**: అ (inherent, no sign), ఆ–ఔ (12 standard vowel signs), ౠ (long vocalic r), అం (anusvara), అః (visarga).

**vs Kannada S11**: Kannada used only 20 hardcoded Q&A pairs repeated to fill 10K. Telugu S11 generates **programmatically diverse** pairs from 37 consonants × 16 vowel signs × multiple template variants + vocabulary-based ottulu detection.

---

## 9. File Inventory

### Foundation Files
| File | Purpose | Size |
|------|---------|------|
| `prompt_utils_telugu.py` | Token counting, QA formatting, line combining | 4.6 KB |
| `telugu_grammar.py` | Akshara segmentation (virama merging) | 1.2 KB |
| `telugu_vocabulary.py` | 958 unique words, numbers, vargas, rhyming | 38 KB |

### Generators
| File | Statement | Target |
|------|-----------|--------|
| `generate_s1_spelling.py` | S1: Spelling | 30,000 |
| `generate_s2_letter_position.py` | S2: Letter Position | 26,000 |
| `generate_s3_sound.py` | S3: Sound Matching | 20,000 |
| `generate_s4_count.py` | S4: Letter Count | 26,000 |
| `generate_s5_rhyme.py` | S5: Rhyming | 20,000 |
| `generate_s6_classify.py` | S6: Classification | 20,000 |
| `generate_s7_position.py` | S7: Position of Letter | 18,000 |
| `generate_s8_numbers.py` | S8: Number Spelling | 12,000 |
| `generate_s9_last.py` | S9: Last Letter | 18,000 |
| `generate_s10_compare.py` | S10: Word Comparison | 10,000 |
| `generate_s11_ottulu_gunintalu.py` | S11: Ottulu & Gunintalu | 8,000 |

### Orchestrator & Output
| File | Purpose | Size |
|------|---------|------|
| `generate_group1_telugu_dataset.py` | Combines S1–S11, enforces 512-token min | 3.4 KB |
| `output/group1_telugu.txt` | Final dataset | 27 MB |

### Generated Data Files
| File | Lines | Size |
|------|-------|------|
| `group1_s1.txt` | 30,000 | 4.4 MB |
| `group1_s2.txt` | 26,000 | 3.1 MB |
| `group1_s3.txt` | 20,000 | 2.4 MB |
| `group1_s4.txt` | 26,000 | 3.4 MB |
| `group1_s5.txt` | 20,000 | 3.3 MB |
| `group1_s6.txt` | 20,000 | 2.7 MB |
| `group1_s7.txt` | 18,000 | 2.0 MB |
| `group1_s8.txt` | 12,000 | 1.4 MB |
| `group1_s9.txt` | 18,000 | 1.7 MB |
| `group1_s10.txt` | 10,000 | 1.4 MB |
| `group1_s11.txt` | 8,000 | 0.8 MB |

---

## 10. Telugu-Specific Design Decisions

| Decision | Details |
|----------|---------|
| **Akshara-level segmentation** | Conjuncts (సంయుక్తాక్షరాలు) treated as single units, NOT split at Unicode character level |
| **No genitive suffix system** | Telugu uses invariant postpositions (లో, యొక్క, లోని) — eliminates Kannada's 4-suffix system |
| **Answer terminator** | Period (`.`), NOT danda (`।`) |
| **Yes/No for identity** | అవును / కాదు |
| **Yes/No for existence** | అవును / లేదు |
| **Telugu ordinals** | మొదటి (1st, irregular), then {cardinal}వ pattern (రెండవ, మూడవ, ...) |
| **Number words** | Irregular teens (11–19 are fused), compound 21+ (ఇరవై ఒకటి) |
| **Separate prompt_utils** | `prompt_utils_telugu.py` — does NOT modify shared `prompt_utils.py` |
| **Spelling terminology** | అక్షరక్రమం (NOT వర్తని which is Hindi-influenced) |
| **S11 programmatic generation** | Generates from consonant × vowel sign matrix, not hardcoded pairs |

---

## 11. How to Reproduce

```bash
cd experiments/2_curirculum_architects/curriculum_training_data

# Run all generators
uv run python group1_telugu/generate_s1_spelling.py
uv run python group1_telugu/generate_s2_letter_position.py
uv run python group1_telugu/generate_s3_sound.py
uv run python group1_telugu/generate_s4_count.py
uv run python group1_telugu/generate_s5_rhyme.py
uv run python group1_telugu/generate_s6_classify.py
uv run python group1_telugu/generate_s7_position.py
uv run python group1_telugu/generate_s8_numbers.py
uv run python group1_telugu/generate_s9_last.py
uv run python group1_telugu/generate_s10_compare.py
uv run python group1_telugu/generate_s11_ottulu_gunintalu.py

# Combine into final dataset
uv run python group1_telugu/generate_group1_telugu_dataset.py

# Verify counts
for f in group1_telugu/group1_s*.txt; do echo "$f: $(wc -l < $f)"; done
```

---

## 12. Test Suite

**183 tests** across 4 test files, all passing in ~3.10s.

**Run:** `uv run python -m pytest group1_telugu/tests/ -v`

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_telugu_grammar.py` | 24 | Virama detection, akshara segmentation (15 words), reconstruction invariant, empty input |
| `test_prompt_utils_telugu.py` | 25 | Token counting (Telugu/Kannada/Devanagari/English/mixed/empty), formatting helpers, QA pair formatting, combining |
| `test_telugu_vocabulary.py` | 34 | Word counts (≥950 unique), no duplicates, category minimums (16 categories), Telugu script validation, no Kannada/Hindi chars, numbers (100), days/months, vargas (7 groups), classification (3 categories), rhyming pairs (≥100) |
| `test_telugu_generators.py` | 100 | File existence (11 statement files + final output), line counts (all 11 match targets), total = 208,000, format validation (Q? A. on first 100 lines), no danda (।), cross-script leakage (all files), token minimums (≥512), S11 gunintalu/ottulu content checks |
| **Total** | **183** | |

### Key Test Categories

**Grammar (`test_telugu_grammar.py`)**
- Virama (్) detection: present, absent, empty, standalone
- Akshara segmentation: simple words (బడి, నీరు), gemination (అమ్మ, కుక్క), complex conjuncts (పుస్తకం, విద్యార్థి, జ్ఞానం), anusvara (నగరం)
- Reconstruction: `"".join(aksharas) == word` for all test words
- Count verification: exact akshara counts for 7 key words

**Prompt Utils (`test_prompt_utils_telugu.py`)**
- Token counting across Unicode ranges: Telugu (U+0C00–U+0C7F), Kannada (U+0C80–U+0CFF), Devanagari (U+0900–U+097F)
- Period/question mark enforcement (no danda)
- QA pair combining to reach 512-token minimum

**Vocabulary (`test_telugu_vocabulary.py`)**
- All 958 words contain Telugu characters, zero Kannada/Devanagari leakage
- Category minimum thresholds: EASY_ANIMALS ≥ 30, EASY_OBJECTS ≥ 50, MEDIUM_FOOD ≥ 50, etc.
- Numbers: exactly 100, first = ఒకటి, last = వంద, tens verified (పది, ఇరవై, ముప్పై, నలభై, యాభై)
- Vargas: 7 groups, ka-varga = [క, ఖ, గ, ఘ, ఙ], ta-varga = [త, థ, ద, ధ, న]
- Rhyming pairs share last akshara (≤5 mismatches tolerance)

**Generators (`test_telugu_generators.py`)**
- All 11 `group1_sN.txt` files exist with exact line counts
- Total = 208,000 pairs verified
- Format: every line (first 100 sampled) contains `?` and ends with `.`
- Zero Kannada (U+0C80–U+0CFF) and Hindi (U+0900–U+097F) characters in all files
- Final output: ≥10,000 data points, all lines ≥512 tokens
- S11-specific: gunintam chart entries present, ottulu entries present, vowel/consonant classification present, క gunintam correctness (కా, కి, కు verified)

---

## 13. Verdict

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Total Q&A pairs = 208,000 | 208,000 | 208,000 | PASS |
| All 11 generators hit targets | All match | All match | PASS |
| Akshara segmentation correct | 8/8 pass | 8/8 pass | PASS |
| Unique vocabulary >= 950 | >= 950 | 958 | PASS |
| Rhyming pairs >= 100 | >= 100 | 834 | PASS |
| Number words = 100 | 100 | 100 | PASS |
| Min tokens >= 512 | >= 512 | 512 | PASS |
| Lines < 512 tokens = 0 | 0 | 0 | PASS |
| Kannada leakage = 0 | 0 | 0 | PASS |
| Hindi leakage = 0 | 0 | 0 | PASS |
| S11 gunintalu systematic | 37 consonants × 16 vowels | Covered | PASS |
| S11 ottulu from vocabulary | Words with conjuncts | Covered | PASS |

**Result: ALL CHECKS PASSED**
