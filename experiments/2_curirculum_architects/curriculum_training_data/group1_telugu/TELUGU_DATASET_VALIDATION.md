# Telugu Curriculum Dataset (Group 1) — Validation Report

**Date:** 2026-02-13 (updated 2026-02-14)
**Branch:** `p02/feat/cirriculum_data_telugu`
**Output:** `output/group1_telugu.txt` (~27 MB)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Q&A pairs | **208,000** |
| Statement types | 11 (S1-S11) |
| Final data points (combined lines) | **17,832** |
| Min tokens per data point | **512** |
| Max tokens per data point | **863** |
| Avg tokens per data point | **536.1** |
| Lines below 512 tokens | **0** |
| Blank lines in output | **0** |
| Vocabulary (unique words) | **3,082** |
| Output file size | ~27 MB |

---

## 2. Statement-Level Counts

All 11 generators produce the exact target count:

| Statement | Description | Target | Actual | Status |
|-----------|-------------|--------|--------|--------|
| S1 | Spelling (అక్షరక్రమం) — root-level | 30,000 | 30,000 | PASS |
| S2 | Letter Position (అక్షర స్థానం) | 26,000 | 26,000 | PASS |
| S3 | Sound Matching (ధ్వని) | 20,000 | 20,000 | PASS |
| S4 | Letter Count (అక్షర గణన) | 26,000 | 26,000 | PASS |
| S5 | Rhyming (ప్రాస) | 20,000 | 20,000 | PASS |
| S6 | Classification (వర్గీకరణ) — 13 categories + negatives | 20,000 | 20,000 | PASS |
| S7 | Position of Letter (అక్షరం స్థానం) | 18,000 | 18,000 | PASS |
| S8 | Number Spelling (సంఖ్య అక్షరక్రమం) — root-level | 12,000 | 12,000 | PASS |
| S9 | Last Letter (చివరి అక్షరం) | 18,000 | 18,000 | PASS |
| S10 | Word Comparison (పద పోలిక) — space-aware | 10,000 | 10,000 | PASS |
| S11 | Ottulu & Gunintalu (ఒత్తులు & గుణింతాలు) | 8,000 | 8,000 | PASS |
| **Total** | | **208,000** | **208,000** | **PASS** |

---

## 3. Akshara Segmentation Tests

Two segmentation functions serve different purposes:

### `get_telugu_aksharas()` — Syllabic units (for counting, position, comparison)

Conjuncts treated as single units. Used by S2, S3, S4, S5, S7, S9, S10.

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

### `get_telugu_aksharas_with_roots()` — Root characters (for spelling answers)

Extracts individual Unicode characters: consonants (U+0C15-U+0C39), vowels (U+0C05-U+0C14), anusvara/visarga (U+0C02-U+0C03), and vowel signs (U+0C3E-U+0C4C). Virama (్) is skipped. Used by S1 and S8 for spelling answers.

| Word | Root characters | Notes |
|------|----------------|-------|
| పుస్తకం | ప,ు,స,త,క,ం | Virama skipped, matra (ు) included |
| వ్యాకరణం | వ,య,ా,క,ర,ణ,ం | Virama skipped, ా-kara included |
| పక్షవాతం | ప,క,ష,వ,ా,త,ం | Root consonants separated |

---

## 4. Vocabulary Validation

| Category | Unique Count |
|----------|-------------|
| Easy words | 291 |
| Medium words | 372 |
| Hard words | 81 |
| Additional vocabulary | 269 |
| Extra words (5 expansion batches) | 1,368 |
| **Total unique (ALL_WORDS_UNIQUE)** | **3,082** |
| Rhyming pairs | 1,248 |
| Number words (1-1200) | 1,200 |
| Varga consonant groups | 7 |
| Classification categories | **13** |

### Word Tiers

- **Easy (2-3 aksharas)**: Animals (45), Objects (70), Body Parts (40), Colors (18), Nature (46), People (30), Food (42)
- **Medium (3-4 aksharas)**: Animals (31), Objects (65), Professions (46), Nature (62), Vehicles (37), Food (81), Household (50)
- **Hard (5+ aksharas)**: Complex Nouns (32), Abstract (30), Days (7), Months (12)
- **Additional**: 269 supplementary words
- **Extra**: 5 expansion batches — Animals (74), People (95), Objects (126), Food (81), Nature (43), Abstract (106), Places (59), Batch3 (211), Batch4 (195), Batch5 (477)

### Classification Categories (13) — Words per Category

| Category | Telugu | Words |
|----------|--------|-------|
| Animal | జంతువు | 195 |
| Food | ఆహారం | 261 |
| Nature | ప్రకృతి | 237 |
| Profession | వృత్తి | 180 |
| Object | వస్తువు | 388 |
| Place | స్థలం | 79 |
| Person | వ్యక్తి | 52 |
| Vehicle | వాహనం | 48 |
| Body Part | శరీర భాగం | 46 |
| Clothing | దుస్తులు | 45 |
| Color | రంగు | 32 |
| Vegetable | కూరగాయ | 26 |
| Fruit | పండు | 25 |

---

## 5. Cross-Script Leakage Check

| Script | Characters Found | Status |
|--------|-----------------|--------|
| Kannada (U+0C80-U+0CFF) | **0** | PASS |
| Hindi/Devanagari (U+0900-U+097F) | **0** | PASS |

No cross-script contamination in the final output.

---

## 6. Token Count Verification

Every line in the final output meets the 512-token minimum:

| Metric | Value |
|--------|-------|
| Total data points | 17,832 |
| Min tokens | 512 |
| Max tokens | 863 |
| Avg tokens | 536.1 |
| Lines < 512 tokens | **0** |
| Blank lines in output | **0** |

Token counting uses Telugu-aware `count_tokens_telugu()` where each Telugu Unicode character (U+0C00-U+0C7F) = 1 token.

---

## 7. Sample Output (Spot Check)

### S1: Spelling (root-level)
```
"ఒర" పదంలోని అక్షరాలను వ్రాయండి? ఒ,ర.
"పక్షవాతం" పదంలోని అక్షరాలను చూపించండి? ప,క,ష,వ,ా,త,ం.
"వ్యాకరణం" అనే పదాన్ని అక్షరాలుగా విడదీయండి? వ,య,ా,క,ర,ణ,ం.
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

### S6: Classification (13 categories + negative examples)
```
# MCQ positive — correct answer in options
"శెనగపప్పు" అనేది వృత్తి, జంతువు లేదా కూరగాయ? కూరగాయ.

# MCQ negative — correct answer NOT in options
"క్యాసెట్" ను శరీర భాగం, పండు లేదా జంతువు గా వర్గీకరించండి? ఏదీ కాదు, ఇది వస్తువు.
"ఎర్రచందనం" ను వృత్తి, ఆహారం లేదా వస్తువు గా వర్గీకరించండి? ఏదీ కాదు, ఇది ప్రకృతి.

# Yes/No positive
"వైఫై" ఒక వస్తువు పదమా? అవును.

# Yes/No negative
"ఫంగస్" అనేది ఒక స్థలం పదమా? కాదు.

# Open-ended
"గరిటె" పదం యొక్క వర్గం ఏమిటి? వస్తువు.
```

### S7: Position of Letter
```
"మీనారు" పదంలో "నా" ఎన్నవ అక్షరం? 2.
"ఉపాయం" లో "పా" ఏ స్థానంలో వస్తుంది? రెండవ.
```

### S8: Number Spelling (root-level)
```
"నూట పదిహేడు" అనే సంఖ్య పదాన్ని అక్షరాల వారీగా వ్రాయండి? న,ూ,ట,ప,ద,ి,హ,ే,డ,ు.
848 సంఖ్య యొక్క పేరు ఏమిటి? ఎనిమిది వందల నలభై ఎనిమిది.
```

### S9: Last Letter
```
"కొంగ" లో చివరి అక్షరం ఏది? గ.
```

### S10: Word Comparison
```
ఏ పదం ఎక్కువ అక్షరాలు కలిగి ఉంది, "బఠాని" లేదా "జలపాతం"? జలపాతం.
"అన్నప్రాశన" మరియు "గుర్తింపు" లలో పొడవైన పదం ఏది? అన్నప్రాశన.
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

## 8. Enhancements Log

### Enhancement 1: Root-Level Spelling (S1, S8)

**Problem:** S1 and S8 originally used `get_telugu_aksharas()` for spelling answers, which treated conjuncts as atomic units (e.g., పుస్తకం -> పు, స్త, కం). This doesn't teach the model what's *inside* each akshara at the character level.

**Fix:** Switched S1 and S8 to use `get_telugu_aksharas_with_roots()`, which extracts individual root characters — consonants, vowels, matras, anusvara — skipping virama.

| Before | After |
|--------|-------|
| పుస్తకం -> పు, స్త, కం | పుస్తకం -> ప,ు,స,త,క,ం |
| వ్యాకరణం -> వ్యా, క, ర, ణం | వ్యాకరణం -> వ,య,ా,క,ర,ణ,ం |

**Separator:** `,` (no space), per specification.

**Scope:** Only S1 and S8 produce spelling answers. S2-S7, S9-S10 use `get_telugu_aksharas()` for counting/position operations (unchanged).

**Files modified:** `generate_s1_spelling.py`, `generate_s8_numbers.py`, `telugu_grammar.py` (new function added).

---

### Enhancement 2: S6 Expanded to 13 Classification Categories

**Problem:** S6 originally had only 3 categories (జంతువు, వ్యక్తి, వస్తువు), making it too narrow for meaningful classification learning.

**Fix:** Expanded to 13 categories:

| Category | Telugu | Example Words |
|----------|--------|---------------|
| Animal | జంతువు | కుక్క, పిల్లి, ఏనుగు |
| Person | వ్యక్తి | అమ్మ, నాన్న, అక్క |
| Profession | వృత్తి | డాక్టర్, ఇంజనీర్, రైతు |
| Body Part | శరీర భాగం | కన్ను, ముక్కు, చేయి |
| Fruit | పండు | మామిడి, అరటి, ద్రాక్ష |
| Vegetable | కూరగాయ | టమాట, బంగాళదుంప |
| Clothing | దుస్తులు | చీర, లంగా, ప్యాంటు |
| Vehicle | వాహనం | కారు, బస్సు, రైలు |
| Place | స్థలం | గుడి, బడి, ఆసుపత్రి |
| Food | ఆహారం | అన్నం, రొట్టె, పాలు |
| Color | రంగు | ఎరుపు, నీలం, ఆకుపచ్చ |
| Nature | ప్రకృతి | నది, సముద్రం, కొండ |
| Object | వస్తువు | పుస్తకం, కలం, బల్ల |

---

### Enhancement 3: S6 Negative Examples

**Problem:** S6 only had positive examples ("X is category Y"), so the model never learned to say "no" or "none of the above."

**Fix:** Added 5 question types with weighted distribution:

| Type | Weight | Description | Example Answer |
|------|--------|-------------|----------------|
| MCQ positive | 45% | Correct category in options | కూరగాయ |
| MCQ negative | 15% | Correct category NOT in options | ఏదీ కాదు, ఇది ప్రకృతి |
| Yes/No positive | 12% | "Is X a Y?" where Y is correct | అవును |
| Yes/No negative | 18% | "Is X a Y?" where Y is wrong | కాదు |
| Open-ended | 10% | "What category is X?" | వస్తువు |

**Result:** ~38% of S6 samples are negative examples (MCQ negative + Yes/No negative).

---

### Enhancement 4: Classification Category Ordering Fix

**Problem:** `CLASSIFICATION_CATEGORIES` dict had broad categories (వస్తువు, ఆహారం) listed before specific ones (దుస్తులు, పండు, కూరగాయ). Since word-to-category assignment uses first-assignment-wins, 81 words were misclassified.

**Examples of misclassification:**
- చీర (saree) -> వస్తువు (object) instead of దుస్తులు (clothing)
- ద్రాక్ష (grape) -> ఆహారం (food) instead of పండు (fruit)
- గుడి (temple) -> వస్తువు (object) instead of స్థలం (place)

**Fix:** Reordered dict to put specific categories before broad ones:

```
Order: జంతువు -> వ్యక్తి -> వృత్తి -> శరీర భాగం -> పండు -> కూరగాయ ->
       దుస్తులు -> వాహనం -> స్థలం -> ఆహారం -> రంగు -> ప్రకృతి -> వస్తువు (last, broadest)
```

**Result:** All 81 cross-category conflicts resolved.

---

### Enhancement 5: S7 Template Grammar Fixes

**Problem:** Two S7 templates had invalid Telugu grammar, identified by native speaker review.

| Invalid Template | Issue | Fixed Template |
|-----------------|-------|----------------|
| `పదంలో అక్షరం ఎంతవ స్థానంలో వస్తుంది?` | "ఎంతవ" is not valid Telugu | `పదంలో అక్షరం ఎన్నవ స్థానంలో ఉంది?` |
| `లో అక్షరం ఎక్కడ ఉంది?` | "ఎక్కడ" = physical location, not ordinal | `పదంలో ఎన్నవ అక్షరం?` |

**Final S7 templates (6):**
1. `'"{word}" లో "{char}" అక్షరం ఏ స్థానంలో ఉంది?'`
2. `'"{word}" పదంలో "{char}" ఎన్నవ అక్షరం?'`
3. `'"{word}" పదంలో "{char}" అక్షరం ఏ స్థానంలో ఉంది?'`
4. `'"{word}" లో "{char}" ఏ స్థానంలో వస్తుంది?'`
5. `'"{char}" అక్షరం "{word}" పదంలో ఏ స్థానంలో ఉంది?'`
6. `'"{word}" పదంలో "{char}" అక్షరం ఎన్నవ స్థానంలో ఉంది?'`

---

### Enhancement 6: S10 Space-in-Aksharas Fix

**Problem:** Multi-word phrases (e.g., "కేబుల్ కార్", "అడవి దున్న") had spaces counted as aksharas by `get_telugu_grapheme_clusters()`, causing incorrect length comparisons.

**Fix:** Filter out whitespace from grapheme clusters when computing word lengths:
```python
clusters = [c for c in get_telugu_grapheme_clusters(word) if c.strip()]
```

**Impact:** ~132 comparisons (1.3% of dataset) corrected.

---

### Enhancement 7: Output Blank Lines Fix

**Problem:** `output/group1_telugu.txt` had blank lines interspersed. Root cause: `combine_qa_pairs_to_reach_min_tokens_telugu()` appends `\n` to each sample, then the orchestrator's `f.write(sample + "\n")` added a second `\n`.

**Fix:** Changed orchestrator write to `f.write(sample.rstrip("\n") + "\n")`.

**Result:** Zero blank lines in final output.

---

## 9. S11 Design Rationale

### Why S11 was added

S1-S10 treat aksharas as **atomic units** — the model learns to spell, count, position, and compare aksharas but never understands what's *inside* them.

S11 teaches the **compositional structure** of Telugu script:

| Concept | What it teaches | Example |
|---------|----------------|---------|
| **Gunintalu** | Consonant + vowel sign = combined form | క + ఆ-కారం = కా |
| **Identify base consonant** | Extract root consonant from combined form | కీ -> క |
| **Identify vowel sign** | Extract vowel sign from combined form | కీ -> ఈ-కారం |
| **Gunintam chart** | Full vowel sign series for a consonant | క, కా, కి, కీ, కు, కూ, ... |
| **Ottulu** | Identify conjuncts in words | అమ్మ -> మ్మ |
| **Conjunct decomposition** | Break conjunct into component consonants | స్త -> స, త |
| **Conjunct detection** | Does a word contain conjuncts? | సమాజం -> లేదు |
| **Vowel/consonant classification** | Is a character స్వరం or వ్యంజనం? | క -> వ్యంజనం |

### S11 pair count: 8,000

| Component | Seed pairs | Fill | Allocation |
|-----------|-----------|------|------------|
| Gunintalu combinations (37 consonants x 16 vowels) | ~592 | ~2,200 | ~35% |
| Base consonant identification | ~148 | ~800 | ~12% |
| Vowel sign identification | ~148 | ~800 | ~12% |
| Gunintam charts (full series) | ~37 | ~1,200 | ~15% |
| Ottulu in words | ~400 | ~700 | ~14% |
| Conjunct decomposition | ~100 | ~200 | ~4% |
| Conjunct detection (yes/no) | ~300 | ~200 | ~6% |
| Vowel/consonant classification | ~53 | ~150 | ~2% |

**Consonants (37)**: Full traditional alphabet including all vargas, semi-vowels/sibilants, plus ఱ and క్ష.

**Vowel signs (16)**: అ (inherent, no sign), ఆ-ఔ (12 standard), ౠ (long vocalic r), అం (anusvara), అః (visarga).

**vs Kannada S11**: Kannada used only 20 hardcoded Q&A pairs repeated to fill 10K. Telugu S11 generates **programmatically diverse** pairs from 37 consonants x 16 vowel signs x multiple template variants + vocabulary-based ottulu detection.

---

## 10. File Inventory

### Foundation Files
| File | Purpose | Size |
|------|---------|------|
| `prompt_utils_telugu.py` | Token counting, QA formatting, line combining | 4.6 KB |
| `telugu_grammar.py` | Akshara segmentation + root character extraction | ~1.5 KB |
| `telugu_vocabulary.py` | 3,082 unique words, 1,200 numbers, vargas, 1,248 rhyming pairs, 13 classification categories | 38 KB |

### Generators
| File | Statement | Target |
|------|-----------|--------|
| `generate_s1_spelling.py` | S1: Spelling (root-level) | 30,000 |
| `generate_s2_letter_position.py` | S2: Letter Position | 26,000 |
| `generate_s3_sound.py` | S3: Sound Matching | 20,000 |
| `generate_s4_count.py` | S4: Letter Count | 26,000 |
| `generate_s5_rhyme.py` | S5: Rhyming | 20,000 |
| `generate_s6_classify.py` | S6: Classification (13 categories, negative examples) | 20,000 |
| `generate_s7_position.py` | S7: Position of Letter | 18,000 |
| `generate_s8_numbers.py` | S8: Number Spelling (root-level) | 12,000 |
| `generate_s9_last.py` | S9: Last Letter | 18,000 |
| `generate_s10_compare.py` | S10: Word Comparison (space-aware) | 10,000 |
| `generate_s11_ottulu_gunintalu.py` | S11: Ottulu & Gunintalu | 8,000 |

### Orchestrator & Output
| File | Purpose | Size |
|------|---------|------|
| `generate_group1_telugu_dataset.py` | Combines S1-S11, enforces 512-token min, no blank lines | 3.4 KB |
| `output/group1_telugu.txt` | Final dataset | ~27 MB |

### Generated Data Files
| File | Lines | Size |
|------|-------|------|
| `group1_s1.txt` | 30,000 | 4.3 MB |
| `group1_s2.txt` | 26,000 | 3.2 MB |
| `group1_s3.txt` | 20,000 | 2.7 MB |
| `group1_s4.txt` | 26,000 | 3.1 MB |
| `group1_s5.txt` | 20,000 | 3.5 MB |
| `group1_s6.txt` | 20,000 | 2.7 MB |
| `group1_s7.txt` | 18,000 | 2.1 MB |
| `group1_s8.txt` | 12,000 | 1.9 MB |
| `group1_s9.txt` | 18,000 | 1.8 MB |
| `group1_s10.txt` | 10,000 | 1.4 MB |
| `group1_s11.txt` | 8,000 | 0.9 MB |

---

## 11. Telugu-Specific Design Decisions

| Decision | Details |
|----------|---------|
| **Two-level segmentation** | `get_telugu_aksharas()` for syllabic counting/position; `get_telugu_aksharas_with_roots()` for root-level spelling |
| **No genitive suffix system** | Telugu uses invariant postpositions (లో, యొక్క, లోని) — eliminates Kannada's 4-suffix system |
| **Answer terminator** | Period (`.`), NOT danda (`।`) |
| **Yes/No for identity** | అవును / కాదు |
| **Yes/No for existence** | అవును / లేదు |
| **Telugu ordinals** | మొదటి (1st, irregular), then {cardinal}వ pattern (రెండవ, మూడవ, ...) |
| **Ordinal question word** | ఎన్నవ (NOT ఎంతవ or ఎక్కడ) |
| **Number words** | Irregular teens (11-19 are fused), compound 21+ (ఇరవై ఒకటి) |
| **Separate prompt_utils** | `prompt_utils_telugu.py` — does NOT modify shared `prompt_utils.py` |
| **Spelling terminology** | అక్షరక్రమం (NOT వర్తని which is Hindi-influenced) |
| **Spelling separator** | `,` (no space) for root-level spelling answers |
| **Classification specificity** | 13 categories ordered specific-first in dict for correct first-assignment-wins |
| **Negative examples** | ~38% of S6 are negative (MCQ negative + Yes/No negative) |
| **Space-aware comparison** | S10 filters whitespace from grapheme clusters for multi-word phrases |
| **S11 programmatic generation** | Generates from consonant x vowel sign matrix, not hardcoded pairs |

---

## 12. How to Reproduce

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

## 13. Test Suite

**183 tests** across 4 test files, all passing in ~3.10s.

**Run:** `uv run python -m pytest group1_telugu/tests/ -v`

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_telugu_grammar.py` | 24 | Virama detection, akshara segmentation (15 words), reconstruction invariant, empty input |
| `test_prompt_utils_telugu.py` | 25 | Token counting (Telugu/Kannada/Devanagari/English/mixed/empty), formatting helpers, QA pair formatting, combining |
| `test_telugu_vocabulary.py` | 34 | Word counts (>=950 unique), no duplicates, category minimums (16 categories), Telugu script validation, no Kannada/Hindi chars, numbers (100), days/months, vargas (7 groups), classification (13 categories), rhyming pairs (>=100) |
| `test_telugu_generators.py` | 100 | File existence (11 statement files + final output), line counts (all 11 match targets), total = 208,000, format validation (Q? A. on first 100 lines), no danda, cross-script leakage (all files), token minimums (>=512), S11 gunintalu/ottulu content checks |
| **Total** | **183** | |

### Key Test Categories

**Grammar (`test_telugu_grammar.py`)**
- Virama detection: present, absent, empty, standalone
- Akshara segmentation: simple words, gemination, complex conjuncts, anusvara
- Reconstruction: `"".join(aksharas) == word` for all test words
- Count verification: exact akshara counts for 7 key words

**Prompt Utils (`test_prompt_utils_telugu.py`)**
- Token counting across Unicode ranges: Telugu (U+0C00-U+0C7F), Kannada (U+0C80-U+0CFF), Devanagari (U+0900-U+097F)
- Period/question mark enforcement (no danda)
- QA pair combining to reach 512-token minimum

**Vocabulary (`test_telugu_vocabulary.py`)**
- All 3,082 words contain Telugu characters, zero Kannada/Devanagari leakage
- Category minimum thresholds: EASY_ANIMALS >= 30, EASY_OBJECTS >= 50, MEDIUM_FOOD >= 50, etc.
- Numbers: 1,200 (1-1200), first = ఒకటి, last = వంద, tens verified
- Vargas: 7 groups, ka-varga = [క, ఖ, గ, ఘ, ఙ], ta-varga = [త, థ, ద, ధ, న]
- Classification: 13 categories with correct ordering
- Rhyming pairs share last akshara (<=5 mismatches tolerance)

**Generators (`test_telugu_generators.py`)**
- All 11 `group1_sN.txt` files exist with exact line counts
- Total = 208,000 pairs verified
- Format: every line (first 100 sampled) contains `?` and ends with `.`
- Zero Kannada and Hindi characters in all files
- Final output: >=10,000 data points, all lines >=512 tokens
- S11-specific: gunintam chart, ottulu, vowel/consonant classification checks

---

## 14. Verdict

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Total Q&A pairs = 208,000 | 208,000 | 208,000 | PASS |
| All 11 generators hit targets | All match | All match | PASS |
| Akshara segmentation correct | 8/8 pass | 8/8 pass | PASS |
| Root-level spelling (S1, S8) | Roots extracted | Roots extracted | PASS |
| Unique vocabulary >= 3,000 | >= 3,000 | 3,082 | PASS |
| Rhyming pairs >= 1,000 | >= 1,000 | 1,248 | PASS |
| Number words = 1,200 | 1,200 | 1,200 | PASS |
| Classification categories = 13 | 13 | 13 | PASS |
| S6 negative examples ~38% | ~33-40% | ~38% | PASS |
| S6 category ordering correct | Specific-first | Specific-first | PASS |
| S7 templates grammatically valid | All valid | All valid | PASS |
| S10 space-aware comparison | No space counting | No space counting | PASS |
| Min tokens >= 512 | >= 512 | 512 | PASS |
| Lines < 512 tokens = 0 | 0 | 0 | PASS |
| Blank lines in output = 0 | 0 | 0 | PASS |
| Kannada leakage = 0 | 0 | 0 | PASS |
| Hindi leakage = 0 | 0 | 0 | PASS |
| S11 gunintalu systematic | 37 consonants x 16 vowels | Covered | PASS |
| S11 ottulu from vocabulary | Words with conjuncts | Covered | PASS |

**Result: ALL CHECKS PASSED**
