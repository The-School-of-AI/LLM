# Telugu Curriculum Dataset — Technical Approach

## Overview

A curriculum dataset of **200,000 Telugu question-answer pairs** for language and literacy training across 10 statement types (S1–S10). Each final data point contains at least **512 tokens**, achieved by combining multiple Q&A pairs per line.

**Output**: `curriculum_training_data/output/group1_telugu.txt`

---

## Format Specifications

- **Q&A format**: `Q? A.` — query ends with `?`, answer ends with period `.`
- **Combining**: Multiple pairs per line joined with spaces; each pair is `Q? A.`
- **Minimum tokens**: 512 per line (Telugu U+0C00–U+0C7F counted as 1 token each via `prompt_utils.count_tokens`)
- **Encoding**: UTF-8
- **Punctuation**: Telugu uses `.` (period), NOT `।` (danda)

**Example output line:**
```
"పుస్తకం" పదం యొక్క అక్షరక్రమం ఏమిటి? పు, స్త, కం. "నీరు" లో ఎన్ని అక్షరాలు ఉన్నాయి? 2. "కుక్క" వ్యక్తి, జంతువు లేదా వస్తువు? జంతువు.
```

**Utilities** (`prompt_utils.py`):
- `format_qa_pair_kannada(query, answer)` — reused; formats one pair as `Q? A.`
- `combine_qa_pairs_to_reach_min_tokens_kannada(qa_pairs, min_tokens=512)` — reused; combines pairs into lines of ≥512 tokens
- `count_tokens()` — **MUST be updated** to add Telugu range `U+0C00`–`U+0C7F`

---

## Akshara-Level Segmentation

Telugu uses syllabic units (aksharas / అక్షరాలు), not raw Unicode graphemes. `telugu_grammar.get_telugu_aksharas(word)` segments words per Telugu linguistics:

- **Conjuncts (సంయుక్తాక్షరాలు)** like స్త, ద్య, క్ష = 1 akshara
- **Geminates** like మ్మ, క్క, న్న = 1 akshara
- **Anusvara (ం)** = part of preceding akshara
- **Visarga (ః)** = part of preceding akshara

**Algorithm**: `regex.findall(r"\X", word)` for grapheme clusters + virama (్ U+0C4D) merging — identical logic to Kannada with Telugu Unicode constants.

### Verified Akshara Counts

| Word | Meaning | Aksharas | Count |
|------|---------|----------|-------|
| పుస్తకం | book | పు, స్త, కం | 3 |
| విద్యార్థి | student | వి, ద్యా, ర్థి | 3 |
| నీరు | water | నీ, రు | 2 |
| అమ్మ | mother | అ, మ్మ | 2 |
| జ్ఞానం | knowledge | జ్ఞా, నం | 2 |
| క్షమ | forgiveness | క్ష, మ | 2 |
| ఆస్పత్రి | hospital | ఆ, స్ప, త్రి | 3 |
| కుక్క | dog | కు, క్క | 2 |
| బడి | school | బ, డి | 2 |
| విద్యాలయం | school (formal) | వి, ద్యా, ల, యం | 4 |

All counting, position, spelling, and last-letter logic use aksharas (S1–S10).

---

## Vocabulary

**`telugu_vocabulary.py`**:

### Target: 950+ unique Telugu words

| Category | Easy | Medium | Hard |
|----------|------|--------|------|
| Animals (జంతువులు) | ~45 | ~30 | — |
| Objects (వస్తువులు) | ~70 | ~65 | ~30 |
| Body Parts (శరీర భాగాలు) | ~40 | — | — |
| Colors (రంగులు) | ~20 | — | — |
| Nature (ప్రకృతి) | ~45 | ~60 | — |
| People (వ్యక్తులు) | ~30 | — | — |
| Food (ఆహారం) | ~40 | ~80 | — |
| Professions (వృత్తులు) | — | ~45 | — |
| Vehicles (వాహనాలు) | — | ~35 | — |
| Household (గృహ వస్తువులు) | — | ~48 | — |
| Complex Nouns | — | — | ~30 |
| Abstract Concepts | — | — | ~30 |
| Days & Months | 19 | — | — |
| Numbers (1–100) | 100 | — | — |

### Difficulty Classification
- **Easy**: 2–3 akshara words (e.g., నీరు, అమ్మ, కుక్క)
- **Medium**: 3–4 akshara words (e.g., పుస్తకం, ఆస్పత్రి)
- **Hard**: 5+ akshara words or complex conjuncts (e.g., విద్యార్థి, ప్రధానమంత్రి)

### Rhyming Pairs
Built programmatically via `build_real_rhyming_pairs()`:
- Group all words by their **last akshara** using `get_telugu_aksharas()[-1]`
- Within each group, create cyclic pairs for variety
- Only real vocabulary words used (both members of each pair exist in the word list)
- Target: **100+ pairs** from 950+ vocabulary

### Classification Categories
```python
CLASSIFICATION_CATEGORIES = {
    "జంతువు": [...],   # Animals
    "వ్యక్తి": [...],    # People + Professions
    "వస్తువు": [...],   # Objects + Household + Vehicles
}
```

### Number Words (1–100)

**1–10 (basic):**
ఒకటి, రెండు, మూడు, నాలుగు, అయిదు, ఆరు, ఏడు, ఎనిమిది, తొమ్మిది, పది

**11–19 (irregular, fused forms):**
పదకొండు, పన్నెండు, పదమూడు, పధ్నాలుగు, పదునయిదు, పదహారు, పదిహేడు, పధ్ధెనిమిది, పందొమ్మిది

**Tens:**
ఇరవై (20), ముప్పై (30), నలభై (40), యాభై (50), అరవై (60), డెబ్బై (70), ఎనభై (80), తొంభై (90), వంద (100)

**Compounds (21+)**: Two separate words — ఇరవై ఒకటి (21), ముప్పై అయిదు (35), తొంభై తొమ్మిది (99)

### Consonant Vargas
```python
VARGAS = {
    "క": ["క", "ఖ", "గ", "ఘ", "ఙ"],     # Velar
    "చ": ["చ", "ఛ", "జ", "ఝ", "ఞ"],     # Palatal
    "ట": ["ట", "ఠ", "డ", "ఢ", "ణ"],     # Retroflex
    "త": ["త", "థ", "ద", "ధ", "న"],     # Dental
    "ప": ["ప", "ఫ", "బ", "భ", "మ"],     # Labial
    "య": ["య", "ర", "ల", "వ"],           # Semi-vowels
    "శ": ["శ", "ష", "స", "హ", "ళ"],      # Sibilants
}
```

---

## Grammar Helpers (`telugu_grammar.py`)

### Core Function: `get_telugu_aksharas(word)`

```python
TELUGU_VIRAMA = "\u0C4D"  # ్

def get_telugu_aksharas(word: str) -> list[str]:
    clusters = regex.findall(r"\X", word)
    aksharas = []
    i = 0
    while i < len(clusters):
        akshara = clusters[i]
        while i + 1 < len(clusters) and akshara[-1] == TELUGU_VIRAMA:
            i += 1
            akshara += clusters[i]
        aksharas.append(akshara)
        i += 1
    return aksharas
```

### Postpositions (SIMPLER than Kannada)

Telugu uses **invariant postpositions** — they never change based on the preceding word:

| Postposition | Meaning | Usage |
|--------------|---------|-------|
| **లో** | in/within | `"నీరు" లో` — always "లో", never changes |
| **లోని** | that which is in | `"నీరు" లోని చివరి అక్షరం` |
| **యొక్క** | of (possessive) | `"నీరు" యొక్క చివరి అక్షరం` |

This eliminates the genitive suffix selection logic entirely. No equivalent of Kannada's `get_genitive_suffix()` is needed.

### Unicode Constants

```python
TELUGU_VIRAMA = "\u0C4D"           # ్
TELUGU_ANUSVARA = "\u0C02"         # ం
TELUGU_VISARGA = "\u0C03"          # ః
TELUGU_CONSONANT_FIRST = "\u0C15"  # క
TELUGU_CONSONANT_LAST = "\u0C39"   # హ
TELUGU_VOWEL_FIRST = "\u0C05"      # అ
TELUGU_VOWEL_LAST = "\u0C14"       # ఔ
TELUGU_BLOCK_START = "\u0C00"
TELUGU_BLOCK_END = "\u0C7F"

VOWELS = set(chr(c) for c in range(0x0C05, 0x0C15))  # అ through ఔ
CONSONANTS = set(chr(c) for c in range(0x0C15, 0x0C3A))  # క through హ
```

### Ordinal Position Names

```python
POSITIONS = [
    ("మొదటి", "1"),      # 1st (irregular)
    ("రెండవ", "2"),      # 2nd
    ("మూడవ", "3"),       # 3rd
    ("నాల్గవ", "4"),     # 4th
    ("ఐదవ", "5"),        # 5th
    ("ఆరవ", "6"),        # 6th
    ("ఏడవ", "7"),        # 7th
    ("ఎనిమిదవ", "8"),    # 8th
    ("తొమ్మిదవ", "9"),   # 9th
    ("పదవ", "10"),       # 10th
]
# For >10: "{number}వ" (e.g., పదకొండవ)
```

### Yes/No Answers

- **అవును** = yes (general affirmative)
- **కాదు** = no (negation of identity — "it is not")
- **లేదు** = no (negation of existence — "there isn't")

Usage:
- Identity questions (Is this X? Do these rhyme?) → అవును / కాదు
- Existence questions (Is there a 5th akshara?) → అవును / లేదు

---

## Telugu Terminology

| Concept | Telugu Term | English Loan (if used) |
|---------|-----------|----------------------|
| Spelling | అక్షరక్రమం / వర్ణక్రమం | స్పెల్లింగ్ |
| Letter/Akshara | అక్షరం | — |
| Vowel | స్వరం | — |
| Consonant | వ్యంజనం | — |
| Rhyme | ప్రాస | — |
| Position | స్థానం | — |
| Count | లెక్క (colloquial) / గణన (formal) | — |
| Last | చివరి | — |
| Word | పదం | — |
| Sound | ధ్వని | — |
| Conjunct | సంయుక్తాక్షరం | — |
| Vowel sign | గుణింతం | — |
| Person | వ్యక్తి | — |
| Animal | జంతువు | — |
| Thing/Object | వస్తువు | — |

**Important**: Do NOT use "వర్తని" (Hindi/Sanskrit term). Use native Telugu "అక్షరక్రమం" or "వర్ణక్రమం" for spelling.

---

## Statement Types (S1–S10)

### Target Distribution

| Statement | Focus | Target Pairs | % |
|-----------|-------|-------------|---|
| S1 | Spelling (అక్షరక్రమం) | 30,000 | 15% |
| S2 | Letter Position (అక్షర స్థానం) | 26,000 | 13% |
| S3 | Sound Matching (ధ్వని) | 20,000 | 10% |
| S4 | Letter Count (అక్షర గణన) | 26,000 | 13% |
| S5 | Rhyming (ప్రాస) | 20,000 | 10% |
| S6 | Classification (వర్గీకరణ) | 20,000 | 10% |
| S7 | Position of Letter (అక్షరం స్థానం) | 18,000 | 9% |
| S8 | Number Spelling (సంఖ్య అక్షరక్రమం) | 12,000 | 6% |
| S9 | Last Letter (చివరి అక్షరం) | 18,000 | 9% |
| S10 | Word Comparison (పద పోలిక) | 10,000 | 5% |
| **Total** | | **200,000** | **100%** |

---

### S1: Spelling (అక్షరక్రమం) — 30,000 pairs

**Two sub-types:**

1. **Spelling** (hyphen-separated aksharas):
   - Q: `"పుస్తకం" పదం యొక్క అక్షరక్రమం ఏమిటి?`
   - A: `పు-స్త-కం`

2. **Listing** (comma-separated aksharas):
   - Q: `"పుస్తకం" పదంలోని అక్షరాలను జాబితా చేయండి?`
   - A: `పు, స్త, కం`

**Key**: Uses `get_telugu_aksharas()` — akshara-level, NOT Unicode character-level.

**Templates (~20):** 10 spelling + 10 listing variations in Telugu:
- `"X" పదం యొక్క అక్షరక్రమం ఏమిటి?`
- `"X" అనే పదాన్ని అక్షరాలుగా విడదీయండి?`
- `"X" పదం యొక్క స్పెల్లింగ్ చెప్పండి?`
- `"X" పదంలోని అక్షరాలు ఏమిటి?`
- `"X" పదాన్ని అక్షరాల వారీగా వ్రాయండి?`
- `"X" పదం యొక్క సరైన అక్షరక్రమం ఏది?`
- `"X" పదంలోని అక్షరాలను ప్రత్యేకంగా చెప్పండి?`
- `"X" పదంలోని అన్ని అక్షరాలను జాబితా చేయండి?`
- `"X" పదంలోని అక్షరాలను క్రమంలో చెప్పండి?`
- etc.

**Word expansion**: Easy×50, Medium×60, Hard×70

---

### S2: Letter Position (అక్షర స్థానం) — 26,000 pairs

**Question types:**
- First/last/middle/Nth akshara
- Is there a 5th akshara? (అవును / లేదు)
- Is first akshara vowel or consonant? (స్వరం / వ్యంజనం)

**Example Q&A:**
- Q: `"పుస్తకం" లో మొదటి అక్షరం ఏమిటి?` → A: `పు`
- Q: `"నీరు" లో రెండవ అక్షరం ఏమిటి?` → A: `రు`
- Q: `"బడి" లో ఐదవ అక్షరం ఉందా?` → A: `లేదు`

**Templates (~11):**
- `"X" లో {position} అక్షరం ఏమిటి?`
- `"X" పదంలో {position} అక్షరం చెప్పండి?`
- `"X" లో చివరి అక్షరం ఏమిటి?`
- `"X" లో మధ్య అక్షరం ఏమిటి?`
- `"X" లో {position} అక్షరం ఉందా?`
- `"X" లో మొదటి అక్షరం స్వరమా లేదా వ్యంజనమా?`
- etc.

---

### S3: Sound Matching (ధ్వని) — 20,000 pairs

**20 template types:**

1. **Rhyme word**: `"నీరు" పదానికి ప్రాసబద్ధమైన పదం ఏమిటి?` → `దూరు`
2. **Word starting**: `"క" అక్షరంతో మొదలయ్యే పదం చెప్పండి?` → `కుక్క`
3. **Do rhyme yes/no**: `"నీరు" మరియు "దూరు" పదాలు ప్రాసబద్ధమా?` → `అవును`
4. **Word with vowel**: `"అ" స్వరం ధ్వని ఉన్న పదం ఏమిటి?` → `అమ్మ`
5. **Word ending**: `"రు" అక్షరంతో అంతమయ్యే పదం చెప్పండి?` → `నీరు`
6. **Same pronunciation**: `"హ" మరియు "ప" అక్షరాల ఉచ్చారణ ఒకటేనా?` → `కాదు`
7. **Animal starting**: `"కు" తో మొదలయ్యే జంతువు పేరు ఏమిటి?` → `కుక్క`
8. **Identify sound**: `"త" వర్గం అక్షరాల ధ్వనిని గుర్తించండి?` → `త, థ, ద, ధ, న`
9. **First sound**: `"పుస్తకం" పదం యొక్క మొదటి ధ్వని ఏమిటి?` → `పు`
10. **Word with nasal**: `"న" అక్షరం యొక్క అనునాసిక ధ్వని ఉన్న పదం?` → `నీరు`
11. **Another rhyme**: `"నీరు" పదానికి ప్రాసమయ్యే మరొక పదం చెప్పండి?` → `దూరు`
12. **Do rhyme variant**: `"X" మరియు "Y" పదాలు ప్రాసబద్ధమైనవా?`
13. **Fruit starting**: `"మా" అక్షరంతో మొదలయ్యే పండు పేరు చెప్పండి?` → `మామిడి`
14. **Similar sound**: `"X" పదానికి సమానమైన ధ్వని పదం ఏమిటి?`
15. **Word ending variant**: `"X" ధ్వనితో అంతమయ్యే పదాన్ని చెప్పండి?`
16. **శ vs ష**: `"శ" మరియు "ష" ఉచ్చారణలో సమానత్వం ఉందా?` → `కాదు` (Telugu distinguishes them more clearly)
17. **Two words with sound**: `"క" అక్షరం ధ్వని ఉన్న రెండు పదాలను చెప్పండి?` → `కుక్క, కమలం`
18. **Verb starting**: `"త" అక్షరంతో మొదలయ్యే క్రియాపదం ఏమిటి?`
19. **First sound variant**: `"X" పదం యొక్క మొదటి శబ్దం ఏమిటి?`
20. **Similar sound variant**: `"X" పదం ధ్వనికి దగ్గరగా ఉన్న పదం చెప్పండి?`

**Telugu verb endings heuristic**: `["చు", "డు", "గు", "ను", "తు", "పు", "వు", "ళ్ళు"]`

---

### S4: Letter Count (అక్షర గణన) — 26,000 pairs

**Answer types:**
1. **Count**: `"పుస్తకం" లో ఎన్ని అక్షరాలు ఉన్నాయి?` → `3 అక్షరాలు`
2. **Two-letter yes/no**: `"నీరు" రెండు అక్షరాల పదమా?` → `అవును`
3. **Three-letter yes/no**: `"బడి" మూడు అక్షరాల పదమా?` → `కాదు`
4. **Vowel count**: `"పుస్తకం" లో ఎన్ని స్వరాలు ఉన్నాయి?` → `1 స్వరం` (only పు has vowel-starting cluster)
5. **Consonant count**: `"పుస్తకం" లో ఎన్ని వ్యంజనాలు ఉన్నాయి?` → `2 వ్యంజనాలు`

**Counting**: Uses `len(get_telugu_aksharas(word))` — akshara-level.

**Templates (~21):** varied phrasings in Telugu.

---

### S5: Rhyming (ప్రాస) — 20,000 pairs

**Format**: Multiple choice (MCQ)

**Example:**
```
"నీరు" పదానికి ప్రాస పదం ఏది, "దూరు" లేదా "పుస్తకం"?  → దూరు
```

**7 template variations:**
- `"X" పదానికి ప్రాస పదం ఏది, "Y" లేదా "Z"?`
- `"X" పదానికి ఏ పదం ప్రాసబద్ధమవుతుంది, "Y" లేదా "Z"?`
- `"X" తో ప్రాస పదం ఏమిటి, "Y" లేదా "Z"?`
- `"Y" మరియు "Z" లో "X" పదానికి ప్రాసమయ్యేది ఏది?`
- `"X" పదానికి ప్రాస అయ్యే పదం ఏది, "Y" లేదా "Z"?`
- `ఏ పదం "X" తో ప్రాస అవుతుంది, "Y" లేదా "Z"?`
- `"Y" మరియు "Z" లలో "X" కు ప్రాసబద్ధమైనది ఏది?`

**Bidirectional**: Generates both word→rhyme and rhyme→word pairs.
**Option position**: Randomized (rhyme word can be option1 or option2).

---

### S6: Classification (వర్గీకరణ) — 20,000 pairs

**3 categories:** వ్యక్తి (person), జంతువు (animal), వస్తువు (object)

**Example:**
```
"కుక్క" వ్యక్తి, జంతువు లేదా వస్తువు?  → జంతువు
```

**7 template variations:**
- `"X" వ్యక్తి, జంతువు లేదా వస్తువు?`
- `"X" ఏమిటి, వ్యక్తి, జంతువు లేదా వస్తువు?`
- `"X" పదం ఏ వర్గంలోకి వస్తుంది, వ్యక్తి, జంతువు లేదా వస్తువు?`
- `"X" ను వ్యక్తి, జంతువు లేదా వస్తువుగా వర్గీకరించండి?`
- `"X" అనేది వ్యక్తి, జంతువు లేదా వస్తువు?`
- `"X" ఏ రకం, వ్యక్తి, జంతువు లేదా వస్తువు?`
- `"X" పదం యొక్క వర్గం ఏమిటి?`

**Word expansion**: all_words × 20

---

### S7: Position of Letter (అక్షరం స్థానం) — 18,000 pairs

**Question**: Given a word and an akshara in it, what position is it at?

**Example:**
```
"నీరు" లో "రు" అక్షరం ఏ స్థానంలో ఉంది?  → రెండవ
```

**Key advantage over Kannada**: No genitive suffix logic needed. Telugu uses invariant `లో` (in) for all words.

**6 template variations:**
- `"X" లో "Y" అక్షరం ఏ స్థానంలో ఉంది?`
- `"X" లో "Y" అక్షరం ఎక్కడ ఉంది?`
- `"X" పదంలో "Y" అక్షరం ఏ స్థానంలో ఉంది?`
- `"X" లో "Y" ఏ స్థానంలో వస్తుంది?`
- `"Y" అక్షరం "X" పదంలో ఏ స్థానంలో ఉంది?`
- `"X" పదంలో "Y" ఎంతవ స్థానంలో వస్తుంది?`

**Answer format**: Randomly alternates between ordinal name (రెండవ) and number (2) — 50/50 split.

---

### S8: Number Spelling (సంఖ్య అక్షరక్రమం) — 12,000 pairs

**Two sub-types:**

1. **Number → Name**:
   - Q: `42 యొక్క పేరు ఏమిటి?` → A: `నలభై రెండు`
   - Q: `11 ను తెలుగులో ఏమంటారు?` → A: `పదకొండు`

2. **Name → Spelling** (akshara-level):
   - Q: `"పదకొండు" పదం యొక్క అక్షరక్రమం ఏమిటి?` → A: `ప, ద, కొ, ండు`

**Templates:**
- Number→Name: 5 templates
- Name→Spelling: 5 templates

**Number range**: 1–100 (stored in NUMBERS dictionary in vocabulary).

---

### S9: Last Letter (చివరి అక్షరం) — 18,000 pairs

**Example:**
```
"పుస్తకం" లోని చివరి అక్షరం ఏమిటి?  → కం
```

**7 template variations:**
- `"X" లోని చివరి అక్షరం ఏమిటి?`
- `"X" యొక్క చివరి అక్షరం ఏమిటి?`
- `"X" పదం ఏ అక్షరంతో అంతమవుతుంది?`
- `"X" పదం యొక్క ఆఖరి అక్షరం చెప్పండి?`
- `"X" లో చివరి అక్షరం ఏది?`
- `"X" పదం చివరన ఏ అక్షరం ఉంది?`
- `"X" పదం యొక్క అంతిమ అక్షరం ఏమిటి?`

**Answer**: Last akshara from `get_telugu_aksharas(word)[-1]`

---

### S10: Word Comparison (పద పోలిక) — 10,000 pairs

**Example:**
```
ఏ పదం పొడవు, "నీరు" లేదా "పుస్తకం"?  → పుస్తకం
```

**10 templates (5 longer + 5 shorter):**

Longer:
- `ఏ పదం పొడవు, "X" లేదా "Y"?`
- `"X" మరియు "Y" లో ఏ పదం పొడవు?`
- `"X" మరియు "Y" లో ఏది ఎక్కువ అక్షరాలు కలిగి ఉంది?`
- `ఏ పదం ఎక్కువ అక్షరాలు కలిగి ఉంది, "X" లేదా "Y"?`
- `"X" మరియు "Y" లలో పొడవైన పదం ఏది?`

Shorter:
- `ఏ పదం చిన్నది, "X" లేదా "Y"?`
- `"X" మరియు "Y" లో ఏ పదం చిన్నది?`
- `"X" మరియు "Y" లో ఏది తక్కువ అక్షరాలు కలిగి ఉంది?`
- `ఏ పదం తక్కువ అక్షరాలు కలిగి ఉంది, "X" లేదా "Y"?`
- `"X" మరియు "Y" లలో చిన్న పదం ఏది?`

**Critical**: Skip pairs where both words have the same akshara count.
**Comparison**: `len(get_telugu_aksharas(word1))` vs `len(get_telugu_aksharas(word2))`

---

## Pipeline

### Generation Flow

```
S1–S10 generators → 200,000 individual Q&A pairs (each ~20-50 tokens)
        ↓
Each generator writes: group1_s1.txt ... group1_s10.txt
        ↓
Orchestrator (generate_group1_telugu_dataset.py):
  1. Reads all 10 files
  2. Loads (query, answer) tuples
  3. Shuffles all 200,000 pairs
  4. Calls combine_qa_pairs_to_reach_min_tokens_kannada(pairs, min_tokens=512)
  5. Writes output/group1_telugu.txt
        ↓
Output: ~400-600 data rows, each ≥512 tokens
```

### prompt_utils.py Change Required

Add Telugu Unicode range to `count_tokens()`:
```python
is_telugu = "\u0C00" <= ch <= "\u0C7F"
if is_devanagari or is_kannada or is_telugu:
    count += 1
    i += 1
    continue
```

---

## File Structure

```
curriculum_training_data/
├── group1_telugu/
│   ├── TELUGU_DATASET_APPROACH.md              # This document
│   ├── telugu_grammar.py                       # get_telugu_aksharas(), constants
│   ├── telugu_vocabulary.py                    # 950+ words, numbers, rhyming pairs, vargas
│   ├── generate_s1_spelling.py                 # S1: అక్షరక్రమం (30,000)
│   ├── generate_s2_letter_position.py          # S2: అక్షర స్థానం (26,000)
│   ├── generate_s3_sound.py                    # S3: ధ్వని (20,000)
│   ├── generate_s4_count.py                    # S4: అక్షర గణన (26,000)
│   ├── generate_s5_rhyme.py                    # S5: ప్రాస (20,000)
│   ├── generate_s6_classify.py                 # S6: వర్గీకరణ (20,000)
│   ├── generate_s7_position.py                 # S7: అక్షరం స్థానం (18,000)
│   ├── generate_s8_numbers.py                  # S8: సంఖ్య అక్షరక్రమం (12,000)
│   ├── generate_s9_last.py                     # S9: చివరి అక్షరం (18,000)
│   ├── generate_s10_compare.py                 # S10: పద పోలిక (10,000)
│   └── generate_group1_telugu_dataset.py       # Orchestrator
├── prompt_utils.py                             # UPDATE: add Telugu range
└── output/
    └── group1_telugu.txt                       # Final combined output
```

---

## Build Sequence

Must be built in this exact order (dependencies flow downward):

| Step | File | Depends On |
|------|------|-----------|
| 0 | `prompt_utils.py` (add Telugu range) | Nothing |
| 1 | `telugu_vocabulary.py` | Nothing |
| 2 | `telugu_grammar.py` | Nothing |
| 3 | `generate_s1_spelling.py` | vocab, grammar |
| 4 | `generate_s4_count.py` | vocab, grammar, s1 (imports grapheme fn) |
| 5 | `generate_s2_letter_position.py` | vocab, grammar, s1 |
| 6 | `generate_s9_last.py` | vocab, grammar, s1 |
| 7 | `generate_s7_position.py` | vocab, grammar, s1 |
| 8 | `generate_s6_classify.py` | vocab |
| 9 | `generate_s5_rhyme.py` | vocab (RHYMING_PAIRS) |
| 10 | `generate_s3_sound.py` | vocab, grammar, s1, RHYMING_PAIRS, VARGAS |
| 11 | `generate_s8_numbers.py` | vocab (NUMBERS), grammar |
| 12 | `generate_s10_compare.py` | vocab, grammar, s1 |
| 13 | `generate_group1_telugu_dataset.py` | All S1-S10 outputs |

---

## How to Run

From `curriculum_training_data/`:

```bash
python group1_telugu/generate_s1_spelling.py
python group1_telugu/generate_s2_letter_position.py
python group1_telugu/generate_s3_sound.py
python group1_telugu/generate_s4_count.py
python group1_telugu/generate_s5_rhyme.py
python group1_telugu/generate_s6_classify.py
python group1_telugu/generate_s7_position.py
python group1_telugu/generate_s8_numbers.py
python group1_telugu/generate_s9_last.py
python group1_telugu/generate_s10_compare.py
python group1_telugu/generate_group1_telugu_dataset.py
```

The final script reads `group1_s1.txt` through `group1_s10.txt` from `group1_telugu/`, combines Q&A pairs to ≥512 tokens per line, and writes `output/group1_telugu.txt`.

---

**Last Updated**: 2026-02-13
**Version**: 1.0
**Status**: Design approved, pending implementation
