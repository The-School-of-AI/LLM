#!/usr/bin/env python3
"""
Generate Statement 11: Ottulu & Gunintalu (ఒత్తులు & గుణింతాలు) questions - Telugu
Teaches the internal structure of aksharas: consonant-vowel composition and conjuncts.
Target: 8,000 pairs (4% of 208,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import get_telugu_grapheme_clusters  # noqa: E402
from group1_telugu.telugu_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# ─── Telugu Consonants (హల్లులు) ───
CONSONANTS = [
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ల", "వ",
    "శ", "ష", "స", "హ", "ళ",
]

# Common consonants (skip rare ఙ, ఝ, ఞ for gunintalu charts)
COMMON_CONSONANTS = [
    "క", "ఖ", "గ", "ఘ",
    "చ", "ఛ", "జ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ల", "వ",
    "శ", "ష", "స", "హ", "ళ",
]

# ─── Telugu Vowels & Vowel Signs (అచ్చులు & గుణింతాలు) ───
# (vowel_independent, vowel_sign_unicode, vowel_name)
# అ has no vowel sign (inherent vowel)
VOWELS_WITH_SIGNS = [
    ("ఆ", "\u0C3E", "ఆ-కారం"),
    ("ఇ", "\u0C3F", "ఇ-కారం"),
    ("ఈ", "\u0C40", "ఈ-కారం"),
    ("ఉ", "\u0C41", "ఉ-కారం"),
    ("ఊ", "\u0C42", "ఊ-కారం"),
    ("ఋ", "\u0C43", "ఋ-కారం"),
    ("ఎ", "\u0C46", "ఎ-కారం"),
    ("ఏ", "\u0C47", "ఏ-కారం"),
    ("ఐ", "\u0C48", "ఐ-కారం"),
    ("ఒ", "\u0C4A", "ఒ-కారం"),
    ("ఓ", "\u0C4B", "ఓ-కారం"),
    ("ఔ", "\u0C4C", "ఔ-కారం"),
]

# Independent vowels for classification
INDEPENDENT_VOWELS = ["అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ"]

TELUGU_VIRAMA = "\u0C4D"  # ్


def build_gunintam_chart(consonant: str) -> list[str]:
    """Build full gunintam chart for a consonant: క, కా, కి, కీ, ..."""
    chart = [consonant]  # inherent అ
    for _, sign, _ in VOWELS_WITH_SIGNS:
        chart.append(consonant + sign)
    return chart


def find_conjuncts_in_word(word: str) -> list[str]:
    """Find conjunct aksharas (those containing virama) in a word."""
    clusters = get_telugu_grapheme_clusters(word)
    conjuncts = []
    for c in clusters:
        if TELUGU_VIRAMA in c:
            conjuncts.append(c)
    return conjuncts


def decompose_conjunct(conjunct: str) -> list[str]:
    """Decompose a conjunct akshara into its component consonants."""
    parts = conjunct.split(TELUGU_VIRAMA)
    consonants = []
    for p in parts:
        if p:
            # Take just the base consonant (first char)
            consonants.append(p[0] if p else p)
    return consonants


# ─── Precompute words with conjuncts ───
words_with_conjuncts = []
for word in ALL_WORDS_UNIQUE:
    conjuncts = find_conjuncts_in_word(word)
    if conjuncts:
        words_with_conjuncts.append((word, conjuncts))

# ─── Templates ───

# Gunintalu: consonant + vowel sign = combined form
TEMPLATES_GUNINTAM_COMBINE = [
    '"{cons}" అక్షరానికి {vowel_name} జోడిస్తే ఏమవుతుంది?',
    '"{cons}" హల్లుకు {vowel_name} కలిపితే ఏ అక్షరం వస్తుంది?',
    '"{cons}" కు {vowel_name} చేర్చితే ఏమవుతుంది?',
]

# Gunintalu: identify base consonant
TEMPLATES_GUNINTAM_BASE = [
    '"{combined}" అక్షరంలో ఉన్న మూల హల్లు ఏమిటి?',
    '"{combined}" లో ఏ హల్లు ఉంది?',
    '"{combined}" అక్షరం యొక్క మూల వ్యంజనం ఏది?',
]

# Gunintalu: identify vowel in combined form
TEMPLATES_GUNINTAM_VOWEL = [
    '"{combined}" అక్షరంలో ఏ అచ్చు గుణింతం ఉంది?',
    '"{combined}" లో ఉన్న గుణింతం ఏమిటి?',
    '"{combined}" అక్షరంలో ఏ స్వర చిహ్నం ఉంది?',
]

# Gunintalu: full chart
TEMPLATES_GUNINTAM_CHART = [
    '"{cons}" యొక్క గుణింతాలు చెప్పండి?',
    '"{cons}" అక్షరం యొక్క గుణింతాలు ఏమిటి?',
    '"{cons}" హల్లు యొక్క గుణింత రూపాలు వ్రాయండి?',
]

# Ottulu: identify conjunct in word
TEMPLATES_OTTULU_IDENTIFY = [
    '"{word}" పదంలో ఒత్తు ఏమిటి?',
    '"{word}" పదంలోని ఒత్తక్షరాన్ని గుర్తించండి?',
    '"{word}" పదంలో ఉన్న సంయుక్తాక్షరం ఏది?',
]

# Ottulu: decompose conjunct
TEMPLATES_OTTULU_DECOMPOSE = [
    '"{conjunct}" సంయుక్తాక్షరంలో ఏయే హల్లులు ఉన్నాయి?',
    '"{conjunct}" అనే ఒత్తక్షరంలో కలిసిన అక్షరాలు ఏమిటి?',
    '"{conjunct}" లో ఏయే వ్యంజనాలు కలిసి ఉన్నాయి?',
]

# Ottulu: does word have conjunct?
TEMPLATES_OTTULU_EXISTS = [
    '"{word}" పదంలో సంయుక్తాక్షరం ఉందా?',
    '"{word}" పదంలో ఒత్తు ఉందా?',
]

# Classification: vowel or consonant
TEMPLATES_CLASSIFY = [
    '"{char}" అక్షరం స్వరమా లేదా వ్యంజనమా?',
    '"{char}" అనేది అచ్చా లేదా హల్లా?',
]

# ─── Generate samples ───
samples = []
target_count = 8000
seen = set()

# 1. Gunintalu: consonant + vowel sign = combined (systematic)
for cons in COMMON_CONSONANTS:
    for vowel_indep, sign, vowel_name in VOWELS_WITH_SIGNS:
        combined = cons + sign
        template = random.choice(TEMPLATES_GUNINTAM_COMBINE)
        q = template.format(cons=cons, vowel_name=vowel_name)
        a = combined
        key = ("combine", cons, vowel_name)
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 2. Gunintalu: identify base consonant
for cons in COMMON_CONSONANTS:
    for vowel_indep, sign, vowel_name in random.sample(VOWELS_WITH_SIGNS, min(4, len(VOWELS_WITH_SIGNS))):
        combined = cons + sign
        template = random.choice(TEMPLATES_GUNINTAM_BASE)
        q = template.format(combined=combined)
        a = cons
        key = ("base", combined)
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 3. Gunintalu: identify vowel in combined form
for cons in COMMON_CONSONANTS:
    for vowel_indep, sign, vowel_name in random.sample(VOWELS_WITH_SIGNS, min(4, len(VOWELS_WITH_SIGNS))):
        combined = cons + sign
        template = random.choice(TEMPLATES_GUNINTAM_VOWEL)
        q = template.format(combined=combined)
        a = vowel_name
        key = ("vowel", combined)
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 4. Gunintalu: full chart for each consonant
for cons in COMMON_CONSONANTS:
    chart = build_gunintam_chart(cons)
    template = random.choice(TEMPLATES_GUNINTAM_CHART)
    q = template.format(cons=cons)
    a = ", ".join(chart)
    key = ("chart", cons)
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 5. Ottulu: identify conjunct in word
for word, conjuncts in words_with_conjuncts:
    for conj in conjuncts:
        template = random.choice(TEMPLATES_OTTULU_IDENTIFY)
        q = template.format(word=word)
        a = conj
        key = ("ottulu_id", word, conj)
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 6. Ottulu: decompose conjunct
seen_conjuncts = set()
for word, conjuncts in words_with_conjuncts:
    for conj in conjuncts:
        if conj in seen_conjuncts:
            continue
        seen_conjuncts.add(conj)
        parts = decompose_conjunct(conj)
        if len(parts) >= 2:
            template = random.choice(TEMPLATES_OTTULU_DECOMPOSE)
            q = template.format(conjunct=conj)
            a = ", ".join(parts)
            key = ("decompose", conj)
            if key not in seen:
                seen.add(key)
                samples.append((q, a))

# 7. Ottulu: does word have conjunct? (yes/no)
for word in ALL_WORDS_UNIQUE[:300]:
    conjuncts = find_conjuncts_in_word(word)
    template = random.choice(TEMPLATES_OTTULU_EXISTS)
    q = template.format(word=word)
    if conjuncts:
        a = f"అవును, {conjuncts[0]}"
    else:
        a = "లేదు"
    key = ("exists", word, template)
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 8. Classification: vowel or consonant
for char in INDEPENDENT_VOWELS:
    template = random.choice(TEMPLATES_CLASSIFY)
    q = template.format(char=char)
    a = "స్వరం"
    key = ("classify", char)
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

for char in CONSONANTS:
    template = random.choice(TEMPLATES_CLASSIFY)
    q = template.format(char=char)
    a = "వ్యంజనం"
    key = ("classify", char)
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# ─── Fill to target with random variations ───
while len(samples) < target_count:
    choice = random.random()

    if choice < 0.35:
        # Random gunintalu combination
        cons = random.choice(COMMON_CONSONANTS)
        vowel_indep, sign, vowel_name = random.choice(VOWELS_WITH_SIGNS)
        combined = cons + sign
        r = random.random()
        if r < 0.4:
            template = random.choice(TEMPLATES_GUNINTAM_COMBINE)
            q = template.format(cons=cons, vowel_name=vowel_name)
            a = combined
        elif r < 0.7:
            template = random.choice(TEMPLATES_GUNINTAM_BASE)
            q = template.format(combined=combined)
            a = cons
        else:
            template = random.choice(TEMPLATES_GUNINTAM_VOWEL)
            q = template.format(combined=combined)
            a = vowel_name
        samples.append((q, a))

    elif choice < 0.55:
        # Random gunintalu chart
        cons = random.choice(COMMON_CONSONANTS)
        chart = build_gunintam_chart(cons)
        template = random.choice(TEMPLATES_GUNINTAM_CHART)
        q = template.format(cons=cons)
        a = ", ".join(chart)
        samples.append((q, a))

    elif choice < 0.80:
        # Random ottulu
        if words_with_conjuncts:
            word, conjuncts = random.choice(words_with_conjuncts)
            conj = random.choice(conjuncts)
            r = random.random()
            if r < 0.4:
                template = random.choice(TEMPLATES_OTTULU_IDENTIFY)
                q = template.format(word=word)
                a = conj
            elif r < 0.7:
                parts = decompose_conjunct(conj)
                if len(parts) >= 2:
                    template = random.choice(TEMPLATES_OTTULU_DECOMPOSE)
                    q = template.format(conjunct=conj)
                    a = ", ".join(parts)
                else:
                    continue
            else:
                template = random.choice(TEMPLATES_OTTULU_EXISTS)
                q = template.format(word=word)
                a = f"అవును, {conj}"
            samples.append((q, a))

    elif choice < 0.90:
        # Random ottulu exists (no conjunct)
        word = random.choice(ALL_WORDS_UNIQUE)
        conjuncts = find_conjuncts_in_word(word)
        if not conjuncts:
            template = random.choice(TEMPLATES_OTTULU_EXISTS)
            q = template.format(word=word)
            a = "లేదు"
            samples.append((q, a))

    else:
        # Random classification
        if random.random() < 0.5:
            char = random.choice(INDEPENDENT_VOWELS)
            a = "స్వరం"
        else:
            char = random.choice(CONSONANTS)
            a = "వ్యంజనం"
        template = random.choice(TEMPLATES_CLASSIFY)
        q = template.format(char=char)
        samples.append((q, a))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s11.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S11 Ottulu & Gunintalu (Telugu): Generated {len(samples)} samples")
