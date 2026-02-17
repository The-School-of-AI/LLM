#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ధ్వని) questions - Telugu
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import (  # noqa: E402
    get_telugu_grapheme_clusters,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    CLASSIFICATION_CATEGORIES,
    RHYMING_PAIRS,
    VARGAS,
)

ALL_WORDS = ALL_WORDS_UNIQUE * 30
unique_words = list(set(ALL_WORDS))

# Pre-filter: words with >= 2 aksharas (avoids trivial self-answers)
multi_akshara_words = [
    w for w in unique_words if len(get_telugu_grapheme_clusters(w)) >= 2
]

# Indices for sound questions
words_by_first = {}
for w in unique_words:
    if not w:
        continue
    c = w[0]
    words_by_first.setdefault(c, []).append(w)

words_by_last = {}
for w in unique_words:
    clusters = get_telugu_grapheme_clusters(w)
    if clusters:
        last = clusters[-1]
        words_by_last.setdefault(last, []).append(w)

# Telugu vowels for word_with_vowel (parameterized)
TELUGU_VOWELS = ["అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఎ", "ఏ", "ఒ", "ఓ"]
words_by_vowel = {}
for v in TELUGU_VOWELS:
    words_by_vowel[v] = [w for w in unique_words if v in w]

# Telugu nasals for word_with_nasal (parameterized)
TELUGU_NASALS = ["ణ", "న", "మ"]
words_by_nasal = {}
for n in TELUGU_NASALS:
    words_by_nasal[n] = [w for w in unique_words if n in w]

animals = list(CLASSIFICATION_CATEGORIES.get("జంతువు", []))
animals_by_first = {}
for w in animals:
    if not w:
        continue
    c = w[0]
    animals_by_first.setdefault(c, []).append(w)

# Rhyme set for "do they rhyme?"
rhyme_set = set()
for a, b in RHYMING_PAIRS.items():
    rhyme_set.add((a, b))
    rhyme_set.add((b, a))


def do_rhyme(w1: str, w2: str) -> bool:
    if (w1, w2) in rhyme_set or (w2, w1) in rhyme_set:
        return True
    clusters1 = get_telugu_grapheme_clusters(w1)
    clusters2 = get_telugu_grapheme_clusters(w2)
    if clusters1 and clusters2 and clusters1[-1] == clusters2[-1]:
        return True
    return False


# Curated Telugu verbs (క్రియాపదాలు)
VERBS = [
    "అడుగు",
    "అమ్ము",
    "ఆడు",
    "ఆపు",
    "ఆలోచించు",
    "ఇచ్చు",
    "ఈదు",
    "ఉండు",
    "ఉడుకు",
    "ఊదు",
    "ఎగురు",
    "ఎక్కు",
    "ఏడ్చు",
    "కడుగు",
    "కట్టు",
    "కలుపు",
    "కాల్చు",
    "కుట్టు",
    "కొట్టు",
    "కొను",
    "కోయు",
    "కురియు",
    "గెలుచు",
    "చదువు",
    "చూడు",
    "చెప్పు",
    "చేయు",
    "తిను",
    "తాగు",
    "తిరుగు",
    "తీయు",
    "తెచ్చు",
    "తెరచు",
    "తరుగు",
    "తడుపు",
    "దిగు",
    "దూకు",
    "నడుచు",
    "నడుపు",
    "నరుకు",
    "నవ్వు",
    "నేయు",
    "పడు",
    "పండు",
    "పంపు",
    "పాడు",
    "పెట్టు",
    "పెంచు",
    "బోధించు",
    "మాట్లాడు",
    "మార్చు",
    "మరచు",
    "మూయు",
    "ముగించు",
    "లేచు",
    "వచ్చు",
    "వండు",
    "వెళ్ళు",
    "విను",
    "విప్పు",
    "వ్రాయు",
    "సాధించు",
    "హరించు",
]

VERBS_BY_FIRST = {}
for v in VERBS:
    if v:
        VERBS_BY_FIRST.setdefault(v[0], []).append(v)

# Fruits list for fruit_starting
TELUGU_FRUITS = [
    "మామిడి",
    "అరటి",
    "బత్తాయి",
    "ద్రాక్ష",
    "దానిమ్మ",
    "పనస",
    "జామ",
    "సపోట",
    "కమల",
    "నారింజ",
    "బొప్పాయి",
    "పుచ్చకాయ",
    "కర్బూజ",
    "సీతాఫలం",
    "రామాఫలం",
    "చెర్రీ",
    "నేరేడు",
    "ఖర్జూరం",
    "కొబ్బరి",
    "నిమ్మ",
    "బేడ",
    "జీడి",
    "ఆపిల్",
    "పీచు",
    "అంజీర్",
]

FRUITS_STARTING = {}
for w in TELUGU_FRUITS:
    if not w:
        continue
    c = w[0]
    FRUITS_STARTING.setdefault(c, []).append(w)

# Pronunciation comparison pairs: (letter1, letter2, same_or_not)
PRONUNCIATION_PAIRS = [
    ("హ", "ప", "కాదు"),
    ("శ", "ష", "అవును"),
    ("బ", "వ", "కాదు"),
    ("డ", "ద", "కాదు"),
    ("క", "గ", "కాదు"),
    ("ట", "త", "కాదు"),
    ("ప", "ఫ", "కాదు"),
    ("జ", "ఝ", "కాదు"),
]


# Template types
TEMPLATES = [
    ('"{word}" పదానికి ప్రాసబద్ధమైన పదం ఏది?', "rhyme_word"),
    ('"{letter}" అక్షరంతో ప్రారంభమయ్యే ఒక పదం చెప్పండి?', "word_starting"),
    ('"{word1}" మరియు "{word2}" పదాలు ప్రాసబద్ధమా?', "do_rhyme_yes_no"),
    ('"{vowel}" స్వరం ఉన్న పదం ఏది?', "word_with_vowel"),
    ('"{letter}" అక్షరంతో అంతమయ్యే పదం చెప్పండి?', "word_ending"),
    ('"{l1}" మరియు "{l2}" అక్షరాల ఉచ్చారణ ఒకటేనా?', "same_pronunciation"),
    ('"{letter}" అక్షరంతో మొదలయ్యే జంతువు పేరు ఏమిటి?', "animal_starting"),
    ('"{varga}" వర్గ అక్షరాల ధ్వనిని గుర్తించండి?', "identify_sound"),
    ('"{word}" పదంలో మొదటి ధ్వని ఏమిటి?', "first_sound"),
    ('"{nasal}" అక్షరం యొక్క అనునాసిక ధ్వని ఉన్న పదం ఏది?', "word_with_nasal"),
    ('"{word}" పదానికి ప్రాసమయ్యే మరొక పదం చెప్పండి?', "rhyme_word"),
    ('"{word1}" మరియు "{word2}" పదాలు ప్రాస అవుతాయా?', "do_rhyme_yes_no"),
    ('"{letter}" అక్షరంతో మొదలయ్యే పండు పేరు చెప్పండి?', "fruit_starting"),
    ('"{word}" పదానికి సమానమైన ధ్వని ఉన్న పదం ఏది?', "similar_sound"),
    ('"{letter}" ధ్వనితో అంతమయ్యే పదాన్ని చెప్పండి?', "word_ending"),
    ('"{l1}" మరియు "{l2}" ఉచ్చారణలో సమానత ఉందా?', "same_pronunciation"),
    (
        '"{letter}" అక్షరం యొక్క ధ్వని ఉన్న రెండు పదాలు చెప్పండి?',
        "two_words_with_sound",
    ),
    ('"{letter}" అక్షరంతో ప్రారంభమయ్యే క్రియాపదం ఏది?', "verb_starting"),
    ('"{word}" పదంలో మొదటి శబ్దం ఏమిటి?', "first_sound"),
    ('"{word}" పదం యొక్క ధ్వనికి దగ్గరగా ఉన్న పదం చెప్పండి?', "similar_sound"),
]

samples = []
target_count = 20000
seen = set()

# 1. rhyme_word
for word, rhyme_word in RHYMING_PAIRS.items():
    q = TEMPLATES[0][0].format(word=word)
    a = rhyme_word
    key = ("rhyme_word", word, TEMPLATES[0][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 11. another_rhyme
for word, rhyme_word in RHYMING_PAIRS.items():
    q = TEMPLATES[10][0].format(word=word)
    a = rhyme_word
    key = ("rhyme_word", word, TEMPLATES[10][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 2. word_starting
for letter, word_list in words_by_first.items():
    if not word_list:
        continue
    w = random.choice(word_list)
    q = TEMPLATES[1][0].format(letter=letter)
    a = w
    key = ("word_starting", letter, TEMPLATES[1][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 3. do_rhyme_yes_no
for _ in range(100):
    word1 = random.choice(unique_words)
    if word1 in RHYMING_PAIRS:
        word2 = RHYMING_PAIRS[word1]
        q = TEMPLATES[2][0].format(word1=word1, word2=word2)
        a = "అవును"
    else:
        non_rhyming_words = [
            w for w in unique_words if w != word1 and not do_rhyme(word1, w)
        ]
        if not non_rhyming_words:
            continue
        word2 = random.choice(non_rhyming_words)
        q = TEMPLATES[2][0].format(word1=word1, word2=word2)
        a = "కాదు"
    key = ("do_rhyme_yes_no", word1, word2, TEMPLATES[2][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 12. do_rhyme_yes_no variant
for _ in range(100):
    word1 = random.choice(unique_words)
    if word1 in RHYMING_PAIRS:
        word2 = RHYMING_PAIRS[word1]
        q = TEMPLATES[11][0].format(word1=word1, word2=word2)
        a = "అవును"
    else:
        non_rhyming_words = [
            w for w in unique_words if w != word1 and not do_rhyme(word1, w)
        ]
        if not non_rhyming_words:
            continue
        word2 = random.choice(non_rhyming_words)
        q = TEMPLATES[11][0].format(word1=word1, word2=word2)
        a = "కాదు"
    key = ("do_rhyme_yes_no", word1, word2, TEMPLATES[11][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 4. word_with_vowel (parameterized across vowels)
for vowel, word_list in words_by_vowel.items():
    if not word_list:
        continue
    for w in word_list[:50]:
        q = TEMPLATES[3][0].format(vowel=vowel)
        a = w
        key = ("word_with_vowel", vowel, w, TEMPLATES[3][0])
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 5. word_ending (prefer multi-akshara words)
for letter, word_list in list(words_by_last.items())[:80]:
    multi = [w for w in word_list if len(get_telugu_grapheme_clusters(w)) >= 2]
    if not multi:
        continue
    w = random.choice(multi)
    q = TEMPLATES[4][0].format(letter=letter)
    a = w
    key = ("word_ending", letter, TEMPLATES[4][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 15. word_ending variant
for letter, word_list in list(words_by_last.items())[:80]:
    multi = [w for w in word_list if len(get_telugu_grapheme_clusters(w)) >= 2]
    if not multi:
        continue
    w = random.choice(multi)
    q = TEMPLATES[14][0].format(letter=letter)
    a = w
    key = ("word_ending", letter, TEMPLATES[14][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 6 & 16. same_pronunciation (parameterized across pairs)
for l1, l2, answer in PRONUNCIATION_PAIRS:
    q = TEMPLATES[5][0].format(l1=l1, l2=l2)
    a = answer
    key = ("same_pronunciation", l1, l2, TEMPLATES[5][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))
    q2 = TEMPLATES[15][0].format(l1=l1, l2=l2)
    key2 = ("same_pronunciation", l1, l2, TEMPLATES[15][0])
    if key2 not in seen:
        seen.add(key2)
        samples.append((q2, a))

# 7. animal_starting (parameterized across starting letters)
for letter, animal_list in animals_by_first.items():
    if not animal_list:
        continue
    for w in animal_list[:5]:
        q = TEMPLATES[6][0].format(letter=letter)
        a = w
        key = ("animal_starting", letter, w, TEMPLATES[6][0])
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 13. fruit_starting
for letter, fruit_list in FRUITS_STARTING.items():
    if not fruit_list:
        continue
    w = random.choice(fruit_list)
    q = TEMPLATES[12][0].format(letter=letter)
    a = w
    key = ("fruit_starting", letter, TEMPLATES[12][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 8. identify_sound (parameterized across vargas)
for varga_name, varga_letters in VARGAS.items():
    q = TEMPLATES[7][0].format(varga=varga_name)
    a = ", ".join(varga_letters)
    key = ("identify_sound", varga_name, TEMPLATES[7][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 9. first_sound (skip single-akshara words to avoid trivial self-answers)
for word in multi_akshara_words[:150]:
    clusters = get_telugu_grapheme_clusters(word)
    if not clusters or len(clusters) < 2:
        continue
    q = TEMPLATES[8][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[8][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 19. first_sound variant
for word in multi_akshara_words[:150]:
    clusters = get_telugu_grapheme_clusters(word)
    if not clusters or len(clusters) < 2:
        continue
    q = TEMPLATES[18][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[18][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 10. word_with_nasal (parameterized across nasals)
for nasal, word_list in words_by_nasal.items():
    if not word_list:
        continue
    for w in word_list[:80]:
        q = TEMPLATES[9][0].format(nasal=nasal)
        a = w
        key = ("word_with_nasal", nasal, w, TEMPLATES[9][0])
        if key not in seen:
            seen.add(key)
            samples.append((q, a))

# 14 & 20. similar_sound (skip if no similar word found)
for word in unique_words[:100]:
    word_clusters = get_telugu_grapheme_clusters(word)
    if not word_clusters:
        continue
    similar_words = [
        w
        for w in unique_words
        if w != word
        and get_telugu_grapheme_clusters(w)
        and get_telugu_grapheme_clusters(w)[-1] == word_clusters[-1]
    ]
    if not similar_words:
        continue
    q_idx = random.choice([13, 19])
    q = TEMPLATES[q_idx][0].format(word=word)
    a = random.choice(similar_words)
    key = ("similar_sound", word, TEMPLATES[q_idx][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 17. two_words_with_sound
for letter, word_list in list(words_by_first.items())[:50]:
    if len(word_list) < 2:
        continue
    w1, w2 = random.sample(word_list, 2)
    q = TEMPLATES[16][0].format(letter=letter)
    a = f"{w1}, {w2}"
    key = ("two_words_with_sound", letter, TEMPLATES[16][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 18. verb_starting (using curated verb list)
for letter, verb_list in VERBS_BY_FIRST.items():
    if not verb_list:
        continue
    w = random.choice(verb_list)
    q = TEMPLATES[17][0].format(letter=letter)
    a = w
    key = ("verb_starting", letter, TEMPLATES[17][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# Track seen lines for dedup in fill loop
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

# Fill to target
max_attempts = target_count * 10
fill_attempts = 0
while len(samples) < target_count and fill_attempts < max_attempts:
    fill_attempts += 1
    tpl_full, ttype = random.choice(TEMPLATES)
    q, a = None, None
    template_text = tpl_full

    if ttype == "rhyme_word" and RHYMING_PAIRS:
        word = random.choice(list(RHYMING_PAIRS.keys()))
        a = RHYMING_PAIRS[word]
        q = template_text.format(word=word)
    elif ttype == "word_starting" and words_by_first:
        letter = random.choice(list(words_by_first.keys()))
        a = random.choice(words_by_first[letter])
        q = template_text.format(letter=letter)
    elif ttype == "do_rhyme_yes_no":
        word1 = random.choice(unique_words)
        if word1 in RHYMING_PAIRS and random.random() < 0.7:
            word2 = RHYMING_PAIRS[word1]
            a = "అవును"
        else:
            non_rhyming_words = [
                w for w in unique_words if w != word1 and not do_rhyme(word1, w)
            ]
            if not non_rhyming_words:
                q, a = None, None
            else:
                word2 = random.choice(non_rhyming_words)
                a = "కాదు"
        if a is not None:
            q = template_text.format(word1=word1, word2=word2)
    elif ttype == "word_with_vowel":
        vowel = random.choice(TELUGU_VOWELS)
        lst = words_by_vowel.get(vowel, [])
        if lst:
            a = random.choice(lst)
            q = template_text.format(vowel=vowel)
        else:
            q, a = None, None
    elif ttype == "word_ending" and words_by_last:
        letter = random.choice(list(words_by_last.keys()))
        multi = [
            w
            for w in words_by_last[letter]
            if len(get_telugu_grapheme_clusters(w)) >= 2
        ]
        if not multi:
            q, a = None, None
        else:
            a = random.choice(multi)
            q = template_text.format(letter=letter)
    elif ttype == "same_pronunciation":
        pair = random.choice(PRONUNCIATION_PAIRS)
        l1, l2, a = pair
        q = template_text.format(l1=l1, l2=l2)
    elif ttype == "animal_starting" and animals_by_first:
        letter = random.choice(list(animals_by_first.keys()))
        lst = animals_by_first[letter]
        if lst:
            a = random.choice(lst)
            q = template_text.format(letter=letter)
        else:
            q, a = None, None
    elif ttype == "identify_sound":
        varga_name = random.choice(list(VARGAS.keys()))
        q = template_text.format(varga=varga_name)
        a = ", ".join(VARGAS[varga_name])
    elif ttype == "first_sound":
        word = random.choice(multi_akshara_words)
        clusters = get_telugu_grapheme_clusters(word)
        if not clusters or len(clusters) < 2:
            q, a = None, None
        else:
            q = template_text.format(word=word)
            a = clusters[0]
    elif ttype == "word_with_nasal":
        nasal = random.choice(TELUGU_NASALS)
        lst = words_by_nasal.get(nasal, [])
        if lst:
            a = random.choice(lst)
            q = template_text.format(nasal=nasal)
        else:
            q, a = None, None
    elif ttype == "fruit_starting" and FRUITS_STARTING:
        letter = random.choice(list(FRUITS_STARTING.keys()))
        a = random.choice(FRUITS_STARTING[letter])
        q = template_text.format(letter=letter)
    elif ttype == "similar_sound":
        word = random.choice(unique_words)
        word_clusters = get_telugu_grapheme_clusters(word)
        if not word_clusters:
            q, a = None, None
        else:
            similar_words = [
                w
                for w in unique_words
                if w != word
                and get_telugu_grapheme_clusters(w)
                and get_telugu_grapheme_clusters(w)[-1] == word_clusters[-1]
            ]
            if not similar_words:
                q, a = None, None
            else:
                a = random.choice(similar_words)
                q = template_text.format(word=word)
    elif ttype == "two_words_with_sound" and words_by_first:
        letter = random.choice(list(words_by_first.keys()))
        words_with_letter = [w for w in unique_words if w.startswith(letter)]
        if len(words_with_letter) >= 2:
            w1, w2 = random.sample(words_with_letter, 2)
            a = f"{w1}, {w2}"
            q = template_text.format(letter=letter)
        else:
            q, a = None, None
    elif ttype == "verb_starting" and VERBS_BY_FIRST:
        letter = random.choice(list(VERBS_BY_FIRST.keys()))
        verb_list = VERBS_BY_FIRST[letter]
        if verb_list:
            a = random.choice(verb_list)
            q = template_text.format(letter=letter)
        else:
            q, a = None, None
    else:
        q, a = None, None

    if q is not None and a is not None and (q, a) not in seen_lines:
        seen_lines.add((q, a))
        samples.append((q, a))

# Final dedup
unique_samples = []
final_seen = set()
for q, a in samples:
    if (q, a) not in final_seen:
        final_seen.add((q, a))
        unique_samples.append((q, a))
samples = unique_samples

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S3 Sound Matching (Telugu): Generated {len(samples)} samples")
