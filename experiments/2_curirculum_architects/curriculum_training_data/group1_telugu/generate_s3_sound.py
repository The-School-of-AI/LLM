#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ధ్వని) questions - Telugu
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import get_telugu_grapheme_clusters  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    CLASSIFICATION_CATEGORIES,
    RHYMING_PAIRS,
    VARGAS,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

ALL_WORDS = ALL_WORDS_UNIQUE * 30
unique_words = list(set(ALL_WORDS))

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

animals = list(CLASSIFICATION_CATEGORIES.get("జంతువు", []))
animals_by_first = {}
for w in animals:
    if not w:
        continue
    for ln in [2, 1]:
        if len(w) >= ln:
            k = w[:ln]
            animals_by_first.setdefault(k, []).append(w)
            break

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


# Telugu verb endings heuristic
def get_verbs(word_list):
    verbs = []
    verb_endings = ["చు", "డు", "గు", "ను", "తు", "పు", "వు", "ళ్ళు"]
    for w in word_list:
        if any(w.endswith(end) for end in verb_endings):
            verbs.append(w)
    return verbs


VERBS = get_verbs(unique_words)

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


# Template types
TEMPLATES = [
    ('"{word}" పదానికి ప్రాసబద్ధమైన పదం ఏది?', "rhyme_word"),
    ('"{letter}" అక్షరంతో ప్రారంభమయ్యే ఒక పదం చెప్పండి?', "word_starting"),
    ('"{word1}" మరియు "{word2}" పదాలు ప్రాసబద్ధమా?', "do_rhyme_yes_no"),
    ('"అ" స్వరం ఉన్న పదం ఏది?', "word_with_vowel"),
    ('"{letter}" అక్షరంతో అంతమయ్యే పదం చెప్పండి?', "word_ending"),
    ('"హ" మరియు "ప" అక్షరాల ఉచ్చారణ ఒకటేనా?', "same_pronunciation"),
    ('"బా" అక్షరంతో మొదలయ్యే జంతువు పేరు ఏమిటి?', "animal_starting"),
    ('"త" వర్గ అక్షరాల ధ్వనిని గుర్తించండి?', "identify_sound"),
    ('"{word}" పదంలో మొదటి ధ్వని ఏమిటి?', "first_sound"),
    ('"న" అక్షరం యొక్క అనునాసిక ధ్వని ఉన్న పదం ఏది?', "word_with_nasal"),
    ('"{word}" పదానికి ప్రాసమయ్యే మరొక పదం చెప్పండి?', "rhyme_word"),
    ('"{word1}" మరియు "{word2}" పదాలు ప్రాస అవుతాయా?', "do_rhyme_yes_no"),
    ('"{letter}" అక్షరంతో మొదలయ్యే పండు పేరు చెప్పండి?', "fruit_starting"),
    ('"{word}" పదానికి సమానమైన ధ్వని ఉన్న పదం ఏది?', "similar_sound"),
    ('"{letter}" ధ్వనితో అంతమయ్యే పదాన్ని చెప్పండి?', "word_ending"),
    ('"శ" మరియు "ష" ఉచ్చారణలో సమానత ఉందా?', "same_pronunciation_sh_sha"),
    ('"{letter}" అక్షరం యొక్క ధ్వని ఉన్న రెండు పదాలు చెప్పండి?', "two_words_with_sound"),
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
        non_rhyming_words = [w for w in unique_words if w != word1 and not do_rhyme(word1, w)]
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
        non_rhyming_words = [w for w in unique_words if w != word1 and not do_rhyme(word1, w)]
        if not non_rhyming_words:
            continue
        word2 = random.choice(non_rhyming_words)
        q = TEMPLATES[11][0].format(word1=word1, word2=word2)
        a = "కాదు"
    key = ("do_rhyme_yes_no", word1, word2, TEMPLATES[11][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 4. word_with_vowel (అ)
for w in words_by_first.get("అ", [])[:50]:
    q = TEMPLATES[3][0]
    a = w
    key = ("word_with_vowel", w, TEMPLATES[3][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 5. word_ending
for letter, word_list in list(words_by_last.items())[:80]:
    if not word_list:
        continue
    w = random.choice(word_list)
    q = TEMPLATES[4][0].format(letter=letter)
    a = w
    key = ("word_ending", letter, TEMPLATES[4][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 15. word_ending variant
for letter, word_list in list(words_by_last.items())[:80]:
    if not word_list:
        continue
    w = random.choice(word_list)
    q = TEMPLATES[14][0].format(letter=letter)
    a = w
    key = ("word_ending", letter, TEMPLATES[14][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 6. same_pronunciation: హ and ప - కాదు
q = TEMPLATES[5][0]
a = "కాదు"
key = ("same_pronunciation", TEMPLATES[5][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# 16. same_pronunciation_sh_sha: శ and ష - అవును
q = TEMPLATES[15][0]
a = "అవును"
key = ("same_pronunciation_sh_sha", TEMPLATES[15][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# 7. animal_starting with బా
ba_animals = animals_by_first.get("బా", [])
if not ba_animals:
    ba_animals = ["బాతు"]
for w in ba_animals:
    q = TEMPLATES[6][0]
    a = w
    key = ("animal_starting", "బా", w, TEMPLATES[6][0])
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

# 8. identify_sound: త వర్గ
q = TEMPLATES[7][0]
a = ", ".join(VARGAS.get("త", []))
key = ("identify_sound", TEMPLATES[7][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# 9. first_sound
for word in unique_words[:150]:
    clusters = get_telugu_grapheme_clusters(word)
    if not clusters:
        continue
    q = TEMPLATES[8][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[8][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 19. first_sound variant
for word in unique_words[:150]:
    clusters = get_telugu_grapheme_clusters(word)
    if not clusters:
        continue
    q = TEMPLATES[18][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[18][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 10. word_with_nasal (న)
words_with_n = [w for w in unique_words if "న" in w]
for w in (words_with_n or unique_words)[:80]:
    q = TEMPLATES[9][0]
    a = w
    key = ("word_with_nasal", w, TEMPLATES[9][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 14 & 20. similar_sound
for word in unique_words[:100]:
    word_clusters = get_telugu_grapheme_clusters(word)
    if not word_clusters:
        continue
    q_idx = random.choice([13, 19])
    q = TEMPLATES[q_idx][0].format(word=word)
    similar_words = [
        w
        for w in unique_words
        if w != word and get_telugu_grapheme_clusters(w) and get_telugu_grapheme_clusters(w)[-1] == word_clusters[-1]
    ]
    a = random.choice(similar_words) if similar_words else word
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

# 18. verb_starting
for letter, word_list in words_by_first.items():
    verbs_starting_with_letter = [v for v in VERBS if v.startswith(letter)]
    if not verbs_starting_with_letter:
        continue
    w = random.choice(verbs_starting_with_letter)
    q = TEMPLATES[17][0].format(letter=letter)
    a = w
    key = ("verb_starting", letter, TEMPLATES[17][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# Fill to target
while len(samples) < target_count:
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
            non_rhyming_words = [w for w in unique_words if w != word1 and not do_rhyme(word1, w)]
            if not non_rhyming_words:
                q, a = None, None
            else:
                word2 = random.choice(non_rhyming_words)
                a = "కాదు"
        if a is not None:
            q = template_text.format(word1=word1, word2=word2)
    elif ttype == "word_with_vowel":
        lst = words_by_first.get("అ", unique_words)
        a = random.choice(lst) if lst else random.choice(unique_words)
        q = template_text
    elif ttype == "word_ending" and words_by_last:
        letter = random.choice(list(words_by_last.keys()))
        a = random.choice(words_by_last[letter])
        q = template_text.format(letter=letter)
    elif ttype == "same_pronunciation":
        q, a = template_text, "కాదు"
    elif ttype == "same_pronunciation_sh_sha":
        q, a = template_text, "అవును"
    elif ttype == "animal_starting":
        lst = animals_by_first.get("బా", [])
        if not lst:
            lst = ["బాతు"]
        a = random.choice(lst)
        q = template_text
    elif ttype == "identify_sound":
        q, a = template_text, ", ".join(VARGAS.get("త", []))
    elif ttype == "first_sound":
        word = random.choice(unique_words)
        clusters = get_telugu_grapheme_clusters(word)
        if not clusters:
            q, a = None, None
        else:
            q = template_text.format(word=word)
            a = clusters[0]
    elif ttype == "word_with_nasal":
        lst = [w for w in unique_words if "న" in w] or unique_words
        a = random.choice(lst)
        q = template_text
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
                if w != word and get_telugu_grapheme_clusters(w) and get_telugu_grapheme_clusters(w)[-1] == word_clusters[-1]
            ]
            a = random.choice(similar_words) if similar_words else word
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
    elif ttype == "verb_starting" and words_by_first:
        letter = random.choice(list(words_by_first.keys()))
        verbs_starting_with_letter = [v for v in VERBS if v.startswith(letter)]
        if verbs_starting_with_letter:
            a = random.choice(verbs_starting_with_letter)
            q = template_text.format(letter=letter)
        else:
            q, a = None, None
    else:
        q, a = None, None

    if q is not None and a is not None:
        samples.append((q, a))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S3 Sound Matching (Telugu): Generated {len(samples)} samples")
