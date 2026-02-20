#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ಧ್ವನಿ ಹೊಂದಿಕೆ) questions - Kannada
User-specified templates: rhyme, word starting/ending with, do they rhyme?, pronunciation, etc.
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    BAD_RHYME_PAIRS,
    CLASSIFICATION_CATEGORIES,
    RHYMING_PAIRS,
    VARGAS,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

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
    clusters = get_kannada_grapheme_clusters(w)
    if clusters:
        last = clusters[-1]
        words_by_last.setdefault(last, []).append(w)

animals = list(CLASSIFICATION_CATEGORIES.get("ಪ್ರಾಣಿ", []))
animals_by_first = {}
for w in animals:
    if not w:
        continue
    # Match by first grapheme or first 1-2 chars for "ಬಾ" style
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
    if (w1, w2) in BAD_RHYME_PAIRS or (w2, w1) in BAD_RHYME_PAIRS:
        return False
    if (w1, w2) in rhyme_set or (w2, w1) in rhyme_set:
        return True
    clusters1 = get_kannada_grapheme_clusters(w1)
    clusters2 = get_kannada_grapheme_clusters(w2)
    if clusters1 and clusters2 and clusters1[-1] == clusters2[-1]:
        return True
    return False


# Helper to get verbs (simple heuristic - ends with "ಸು" or "ಗು" or "ಳು" etc.)
def get_verbs(word_list):
    verbs = []
    verb_endings = ["ಸು", "ಗು", "ಳು", "ಡು", "ಬು", "ವು", "ಡು"]  # Added ಡು to match ಕಟ್ಟು
    for w in word_list:
        if any(w.endswith(end) for end in verb_endings):
            verbs.append(w)
    return verbs


VERBS = get_verbs(unique_words)

# Helper to get fruits (use dedicated fruit list - ವಸ್ತು contains non-fruits like ಡಬ್ಬಿ, ವಾಹನ)
fruits = list(CLASSIFICATION_CATEGORIES.get("ಹಣ್ಣು", []))
FRUITS_STARTING = {}
for w in fruits:
    if not w:
        continue
    c = w[0]
    FRUITS_STARTING.setdefault(c, []).append(w)


# Template types: (template, type, ...) type determines how we fill and answer
# Types: rhyme_word, word_starting, do_rhyme_yes_no, word_with_vowel, word_ending,
#        same_pronunciation, animal_starting, identify_sound, first_sound, word_with_nasal,
#        another_rhyme, fruit_starting, similar_sound, two_words_with_sound, verb_starting, same_pronunciation_sh_sha
TEMPLATES = [
    ('"{word}" ಪದಕ್ಕೆ ಪ್ರಾಸಬದ್ಧವಾದ ಪದ ಯಾವುದು?', "rhyme_word"),
    ('"{letter}" ಅಕ್ಷರದಿಂದ ಪ್ರಾರಂಭವಾಗುವ ಒಂದು ಪದ ಹೇಳಿ?', "word_starting"),
    ('"{word1}" ಮತ್ತು "{word2}" ಪದಗಳು ಪ್ರಾಸವಾಗುತ್ತವೆಯೇ?', "do_rhyme_yes_no"),
    ('"ಅ" ಸ್ವರದ ಧ್ವನಿ ಇರುವ ಪದ ಯಾವುದು?', "word_with_vowel"),
    ('"{letter}" ಅಕ್ಷರದಿಂದ ಕೊನೆಗೊಳ್ಳುವ ಪದ ತಿಳಿಸಿ?', "word_ending"),
    ('"ಹ" ಮತ್ತು "ಪ" ಅಕ್ಷರಗಳ ಉಚ್ಚಾರಣೆ ಒಂದೇ ಆಗಿದೆಯೇ?', "same_pronunciation"),
    ('"ಬಾ" ಅಕ್ಷರದಿಂದ ಶುರುವಾಗುವ ಪ್ರಾಣಿಯ ಹೆಸರೇನು?', "animal_starting"),
    ('"ತ" ವರ್ಗದ ಅಕ್ಷರಗಳ ಧ್ವನಿಯನ್ನು ಗುರುತಿಸಿ?', "identify_sound"),
    ('"{word}" ಪದದ ಮೊದಲ ಧ್ವನಿ ಯಾವುದು?', "first_sound"),
    ('"ನ" ಅಕ್ಷರದ ಅನುನಾಸಿಕ ಧ್ವನಿ ಇರುವ ಪದ ಯಾವುದು?', "word_with_nasal"),
    ('"{word}" ಪದಕ್ಕೆ ಪ್ರಾಸವಾಗುವ ಮತ್ತೊಂದು ಪದ ತಿಳಿಸಿ?', "rhyme_word"),
    ('"{word1}" ಮತ್ತು "{word2}" ಪದಗಳು ಪ್ರಾಸಬದ್ಧವೇ?', "do_rhyme_yes_no"),
    ('"{letter}" ಅಕ್ಷರದಿಂದ ಶುರುವಾಗುವ ಹಣ್ಣಿನ ಹೆಸರು ಹೇಳಿ?', "fruit_starting"),
    ('"{word}" ಪದಕ್ಕೆ ಹೋಲುವ ಧ್ವನಿಯ ಪದ ಯಾವುದು?', "similar_sound"),
    ('"{letter}" ಧ್ವನಿಯಿಂದ ಕೊನೆಗೊಳ್ಳುವ ಪದವನ್ನು ಹೆಸರಿಸಿ?', "word_ending"),
    ('"ಶ" ಮತ್ತು "ಷ" ಉಚ್ಚಾರಣೆಯಲ್ಲಿ ಸಮಾನತೆ ಇದೆಯೇ?', "same_pronunciation_sh_sha"),
    ('"{letter}" ಅಕ್ಷರದ ಧ್ವನಿ ಇರುವ ಎರಡು ಪದಗಳನ್ನು ನೀಡಿ?', "two_words_with_sound"),
    ('"{letter}" ಅಕ್ಷರದಿಂದ ಆರಂಭವಾಗುವ ಕ್ರಿಯಾಪದ ಯಾವುದು?', "verb_starting"),
    (
        '"{word}" ಪದದ ಮೊದಲ ಅಕ್ಷರ ಯಾವುದು?',
        "first_sound",
    ),  # ಶಬ್ದ=ಪದ; ಅಕ್ಷರ=syllable/letter
    ('"{word}" ಪದದ ಧ್ವನಿಗೆ ಹತ್ತಿರವಿರುವ ಪದ ತಿಳಿಸಿ?', "similar_sound"),
]

samples = []
target_count = 20000
seen = set()

# Initial samples for unique combinations to ensure coverage
# 1. rhyme_word
for word, rhyme_word in RHYMING_PAIRS.items():
    q = TEMPLATES[0][0].format(word=word)
    a = rhyme_word
    key = ("rhyme_word", word, TEMPLATES[0][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# New: 11. another_rhyme (same as rhyme_word)
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
        a = "ಹೌದು"
    else:
        non_rhyming_words = [
            w for w in unique_words if w != word1 and not do_rhyme(word1, w)
        ]
        if not non_rhyming_words:
            continue
        word2 = random.choice(non_rhyming_words)
        q = TEMPLATES[2][0].format(word1=word1, word2=word2)
        a = "ಅಲ್ಲ"  # ಪ್ರಾಸವಾಗುತ್ತವೆಯೇ? → quality → ಅಲ್ಲ
    key = ("do_rhyme_yes_no", word1, word2, TEMPLATES[2][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# New: 12. do_rhyme_yes_no (same as do_rhyme_yes_no)
for _ in range(100):
    word1 = random.choice(unique_words)
    if word1 in RHYMING_PAIRS:
        word2 = RHYMING_PAIRS[word1]
        q = TEMPLATES[11][0].format(word1=word1, word2=word2)
        a = "ಹೌದು"
    else:
        non_rhyming_words = [
            w for w in unique_words if w != word1 and not do_rhyme(word1, w)
        ]
        if not non_rhyming_words:
            continue
        word2 = random.choice(non_rhyming_words)
        q = TEMPLATES[11][0].format(word1=word1, word2=word2)
        a = "ಅಲ್ಲ"  # ಪ್ರಾಸಬದ್ಧವೇ? → quality → ಅಲ್ಲ
    key = ("do_rhyme_yes_no", word1, word2, TEMPLATES[11][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 4. word_with_vowel
for w in words_by_first.get("ಅ", [])[:50]:
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

# New: 15. word_ending (same as word_ending)
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

# 6. same_pronunciation: ಹ and ಪ - ಇಲ್ಲ
q = TEMPLATES[5][0]
a = "ಇಲ್ಲ"
key = ("same_pronunciation", TEMPLATES[5][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# New: 16. same_pronunciation_sh_sha: ಶ and ಷ - ಹೌದು
q = TEMPLATES[15][0]
a = "ಹೌದು"
key = ("same_pronunciation_sh_sha", TEMPLATES[15][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# 7. animal_starting with ಬಾ - animals only (no birds); ಬಾವಲಿ (bat) is mammal
ba_animals = animals_by_first.get("ಬಾ", [])
if not ba_animals:
    ba_animals = ["ಬಾವಲಿ"]
for w in ba_animals:
    q = TEMPLATES[6][0]
    a = w
    key = ("animal_starting", "ಬಾ", w, TEMPLATES[6][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# New: 13. fruit_starting
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

# 8. identify_sound: ತ ವರ್ಗ - answer "ತ, ಥ, ದ, ಧ, ನ"
q = TEMPLATES[7][0]
a = ", ".join(VARGAS.get("ತ", []))
key = ("identify_sound", TEMPLATES[7][0])
if key not in seen:
    seen.add(key)
    samples.append((q, a))

# 9. first_sound
for word in unique_words[:150]:
    clusters = get_kannada_grapheme_clusters(word)
    if not clusters:
        continue
    q = TEMPLATES[8][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[8][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# New: 19. first_sound (same as first_sound)
for word in unique_words[:150]:
    clusters = get_kannada_grapheme_clusters(word)
    if not clusters:
        continue
    q = TEMPLATES[18][0].format(word=word)
    a = clusters[0]
    key = ("first_sound", word, TEMPLATES[18][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# 10. word_with_nasal
words_with_n = [w for w in unique_words if "ನ" in w]
for w in (words_with_n or unique_words)[:80]:
    q = TEMPLATES[9][0]
    a = w
    key = ("word_with_nasal", w, TEMPLATES[9][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))


# New: 14 & 20. similar_sound (ಹೋಲುವ ಧ್ವನಿ): same last akshara + similar length (ಪ್ರಾಸ-like)
def get_similar_sound_words(word: str, word_list: list) -> list:
    """Words with same last akshara and same akshara count (phonetically similar length)."""
    clusters = get_kannada_grapheme_clusters(word)
    if not clusters:
        return []
    last = clusters[-1]
    n = len(clusters)
    return [
        w
        for w in word_list
        if w != word
        and (c := get_kannada_grapheme_clusters(w))
        and c[-1] == last
        and len(c) == n
    ]


# New: 14 & 20. similar_sound (skip when no different similar word exists)
for word in unique_words[:100]:
    similar_words = get_similar_sound_words(word, unique_words)
    if not similar_words:
        continue  # avoid answer = question
    q_idx = random.choice([13, 19])  # Templates 14 and 20
    q = TEMPLATES[q_idx][0].format(word=word)
    a = random.choice(similar_words)
    key = ("similar_sound", word, TEMPLATES[q_idx][0])
    if key not in seen:
        seen.add(key)
        samples.append((q, a))

# New: 17. two_words_with_sound
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

# New: 18. verb_starting
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

# Fill to target (deduplicate, no duplicates)
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    tpl_full, ttype = random.choice(TEMPLATES)
    q, a = None, None  # Initialize q and a
    template_text = tpl_full  # Use full template text as template_text

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
        if (
            word1 in RHYMING_PAIRS and random.random() < 0.7
        ):  # Bias towards positive rhyme answers initially
            word2 = RHYMING_PAIRS[word1]
            a = "ಹೌದು"
        else:
            non_rhyming_words = [
                w for w in unique_words if w != word1 and not do_rhyme(word1, w)
            ]
            if not non_rhyming_words:
                q, a = None, None  # Skip if no non-rhyming words found
            else:
                word2 = random.choice(non_rhyming_words)
                a = "ಅಲ್ಲ"  # ಪ್ರಾಸಬದ್ಧವೇ? → quality → ಅಲ್ಲ
        if q is not None and a is not None:  # Check if a valid pair was generated
            q = template_text.format(word1=word1, word2=word2)
    elif ttype == "word_with_vowel":
        lst = words_by_first.get("ಅ", unique_words)
        a = random.choice(lst) if lst else random.choice(unique_words)
        q = template_text
    elif ttype == "word_ending" and words_by_last:
        letter = random.choice(list(words_by_last.keys()))
        a = random.choice(words_by_last[letter])
        q = template_text.format(letter=letter)
    elif ttype == "same_pronunciation":
        q, a = template_text, "ಇಲ್ಲ"
    elif ttype == "same_pronunciation_sh_sha":
        q, a = template_text, "ಹೌದು"
    elif ttype == "animal_starting":
        lst = animals_by_first.get("ಬಾ", [])
        if not lst:
            lst = ["ಬಾವಲಿ"]
        a = random.choice(lst)
        q = template_text
    elif ttype == "identify_sound":
        q, a = template_text, ", ".join(VARGAS.get("ತ", []))
    elif ttype == "first_sound":
        word = random.choice(unique_words)
        clusters = get_kannada_grapheme_clusters(word)
        if not clusters:
            q, a = None, None
        else:
            q = template_text.format(word=word)
            a = clusters[0]
    elif ttype == "word_with_nasal":
        lst = [w for w in unique_words if "ನ" in w] or unique_words
        a = random.choice(lst)
        q = template_text
    elif ttype == "fruit_starting" and FRUITS_STARTING:
        letter = random.choice(list(FRUITS_STARTING.keys()))
        a = random.choice(FRUITS_STARTING[letter])
        q = template_text.format(letter=letter)
    elif ttype == "similar_sound":
        word = random.choice(unique_words)
        similar_words = get_similar_sound_words(word, unique_words)
        if not similar_words:
            q, a = None, None  # skip - avoid answer = question
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

    if q is not None and a is not None and (q, a) not in seen_qa:
        seen_qa.add((q, a))
        samples.append((q, a))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S3 Sound Matching (Kannada): Generated {len(samples)} samples")
