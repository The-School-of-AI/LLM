#!/usr/bin/env python3
"""
Generate Statement 2: Letter Position (అక్షర స్థానం) questions - Telugu
Target: 26,000 pairs (13% of 200,000)
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
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)

# Position names for Telugu ordinals
POSITION_NAMES = [
    ("మొదటి", 1),
    ("రెండవ", 2),
    ("మూడవ", 3),
    ("నాల్గవ", 4),
    ("ఐదవ", 5),
    ("ఆరవ", 6),
    ("ఏడవ", 7),
    ("ఎనిమిదవ", 8),
    ("తొమ్మిదవ", 9),
    ("పదవ", 10),
]

# Telugu vowels and consonants
VOWELS = set(chr(c) for c in range(0x0C05, 0x0C15))  # అ through ఔ
CONSONANTS = set(chr(c) for c in range(0x0C15, 0x0C3A))  # క through హ

# Templates with generation type
TEMPLATES = [
    ('"{word}" పదంలో మొదటి అక్షరం ఏమిటి?', "first"),
    ('"{word}" పదంలో చివరి అక్షరం ఏమిటి?', "last"),
    ('"{word}" పదంలో మూడవ అక్షరం ఏమిటి?', "third"),
    ('"{word}" పదంలో రెండవ అక్షరం ఏమిటి?', "second"),
    ('"{word}" లో "{char}" అక్షరం ఏ స్థానంలో ఉంది?', "position_of"),
    ('"{word}" పదంలో మధ్య అక్షరం ఏమిటి?', "middle"),
    ('"{word}" పదంలో నాల్గవ అక్షరం ఏమిటి?', "fourth"),
    ('"{word}" పదంలో "{char}" అక్షరం చివరన ఉందా?', "at_end"),
    ('"{word}" పదం యొక్క ఆరంభ అక్షరం ఏమిటి?', "first"),
    ('"{word}" పదంలో ఐదవ అక్షరం ఉందా?', "fifth_exists"),
    ('"{word}" పదంలో ఐదవ స్థానంలో ఉన్న అక్షరం ఏమిటి?', "fifth"),
    ('"{word}" పదంలో చివరి నుండి రెండవ అక్షరం ఏమిటి?', "second_from_end"),
    ('"{word}" పదంలో నాల్గవ అక్షరం చెప్పండి?', "fourth"),
    ('"{word}" పదంలో మధ్య అక్షరం ఏది?', "middle"),
    ('"{word}" పదంలో ఆరవ అక్షరాన్ని గుర్తించండి?', "sixth"),
    ('"{word}" లో "{char}" అక్షరం ఏ చోట ఉంది?', "position_of"),
    ('"{word}" పదంలో రెండవ మరియు నాల్గవ అక్షరాలు ఏమిటి?', "second_and_fourth"),
    ('"{word}" పదంలో మొదటి అక్షరం స్వరమా లేదా వ్యంజనమా?', "first_vowel_or_consonant"),
    ('"{word}" పదం యొక్క చివరి అక్షరం ఏమిటి?', "last"),
    ('"{word}" లో "{char}" అక్షరం ఎంతవ అక్షరం?', "position_of"),
]

EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70
all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS + list(ALL_WORDS_UNIQUE)
unique_words = list(set(all_words))

samples = []
target_count = 26000
seen = set()


def get_position_name(pos_1based: int) -> str:
    if 1 <= pos_1based <= len(POSITION_NAMES):
        return POSITION_NAMES[pos_1based - 1][0]
    return f"{pos_1based}వ"


for word in unique_words:
    clusters = get_telugu_grapheme_clusters(word)
    n = len(clusters)
    if n == 0:
        continue

    for template, ttype in TEMPLATES:
        if ttype == "first":
            q = template.format(word=word)
            a = clusters[0]
            key = (word, "first")
        elif ttype == "last":
            q = template.format(word=word)
            a = clusters[-1]
            key = (word, "last")
        elif ttype == "second":
            q = template.format(word=word)
            if n >= 2:
                a = clusters[1]
            else:
                a = f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
            key = (word, "second", template)
        elif ttype == "third":
            q = template.format(word=word)
            if n >= 3:
                a = clusters[2]
            else:
                a = f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
            key = (word, "third", template)
        elif ttype == "fourth":
            q = template.format(word=word)
            if n >= 4:
                a = clusters[3]
            else:
                a = f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
            key = (word, "fourth", template)
        elif ttype == "fifth":
            q = template.format(word=word)
            if n >= 5:
                a = clusters[4]
            else:
                a = f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
            key = (word, "fifth", template)
        elif ttype == "sixth":
            q = template.format(word=word)
            if n >= 6:
                a = clusters[5]
            else:
                a = f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
            key = (word, "sixth", template)
        elif ttype == "middle":
            q = template.format(word=word)
            mid = n // 2
            a = clusters[mid]
            key = (word, "middle", template)
        elif ttype == "position_of":
            for c in clusters:
                pos_1 = next((i + 1 for i, x in enumerate(clusters) if x == c), None)
                if pos_1 is None:
                    continue
                q = template.format(word=word, char=c)
                if "ఎంతవ అక్షరం?" in template:
                    a = f"{get_position_name(pos_1)} అక్షరం"
                else:
                    a = get_position_name(pos_1)
                key = (word, "position_of", c, template)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "at_end":
            for c in clusters:
                q = template.format(word=word, char=c)
                a = "అవును" if clusters[-1] == c else "లేదు"
                key = (word, "at_end", c, template)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "fifth_exists":
            q = template.format(word=word)
            if n >= 5:
                a = f"అవును, {clusters[4]}"
            else:
                a = "లేదు"
            key = (word, "fifth_exists", template)
        elif ttype == "second_from_end" and n >= 2:
            q = template.format(word=word)
            a = clusters[-2]
            key = (word, "second_from_end", template)
        elif ttype == "second_and_fourth" and n >= 4:
            q = template.format(word=word)
            a = f"{clusters[1]}, {clusters[3]}"
            key = (word, "second_and_fourth", template)
        elif ttype == "first_vowel_or_consonant":
            first_char = clusters[0]
            q = template.format(word=word)
            if first_char in VOWELS or (
                len(first_char) > 0 and first_char[0] in VOWELS
            ):
                a = "స్వరం"
            elif first_char in CONSONANTS or (
                len(first_char) > 0 and first_char[0] in CONSONANTS
            ):
                a = "వ్యంజనం"
            else:
                continue
            key = (word, "first_vowel_or_consonant", template)
        else:
            continue

        if key and key not in seen:
            seen.add(key)
            samples.append((q, a))

# Track seen lines for dedup in fill loop
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

# Fill to target with random samples
max_attempts = target_count * 10
fill_attempts = 0
while len(samples) < target_count and fill_attempts < max_attempts:
    fill_attempts += 1
    word = random.choice(unique_words)
    clusters = get_telugu_grapheme_clusters(word)
    n = len(clusters)
    if n == 0:
        continue

    template, ttype = random.choice(TEMPLATES)
    q, a = None, None

    if ttype == "first":
        q = template.format(word=word)
        a = clusters[0]
    elif ttype == "last":
        q = template.format(word=word)
        a = clusters[-1]
    elif ttype == "second":
        q = template.format(word=word)
        a = clusters[1] if n >= 2 else f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
    elif ttype == "third":
        q = template.format(word=word)
        a = clusters[2] if n >= 3 else f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
    elif ttype == "fourth":
        q = template.format(word=word)
        a = clusters[3] if n >= 4 else f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
    elif ttype == "fifth":
        q = template.format(word=word)
        a = clusters[4] if n >= 5 else f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
    elif ttype == "sixth":
        q = template.format(word=word)
        a = clusters[5] if n >= 6 else f"లేదు, ఈ పదంలో {n} అక్షరాలు ఉన్నాయి"
    elif ttype == "middle":
        q = template.format(word=word)
        a = clusters[n // 2]
    elif ttype == "position_of":
        c = random.choice(clusters)
        pos_1 = next(i + 1 for i, x in enumerate(clusters) if x == c)
        q = template.format(word=word, char=c)
        a = (
            f"{get_position_name(pos_1)} అక్షరం"
            if "ఎంతవ అక్షరం?" in template
            else get_position_name(pos_1)
        )
    elif ttype == "at_end":
        c = random.choice(clusters)
        q = template.format(word=word, char=c)
        a = "అవును" if clusters[-1] == c else "లేదు"
    elif ttype == "fifth_exists":
        q = template.format(word=word)
        a = f"అవును, {clusters[4]}" if n >= 5 else "లేదు"
    elif ttype == "second_from_end" and n >= 2:
        q = template.format(word=word)
        a = clusters[-2]
    elif ttype == "second_and_fourth" and n >= 4:
        q = template.format(word=word)
        a = f"{clusters[1]}, {clusters[3]}"
    elif ttype == "first_vowel_or_consonant":
        first_char = clusters[0]
        q = template.format(word=word)
        if first_char in VOWELS or (len(first_char) > 0 and first_char[0] in VOWELS):
            a = "స్వరం"
        elif first_char in CONSONANTS or (
            len(first_char) > 0 and first_char[0] in CONSONANTS
        ):
            a = "వ్యంజనం"
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

output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S2 Letter Position (Telugu): Generated {len(samples)} samples")
