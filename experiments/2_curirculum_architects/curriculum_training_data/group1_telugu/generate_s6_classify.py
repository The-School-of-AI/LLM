#!/usr/bin/env python3
"""
Generate Statement 6: Classification (వర్గీకరణ) questions - Telugu
Target: 20,000 pairs (10% of 200,000)

13 categories: జంతువు, వ్యక్తి, వృత్తి, ఆహారం, పండు, కూరగాయ, రంగు,
శరీర భాగం, వస్తువు, వాహనం, ప్రకృతి, స్థలం, దుస్తులు

Includes positive AND negative examples:
  - MCQ positive: correct answer among options → category name
  - MCQ negative: correct answer NOT among options → "ఏదీ కాదు, ఇది {correct}"
  - Yes/No positive: "Is X a Y?" where Y is correct → "అవును"
  - Yes/No negative: "Is X a Y?" where Y is wrong → "కాదు"
  - Open-ended: "What category?" → category name

Only words with a known category are used — no fallback/guessing.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402
from group1_telugu.telugu_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402

# All category names
ALL_CATEGORY_NAMES = list(CLASSIFICATION_CATEGORIES.keys())

# Build reverse lookup: word → category (first assignment wins for duplicates)
word_to_category = {}
for category, word_list in CLASSIFICATION_CATEGORIES.items():
    for word in word_list:
        if word not in word_to_category:
            word_to_category[word] = category

# Only use words with a known, correct category
classified_words = list(set(word_to_category.keys()))

# ── Template groups ──────────────────────────────────────────

# MCQ: 3 options (used for both positive and negative)
MCQ_TEMPLATES = [
    '"{word}" ఏ వర్గంలోకి వస్తుంది, {opt1}, {opt2} లేదా {opt3}?',
    '"{word}" అనేది {opt1}, {opt2} లేదా {opt3}?',
    '"{word}" ఏ రకం, {opt1}, {opt2} లేదా {opt3}?',
    '"{word}" ను {opt1}, {opt2} లేదా {opt3} గా వర్గీకరించండి?',
    '{opt1}, {opt2} మరియు {opt3} లో "{word}" ఏది?',
]

# Open-ended
OPEN_TEMPLATES = [
    '"{word}" పదం ఏ వర్గంలోకి వస్తుంది?',
    '"{word}" పదం యొక్క వర్గం ఏమిటి?',
]

# Yes/No
YES_NO_TEMPLATES = [
    '"{word}" ఒక {category} పదమా?',
    '"{word}" {category} వర్గంలోకి వస్తుందా?',
    '"{word}" అనేది ఒక {category} పదమా?',
]

# ── Generator functions ──────────────────────────────────────

# Question types with weights (sum = 100)
QUESTION_TYPES = [
    ("mcq_positive", 45),
    ("mcq_negative", 15),
    ("yesno_positive", 12),
    ("yesno_negative", 18),
    ("open", 10),
]
TYPE_NAMES = [t for t, _ in QUESTION_TYPES]
TYPE_WEIGHTS = [w for _, w in QUESTION_TYPES]


def generate_pair(word, category, qtype):
    """Generate a (question, answer) pair for the given type."""
    if qtype == "mcq_positive":
        template = random.choice(MCQ_TEMPLATES)
        distractors = random.sample([c for c in ALL_CATEGORY_NAMES if c != category], 2)
        options = [category] + distractors
        random.shuffle(options)
        q = template.format(
            word=word, opt1=options[0], opt2=options[1], opt3=options[2]
        )
        a = category

    elif qtype == "mcq_negative":
        template = random.choice(MCQ_TEMPLATES)
        wrong_cats = [c for c in ALL_CATEGORY_NAMES if c != category]
        distractors = random.sample(wrong_cats, min(3, len(wrong_cats)))
        random.shuffle(distractors)
        q = template.format(
            word=word, opt1=distractors[0], opt2=distractors[1], opt3=distractors[2]
        )
        a = f"ఏదీ కాదు, ఇది {category}"

    elif qtype == "yesno_positive":
        template = random.choice(YES_NO_TEMPLATES)
        q = template.format(word=word, category=category)
        a = "అవును"

    elif qtype == "yesno_negative":
        template = random.choice(YES_NO_TEMPLATES)
        wrong_cat = random.choice([c for c in ALL_CATEGORY_NAMES if c != category])
        q = template.format(word=word, category=wrong_cat)
        a = "కాదు"

    elif qtype == "open":
        template = random.choice(OPEN_TEMPLATES)
        q = template.format(word=word)
        a = category

    else:
        return None, None

    return q, a


# ── Generate samples ─────────────────────────────────────────

samples = []
target_count = 20000

# Initial pass: one of each type per word (ensures coverage)
unique_combinations = {}
for word in classified_words:
    category = word_to_category[word]
    for qtype in TYPE_NAMES:
        q, a = generate_pair(word, category, qtype)
        if q is None:
            continue
        key = (word, qtype)
        if key not in unique_combinations:
            unique_combinations[key] = (q, a)

samples = list(unique_combinations.values())

# Track seen lines for dedup
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

# Fill to target with weighted random type selection
max_attempts = target_count * 10
attempts = 0
while len(samples) < target_count and attempts < max_attempts:
    attempts += 1
    word = random.choice(classified_words)
    category = word_to_category[word]
    qtype = random.choices(TYPE_NAMES, weights=TYPE_WEIGHTS, k=1)[0]
    q, a = generate_pair(word, category, qtype)
    if q is None:
        continue
    if (q, a) not in seen_lines:
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

# Count type distribution for reporting
type_counts = {
    "mcq_positive": 0,
    "mcq_negative": 0,
    "yesno_positive": 0,
    "yesno_negative": 0,
    "open": 0,
}
for q, a in samples:
    if a == "అవును":
        type_counts["yesno_positive"] += 1
    elif a == "కాదు":
        type_counts["yesno_negative"] += 1
    elif a.startswith("ఏదీ కాదు"):
        type_counts["mcq_negative"] += 1
    elif "{" not in q and "వర్గం" in q:
        type_counts["open"] += 1
    else:
        type_counts["mcq_positive"] += 1

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S6 Classification (Telugu): Generated {len(samples)} samples")
print(f"  Classified words: {len(classified_words)}")
print(f"  Categories: {len(ALL_CATEGORY_NAMES)} ({', '.join(ALL_CATEGORY_NAMES)})")
print(f"  Type distribution: {type_counts}")
neg_pct = (
    (type_counts["mcq_negative"] + type_counts["yesno_negative"]) / len(samples) * 100
)
print(f"  Negative examples: {neg_pct:.1f}%")
