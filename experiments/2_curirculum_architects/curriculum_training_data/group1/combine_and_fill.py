#!/usr/bin/env python3
"""
Combine all group1 TXT files and fill to 70,000 samples
"""

import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import combine_qa_pairs_to_reach_min_tokens  # noqa: E402

print("Loading all generated files...")
all_samples = {}
seen_queries: set[str] = set()

# Load all existing files (TXT or JSON for backward compatibility)
for i in range(1, 11):
    txt_filename = f"group1_s{i}.txt"
    json_filename = f"group1_s{i}.json"
    loaded = 0

    # Try TXT first
    try:
        with open(txt_filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse "query answer." format
                # Find the last period (end of answer)
                last_period = line.rfind(".")
                if last_period > 0:
                    query = line[:last_period].rstrip()
                    # Find where answer starts (after query)
                    # Simple heuristic: find last ? or . before answer
                    query_end = max(query.rfind("?"), query.rfind("."))
                    if query_end > 0:
                        actual_query = query[: query_end + 1].strip()
                        answer = query[query_end + 1 :].strip() + "."
                    else:
                        # Fallback: split on last space before period
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            actual_query = parts[0]
                            answer = parts[1]
                        else:
                            continue

                    if actual_query not in seen_queries:
                        seen_queries.add(actual_query)
                        all_samples[actual_query] = answer
                        loaded += 1
            print(f"  {txt_filename}: loaded {loaded} samples")
        continue
    except FileNotFoundError:
        pass

    # Try JSON for backward compatibility
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for query, answer in data.items():
                    if query not in seen_queries:
                        seen_queries.add(query)
                        all_samples[query] = answer
                        loaded += 1
            else:
                # List of prompts (old format without answers)
                for query in data:
                    if query not in seen_queries:
                        seen_queries.add(query)
                        # No answer available, skip or generate placeholder
                        pass
            print(f"  {json_filename}: loaded {loaded} samples")
    except FileNotFoundError:
        print(f"  {txt_filename}/{json_filename}: NOT FOUND")

target = 70000
print(f"\nTotal samples loaded: {len(all_samples)}")
print(f"Need to generate: {max(target - len(all_samples), 0)} more samples")


# Expand word pools massively
def generate_words(length):
    """Generate pronounceable 3-letter combinations"""
    consonants = list("bcdfghjklmnpqrstvwxyz")
    vowels = list("aeiou")
    words = []
    for c1 in consonants[:10]:
        for v in vowels:
            for c2 in consonants[:10]:
                words.append(c1 + v + c2)
    return words


# Generate lots of synthetic words
synthetic_words = []
for length in [3, 4, 5]:
    synthetic_words.extend(generate_words(length))

common_words = [
    "cat",
    "dog",
    "rat",
    "bat",
    "bee",
    "cow",
    "pig",
    "fox",
    "owl",
    "ant",
    "hat",
    "mat",
    "sat",
    "pat",
    "bed",
    "red",
    "pen",
    "sun",
    "run",
    "cup",
    "tiger",
    "horse",
    "mouse",
    "sheep",
    "table",
    "chair",
    "phone",
    "apple",
    "bread",
    "water",
    "elephant",
    "giraffe",
    "computer",
]

all_words = common_words + synthetic_words[:500]

# Templates for each type
spell_templates = [
    'What is the spelling of "{w}"?',
    'How do you spell "{w}"?',
    'Spell "{w}".',
    'Write the spelling of "{w}".',
]
count_templates = [
    'How many letters are in "{w}"?',
    'Count the letters in "{w}"',
    'What is the length of word "{w}"?',
    'How many alphabets are there in "{w}"?',
]
last_templates = [
    'What letter does "{w}" end with?',
    'What is the last letter of "{w}"?',
    'Which letter does "{w}" end with?',
]

print("\nFilling remaining samples...")
attempts = 0
max_attempts = 1000000

while len(all_samples) < target and attempts < max_attempts:
    attempts += 1

    # Randomly choose template type
    choice = random.randint(1, 10)

    if choice <= 3:  # Spelling (30%)
        word = random.choice(all_words)
        template = random.choice(spell_templates)
        query = template.format(w=word)
        answer = ", ".join(word.lower())
    elif choice <= 5:  # Letter count (20%)
        word = random.choice(all_words)
        template = random.choice(count_templates)
        query = template.format(w=word)
        answer = str(len(word))
    elif choice <= 7:  # Last letter (20%)
        word = random.choice(all_words)
        template = random.choice(last_templates)
        query = template.format(w=word)
        answer = word[-1].lower()
    elif choice <= 9:  # Letter at position (20%)
        word = random.choice(all_words)
        if len(word) > 0:
            pos = random.randint(1, len(word))
            pos_words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
            pos_word = pos_words.get(pos, f"{pos}th")
            query = f'What is the {pos_word} letter in "{word}"?'
            answer = word[pos - 1].lower()
        else:
            continue
    else:  # Word comparison (10%)
        w1 = random.choice(all_words)
        w2 = random.choice(all_words)
        if w1 != w2:
            query = f"Which word is longer, '{w1}' or '{w2}'?"
            if len(w1) > len(w2):
                answer = w1
            elif len(w2) > len(w1):
                answer = w2
            else:
                answer = "equal"
        else:
            continue

    if query not in seen_queries:
        seen_queries.add(query)
        all_samples[query] = answer

        if len(all_samples) % 5000 == 0:
            print(f"  Progress: {len(all_samples):,} / {target:,}")

print(f"\nFinal total: {len(all_samples):,} samples")

# Combine QA pairs into samples where all questions have answers
# Format: "Q1? A1. Q2? A2. Q3? A3. ..." until reaching 512 tokens per sample
print(f"\n{'=' * 80}")
print("Combining QA pairs into samples (all questions with answers)...")
print("  Target: >= 512 tokens per sample")
qa_pairs_list = list(all_samples.items())
combined_samples = combine_qa_pairs_to_reach_min_tokens(qa_pairs_list, min_tokens=512)
print(f"  Original QA pairs: {len(qa_pairs_list):,}")
print(f"  Combined samples: {len(combined_samples):,}")

# Save combined file as TXT (in curriculum_training_data/output folder)
script_dir = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "group1.txt")
print(f"\nSaving to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for sample in combined_samples:
        f.write(sample + "\n")

print(f"\n✓ Successfully saved {len(combined_samples):,} samples to {output_file}")
print("\nValidation:")
print("  - Target: 70,000")
print(f"  - Generated: {len(all_samples):,}")
print(f"  - Difference: {len(all_samples) - target:,}")
