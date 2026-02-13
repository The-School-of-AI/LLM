#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

classified = {
    **{
        w: "animal"
        for w in [
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
            "bird",
            "fish",
            "lion",
            "bear",
            "deer",
            "tiger",
            "horse",
            "mouse",
            "sheep",
            "whale",
            "eagle",
            "shark",
            "zebra",
            "panda",
            "rabbit",
            "monkey",
            "dolphin",
            "penguin",
            "giraffe",
            "elephant",
        ]
        * 10
    },
    **{
        w: "thing"
        for w in [
            "book",
            "pen",
            "desk",
            "chair",
            "table",
            "lamp",
            "phone",
            "watch",
            "clock",
            "laptop",
            "computer",
            "keyboard",
            "window",
            "mirror",
            "bottle",
            "guitar",
            "pencil",
            "door",
            "car",
            "bus",
            "train",
            "plane",
            "house",
            "tree",
            "flower",
            "water",
            "cloud",
            "sun",
            "moon",
            "star",
            "apple",
            "banana",
            "orange",
            "bread",
            "milk",
            "shirt",
            "pants",
            "shoes",
        ]
        * 8
    },
    **{
        w: "person"
        for w in [
            "teacher",
            "doctor",
            "nurse",
            "farmer",
            "artist",
            "writer",
            "singer",
            "dancer",
            "actor",
            "mother",
            "father",
            "sister",
            "brother",
            "uncle",
            "aunt",
            "cousin",
            "friend",
            "child",
            "student",
            "pilot",
            "chef",
            "engineer",
            "scientist",
            "lawyer",
        ]
        * 12
    },
}

templates = [
    "Is '{word}' a person, animal, or thing?",
    "Classify '{word}' as person, animal, or thing",
    "What is '{word}' - a person, animal, or thing?",
    "Tell me if '{word}' is a person, animal, or thing",
    "Is '{word}' a person, an animal, or a thing?",
    "What category is '{word}' - person, animal, or thing?",
]

samples = {}
attempts = 0
words = list(classified.keys())

while len(samples) < 7000 and attempts < 100000:
    attempts += 1
    word = random.choice(words)
    template = random.choice(templates)
    query = template.format(word=word)
    if query not in samples:
        samples[query] = classified[word]

with open("group1_s6.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S6 Classification: Generated {len(samples)} samples")
