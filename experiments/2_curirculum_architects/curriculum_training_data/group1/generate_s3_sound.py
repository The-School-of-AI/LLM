#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

sound_map = {
    "/b/": ["ball", "bat", "book", "box", "boy", "bee", "bed", "big", "bus"],
    "/k/": ["cat", "car", "cup", "can", "cake", "key", "king", "kite", "kit"],
    "/d/": ["dog", "door", "desk", "day", "duck", "dance", "dark", "deep", "deer"],
    "/f/": ["fish", "fox", "fan", "fun", "food", "phone", "photo"],
    "/l/": ["lion", "lamp", "lake", "leaf", "leg", "light", "like", "lock"],
    "/m/": ["mouse", "moon", "milk", "man", "map", "mat", "meat", "meet"],
}

templates = [
    "Which word starts with the sound {sound}, '{w1}' or '{w2}'?",
    "Pick the word that begins with sound {sound}: '{w1}' or '{w2}'",
]

samples = {}
attempts = 0
sounds = list(sound_map.keys())

while len(samples) < 7000 and attempts < 100000:
    attempts += 1
    sound = random.choice(sounds)
    other_sound = random.choice([s for s in sounds if s != sound])
    w1 = random.choice(sound_map[sound])
    w2 = random.choice(sound_map[other_sound])
    if random.choice([True, False]):
        w1, w2 = w2, w1
        answer = w2
    else:
        answer = w1
    template = random.choice(templates)
    query = template.format(sound=sound, w1=w1, w2=w2)
    if query not in samples:
        samples[query] = answer

with open("group1_s3.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S3 Sound Matching: Generated {len(samples)} samples")
