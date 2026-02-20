import json
from collections import Counter

lang_dist = Counter()
allowed_langs = {"en", "hi"}  # Only en and hi are allowed
excluded = {"zh", "ja", "ko", "fr", "de", "es"}

total = 0
allowed = 0

with open("data/datasets/large_sample_chunks.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        lang = chunk.get("language", "en")
        total += 1

        if lang not in excluded:
            allowed += 1
            lang_dist[lang] += 1

print(f"Total chunks: {total}")
print(f"After filtering excluded languages: {allowed} ({100*allowed/total:.1f}%)")
print("\nAllowed language distribution:")
for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
    pct = 100 * count / allowed
    print(f"  {lang}: {count} ({pct:.1f}%)")
