import json
from collections import Counter

# Only primary + secondary from curriculum
allowed = {"en", "hi"}

lang_dist = Counter()
total = 0
allowed_count = 0

with open("data/datasets/large_sample_chunks.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        lang = chunk.get("language", "en")
        total += 1

        if lang in allowed:
            allowed_count += 1
            lang_dist[lang] += 1

print(f"Total chunks: {total}")
print(
    f"Chunks in allowed langs (en, hi): {allowed_count} ({100*allowed_count/total:.1f}%)"
)
print("\nLanguage distribution in allowed langs:")
for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
    pct = 100 * count / allowed_count if allowed_count > 0 else 0
    print(f"  {lang}: {count} ({pct:.1f}%)")

# What should the curriculum target be?
total_allowed_tokens = (
    sum(lang_dist.values()) * 100
)  # rough estimate, avg 100 tokens/chunk
en_target_max = int(total_allowed_tokens * 0.92)
hi_target = int(total_allowed_tokens * 0.08)
print("\nWith curriculum targets (en: 92%, hi: 8%):")
print(f"  EN tokens max: {en_target_max}")
print(f"  HI tokens target: {hi_target}")
