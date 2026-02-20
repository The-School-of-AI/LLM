import json
from collections import defaultdict
from pathlib import Path

chunks_file = Path("data/datasets/large_sample_chunks.jsonl")
domains_70b = ["reasoning", "agentic", "indic"]
lang_by_domain = defaultdict(lambda: defaultdict(int))
lang_tokens_by_domain = defaultdict(lambda: defaultdict(int))

with open(chunks_file) as f:
    for line in f:
        chunk = json.loads(line)
        domain = chunk.get("domain")
        if domain in domains_70b:
            lang = chunk.get("language")
            tokens = chunk.get("token_count")
            lang_by_domain[domain][lang] += 1
            lang_tokens_by_domain[domain][lang] += tokens

print("70B allowed domains language distribution:")
for domain in domains_70b:
    print(f"\n{domain}:")
    total_tokens = sum(lang_tokens_by_domain[domain].values())
    for lang in sorted(lang_by_domain[domain].keys()):
        count = lang_by_domain[domain][lang]
        tokens = lang_tokens_by_domain[domain][lang]
        pct = 100 * tokens / total_tokens
        print(f"  {lang}: {count} chunks, {tokens:,} tokens ({pct:.2f}%)")
