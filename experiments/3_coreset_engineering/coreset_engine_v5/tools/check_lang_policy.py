import json
from pathlib import Path

from src.curriculum.loader import CurriculumLoader

# Load curriculum
loader = CurriculumLoader("config/curriculum.yaml")
ok, errors = loader.load()

# Check language policy
lang_policy = loader.language_policy
print(f"Primary languages: {lang_policy.primary_languages}")
print(f"Secondary languages: {lang_policy.secondary_languages}")
print(f"Explicitly excluded: {lang_policy.explicitly_excluded}")
print()

# Check what the curriculum raw_curriculum says
secondary_raw = loader.raw_curriculum.get("languages", {}).get("secondary", [])
print("Secondary languages (raw from YAML):")
for spec in secondary_raw:
    print(f"  {spec}")
print()

# Now simulate what happens in 70B
# Load some 70B eligible chunks
chunks_file = Path("data/datasets/large_sample_chunks.jsonl")
b5_chunks = {"en": [], "hi": [], "other": []}

with open(chunks_file) as f:
    count = 0
    for line in f:
        if count > 10000:
            break
        chunk = json.loads(line)
        band = chunk.get("band")
        domain = chunk.get("domain")
        lang = chunk.get("language")

        # 70B allowed: B5 with domains [reasoning, agentic, indic]
        if band == "B5" and domain in ["reasoning", "agentic", "indic"]:
            if lang == "en":
                b5_chunks["en"].append(chunk)
            elif lang == "hi":
                b5_chunks["hi"].append(chunk)
            else:
                b5_chunks["other"].append(chunk)
            count += 1

print("Sample B5 (reasoning/agentic/indic) chunks found:")
for lang, chunks in b5_chunks.items():
    total_tokens = sum(c.get("token_count", 0) for c in chunks)
    print(f"  {lang}: {len(chunks)} chunks, {total_tokens:,} tokens")
