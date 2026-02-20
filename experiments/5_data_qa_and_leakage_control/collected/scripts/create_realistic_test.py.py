# scripts/create_realistic_test.py

import json
import random
from pathlib import Path

print("Building realistic contamination test...\n")

# Load benchmarks
benchmarks = {}
for bench_file in Path('benchmarks').glob('*_test.jsonl'):
    name = bench_file.stem
    with open(bench_file, 'r') as f:
        benchmarks[name] = [json.loads(line) for line in f]
    print(f"✓ Loaded {name}: {len(benchmarks[name])} questions")

output = []

# 1. EXACT MATCHES (200 samples) - Should catch 100%
print("\n1. Adding exact matches...")
exact = random.sample(benchmarks['mmlu_test'], 200)
for idx, item in enumerate(exact):
    output.append({
        "id": f"exact_{idx}",
        "text": item['question'],
        "ground_truth": "contaminated_exact"
    })

# 2. LIGHT PARAPHRASES (200 samples) - MinHash should catch
print("2. Adding light paraphrases...")
paraphrases = random.sample(benchmarks['mmlu_test'], 200)
rephrase_templates = [
    lambda q: q.replace("What is", "What's"),
    lambda q: q.replace("?", " - explain."),
    lambda q: q.replace("the", "a"),
    lambda q: q + " Please elaborate.",
    lambda q: "Question: " + q
]
for idx, item in enumerate(paraphrases):
    template = random.choice(rephrase_templates)
    output.append({
        "id": f"paraphrase_{idx}",
        "text": template(item['question']),
        "ground_truth": "contaminated_paraphrase"
    })

# 3. PARTIAL MATCHES (100 samples) - Half of question
print("3. Adding partial matches...")
partials = random.sample(benchmarks['gsm8k_test'], 100)
for idx, item in enumerate(partials):
    words = item['question'].split()
    half = ' '.join(words[:len(words)//2])
    output.append({
        "id": f"partial_{idx}",
        "text": half + " [truncated for context]",
        "ground_truth": "contaminated_partial"
    })

# 4. SHORT QUESTIONS (100 samples) - <13 words, will slip through n-gram
print("4. Adding short questions...")
all_questions = benchmarks['mmlu_test'] + benchmarks['boolq_test']
short = [q for q in all_questions if len(q['question'].split()) < 13]
short_sample = random.sample(short, min(100, len(short)))
for idx, item in enumerate(short_sample):
    output.append({
        "id": f"short_{idx}",
        "text": item['question'],
        "ground_truth": "contaminated_short"
    })

# 5. CODE CONTAMINATION (50 samples) - HumanEval
print("5. Adding code samples...")
code = random.sample(benchmarks['humaneval_test'], 50)
for idx, item in enumerate(code):
    output.append({
        "id": f"code_{idx}",
        "text": item['question'],
        "ground_truth": "contaminated_code"
    })

# 6. MIXED BENCHMARKS (50 samples) - Multiple benchmarks
print("6. Adding mixed benchmark samples...")
for idx in range(50):
    bench_name = random.choice(list(benchmarks.keys()))
    item = random.choice(benchmarks[bench_name])
    output.append({
        "id": f"mixed_{idx}",
        "text": item['question'],
        "ground_truth": f"contaminated_{bench_name}"
    })

# 7. CLEAN SAMPLES (9,300 samples) - Realistic web text
print("7. Adding clean samples...")
clean_templates = [
    "The company reported strong earnings in Q{} with revenue of ${} million.",
    "Recent studies show that {} percent of users prefer {} over alternatives.",
    "The research team discovered {} new compounds with potential applications in {}.",
    "Market analysts predict {} growth in the {} sector over the next {} years.",
    "The new policy aims to reduce {} by implementing stricter {} regulations.",
    "Scientists at {} University published findings on {} in the journal {}.",
    "The startup raised ${} million in Series {} funding led by {} ventures.",
    "Consumer demand for {} products increased {} percent year over year.",
    "The government announced plans to invest in {} infrastructure projects.",
    "Clinical trials showed {} improvement in patients treated with {}."
]

domains = ["technology", "healthcare", "finance", "education", "energy", "transportation"]
for idx in range(9300):
    template = random.choice(clean_templates)
    
    # Fill template with random values
    if '{}' in template:
        filled = template.format(
            random.randint(1, 4),
            random.randint(100, 999),
            random.randint(10, 90),
            random.choice(domains),
            random.randint(1, 10),
            random.choice(["Stanford", "MIT", "Harvard", "Berkeley"]),
            random.choice(["Nature", "Science", "Cell", "PNAS"])
        )
    else:
        filled = template
    
    output.append({
        "id": f"clean_{idx}",
        "text": filled + f" Additional context and details for sample {idx}.",
        "ground_truth": "clean"
    })

# Shuffle
random.shuffle(output)

# Save
Path('tests').mkdir(exist_ok=True)
with open('tests/realistic_10k.jsonl', 'w') as f:
    for item in output:
        f.write(json.dumps(item) + '\n')

print(f"\n✓ Created tests/realistic_10k.jsonl")
print(f"\nBreakdown:")
print(f"  - 200 exact matches (100% detection expected)")
print(f"  - 200 light paraphrases (80-90% detection expected)")
print(f"  - 100 partial matches (50-70% detection expected)")
print(f"  - 100 short questions (<13 words, 20-30% detection expected)")
print(f"  - 50 code samples (90-100% detection expected)")
print(f"  - 50 mixed benchmarks (90-100% detection expected)")
print(f"  - 9,300 clean samples (0% false positives expected)")
print(f"\nTotal contaminated: 700")
print(f"Expected detection: ~450-550 (65-80%)")