# validate_system1.py

import json
import random
from pathlib import Path

print("Building validation dataset...\n")

# Load EXACT benchmark questions from registry
mmlu_benchmark = []
with open('benchmark_registry/mmlu_test.jsonl', 'r') as f:
    for line in f:
        mmlu_benchmark.append(json.loads(line))

print(f"✓ Loaded {len(mmlu_benchmark)} MMLU questions from registry")

# Sample 1,000 contaminated (these WILL be in the index)
contaminated = random.sample(mmlu_benchmark, 1000)

# Generate 9,000 clean samples
clean_texts = [
    "The quarterly earnings report exceeded analyst expectations significantly.",
    "New research indicates promising developments in renewable energy storage.",
    "The legislative committee approved the infrastructure spending package.",
    "Market volatility increased following the central bank's policy announcement.",
    "Scientists discovered a previously unknown species in the deep ocean.",
    "The technology sector led gains in today's trading session.",
    "Urban planning initiatives focus on sustainable transportation solutions.",
    "Educational reforms aim to improve student outcomes across districts.",
    "Healthcare providers implement new telemedicine service offerings.",
    "Consumer confidence indices showed improvement in recent surveys."
]

output = []

# Add contaminated samples
for idx, item in enumerate(contaminated):
    output.append({
        "id": f"contaminated_{idx}",
        "text": item['question'],
        "source": "mmlu",
        "ground_truth": "contaminated"
    })

# Add clean samples
for idx in range(9000):
    output.append({
        "id": f"clean_{idx}",
        "text": random.choice(clean_texts) + f" Additional context {idx}.",
        "source": "synthetic",
        "ground_truth": "clean"
    })

# Shuffle
random.shuffle(output)

# Save
Path('test_data').mkdir(exist_ok=True)
with open('test_data/validation_10k.jsonl', 'w') as f:
    for item in output:
        f.write(json.dumps(item) + '\n')

print(f"✓ Created validation_10k.jsonl")
print(f"  - 1,000 contaminated (from registry)")
print(f"  - 9,000 clean")
print(f"\nExpected detection: ~850-900/1,000 (85-90%)")