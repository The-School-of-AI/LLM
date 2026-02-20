import json
from collections import Counter
from pathlib import Path

# Load sample chunks
chunks_file = Path("data/datasets/sample_chunks.jsonl")
reasoning_b3 = []
with open(chunks_file) as f:
    for line in f:
        chunk = json.loads(line)
        if chunk.get("band") == "B3" and chunk.get("domain") == "reasoning":
            reasoning_b3.append(chunk)
            if len(reasoning_b3) <= 5:  # Show first 5
                print(
                    f"ID: {chunk['id'][:16]}... Band: {chunk['band']}, Domain: {chunk['domain']}, Lang: {chunk['language']}, Tokens: {chunk['token_count']}"
                )

print(f"\nTotal B3 reasoning chunks: {len(reasoning_b3)}")

# Count by language


langs = Counter(c["language"] for c in reasoning_b3)
total_tokens = sum(c["token_count"] for c in reasoning_b3)
for lang, count in langs.items():
    tokens = sum(c["token_count"] for c in reasoning_b3 if c["language"] == lang)
    pct = 100 * tokens / total_tokens
    print(f"{lang}: {count} chunks, {tokens:,} tokens ({pct:.2f}%)")

# Now check what's in the selected indices for 1B
print("\n--- Checking 1B selected indices ---")
selected_file = Path("output/coresets/1B/selected_indices.jsonl")
if selected_file.exists():
    selected_reasoning = []
    with open(selected_file) as f:
        for line in f:
            entry = json.loads(line)
            chunk_id = entry.get("selected_chunk_id")
            # Find this chunk in our sample_chunks
            with open(chunks_file) as fc:
                for line_chunk in fc:
                    chunk = json.loads(line_chunk)
                    if chunk["id"] == chunk_id:
                        if (
                            chunk.get("band") == "B3"
                            and chunk.get("domain") == "reasoning"
                        ):
                            selected_reasoning.append(chunk)
                        break

    print(f"Selected B3 reasoning chunks in 1B: {len(selected_reasoning)}")
    if selected_reasoning:
        langs_selected = Counter(c["language"] for c in selected_reasoning)
        tokens_selected = sum(c["token_count"] for c in selected_reasoning)
        for lang, count in langs_selected.items():
            tokens = sum(
                c["token_count"] for c in selected_reasoning if c["language"] == lang
            )
            pct = 100 * tokens / tokens_selected
            print(f"{lang}: {count} chunks, {tokens:,} tokens ({pct:.2f}%)")
else:
    print("No 1B manifest yet")
