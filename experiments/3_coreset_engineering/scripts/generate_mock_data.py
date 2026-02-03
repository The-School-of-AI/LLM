import argparse
import json
import os
import random

DOMAINS = ["web", "code", "math", "wiki"]


def generate_text(difficulty: str) -> str:
    """
    Generates text with varying entropy to trigger DifficultyScorer.
    """
    if difficulty == "easy":
        # Low entropy: repetitive
        base = "simple text is easy to learn "
        return base * random.randint(10, 50)
    elif difficulty == "medium":
        # Medium: natural language-ish
        words = [
            "apple",
            "banana",
            "function",
            "derivative",
            "policy",
            "agent",
            "code",
            "model",
        ]
        return " ".join([random.choice(words) for _ in range(100)])
    else:  # hard
        # High entropy: random chars
        return "".join([chr(random.randint(97, 122)) for _ in range(500)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/mock_data", help="Output directory")
    parser.add_argument("--count", type=int, default=100000, help="Number of records")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    filename = os.path.join(args.output, "dataset_001.jsonl")

    print(f"Generating {args.count} records to {filename}...")

    with open(filename, "w") as f:
        for i in range(args.count):
            # Biased generation to ensure we have enough for all bands
            r = random.random()
            if r < 0.3:
                diff = "easy"  # B0
                modality = "general_text"
            elif r < 0.6:
                diff = "medium"  # B2/B3
                modality = random.choice(["general_text", "code"])
            else:
                diff = "hard"  # B5
                modality = random.choice(["code", "cot_reasoning", "general_text"])

            record = {
                "id": f"doc_{i}",
                "text": generate_text(diff),
                "domain": "web" if modality == "general_text" else modality,
                "modality": modality,
                "timestamp": "2023-01-01",
            }
            f.write(json.dumps(record) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
