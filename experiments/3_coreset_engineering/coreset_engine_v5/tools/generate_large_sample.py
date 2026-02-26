import argparse
import json
import random
from pathlib import Path

BANDS = ["B0", "B1", "B2", "B3", "B4", "B5"]
DOMAINS = ["code", "math", "reasoning", "agentic", "indic", "clean_web"]
LANGUAGES = ["en", "hi", "es", "zh", "fr", "ar", "ru", "bn", "pt", "id"]


def generate(out_path: Path, n: int, seed: int = 42):
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            band = random.choices(BANDS, weights=[0.1, 0.15, 0.2, 0.25, 0.15, 0.15])[0]
            domain = random.choice(DOMAINS)
            language = random.choices(
                LANGUAGES,
                weights=[0.5, 0.1, 0.07, 0.07, 0.07, 0.07, 0.05, 0.03, 0.02, 0.02],
            )[0]
            token_count = random.randint(64, 256)  # keep smaller to control size
            token_ids = [random.randint(1, 128_000) for _ in range(token_count)]
            chunk = {
                "chunk_id": f"chunk_{i:07d}",
                "dataset_id": f"ds_{domain}",
                "token_count": token_count,
                "byte_length": token_count * 8,
                "domain": domain,
                "language": language,
                "band": band,
                "source_doc_id": f"doc_{i//10:07d}",
                "source_url": f"http://example.com/{i}",
                "quality_flags": [],
                "sensitive_markers": [],
                "start_offset": 0,
                "token_ids": token_ids,
            }
            f.write(json.dumps(chunk) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate large sample JSONL of chunk metadata"
    )
    parser.add_argument(
        "--n", type=int, default=5000000, help="Number of rows to generate"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/datasets/large_sample_chunks.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    out_path = Path(args.out)
    generate(out_path, args.n, seed=args.seed)
    print(f"Wrote {args.n} rows to {out_path}")


if __name__ == "__main__":
    main()
