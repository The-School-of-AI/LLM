import hashlib
import json
import uuid

import pandas as pd

DATASET_NAME = "MegaScience"
DOMAIN = "education"
VERSION = "2025"
LANGUAGE = "en"

# Dataset # type: ignore
#    question : "qqq" ,
#    answer: "aaa",
#    reference_answer ="rrr"
# Normalized to
# text:"### Instruction:qqq
#      ### Reasoning:aaa
#      ### Reference Answer:rrr"


def format_megascience_text(example):
    """
    Combines question, response (CoT), and reference_answer into a
    structured training string.
    """
    q = example.get("question", "").strip()
    # MegaScience often has 'response' (the reasoning) and 'reference_answer' (the result)
    cot = example.get("answer", "").strip()
    ref = example.get("reference_answer", "").strip()

    # Construction: Instruction -> Thought/Reasoning -> Final Answer
    text_block = f"### Instruction:\n{q}\n\n"
    if cot:
        text_block += f"### Reasoning:\n{cot}\n\n"
    text_block += f"### Reference Answer:\n{ref}"

    return text_block.strip()


def generate_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parquet_path = "/Users/jadhavsa/Downloads/ai_dataset/LLM/MegaScience/part-0.parquet"
    print(f"Loading local parquet file: {parquet_path}")
    ds = pd.read_parquet(parquet_path)
    print(f"Loaded {len(ds)} rows.")
    print("First input row:")
    print(json.dumps(ds.iloc[0].to_dict(), indent=2, ensure_ascii=False))

    def transform_row(example, idx):
        full_text = format_megascience_text(example)
        return {
            "id": str(uuid.uuid4()),
            "hash": generate_hash(full_text),
            "dataset": DATASET_NAME,
            "domain": DOMAIN,
            "source": example.get("source", "MegaScience_Mix"),
            "text": full_text,
            "language": LANGUAGE,
            "metadata": {
                "discipline": example.get("discipline", "general_science"),
                "has_reference": bool(example.get("reference_answer")),
                "original_index": idx,
            },
            "added": None,
            "created": None,
            "version": VERSION,
        }

    # Apply transformation
    records = [transform_row(row, idx) for idx, row in ds.iterrows()]
    print("First output row:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))

    output_path = parquet_path.replace(".parquet", "_processed.parquet")
    print(f"Writing processed data to: {output_path}")
    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"Done. Wrote {len(df_out)} rows.")


if __name__ == "__main__":
    main()
