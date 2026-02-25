import hashlib
import json

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

DATASET_NAME = "HuggingFaceTB/smoltalk2"
subset = "Mid"
VERSION = "2025"
LANGUAGE = "en"
SOURCE = "nvidia/Llama-3.1-Nemotron-70B-Reward"

OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"

# Default input path (override via --INPUT_PATH)
# KadamParth/Ncert_dataset from Hugging Face - CSV format
DEFAULT_INPUT_PATH = (
    "s3://t1-dataacquisition-datasets/huggingface_NCERT/NCERT_Dataset.csv"
)


def generate_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_to_text(messages):
    """Converts message list to a training-ready string."""
    formatted_str = ""
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        formatted_str += f"<{role}>\n{content}\n</{role}>\n"
    return formatted_str.strip()


def main():
    from datasets import load_dataset

    parquet_path = (
        "/Users/jadhavsa/Downloads/ai_dataset/LLM/smola/ten_rows_file.parquet"
    )
    print(f"Loading local parquet file: {parquet_path}")
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"Loaded {len(ds)} rows.")
    print("First input row:")
    print(json.dumps(ds[0], indent=2, ensure_ascii=False))

    def transform_row(examples, idx):
        raw_messages = examples["messages"]
        full_text = format_to_text(raw_messages)
        return {
            "id": f"local_{idx}",
            "hash": generate_hash(full_text),
            "dataset": DATASET_NAME,
            "domain": "education",
            "source": SOURCE,
            "text": full_text,
            "language": LANGUAGE,
            "metadata": {"turn_count": len(raw_messages), "has_reasoning": True},
            "added": None,
            "created": None,
            "version": "test",
        }

    common_schema_ds = ds.map(
        transform_row, with_indices=True, remove_columns=["messages"]
    )
    print("First output row:")
    print(json.dumps(common_schema_ds[0], indent=2, ensure_ascii=False))

    # Write output to parquet file
    import pandas as pd

    output_path = parquet_path.replace(".parquet", "_processed.parquet")
    print(f"Writing processed data to: {output_path}")
    df = pd.DataFrame(common_schema_ds)
    df.to_parquet(output_path, index=False)
    print(f"Done. Wrote {len(df)} rows.")


if __name__ == "__main__":
    main()


# 2. Map to common schema
def transform_row(examples, idx):
    # Prepare the main content
    raw_messages = examples["messages"]
    full_text = format_to_text(raw_messages)

    return {
        "id": f"{subset}_{idx}",
        "hash": generate_hash(full_text),
        "dataset": DATASET_NAME,
        "domain": "education",  # Categorized based on r1 reasoning content
        "source": "nvidia/Llama-3.1-Nemotron-70B-Reward",
        "text": full_text,
        "language": LANGUAGE,
        "metadata": {
            "original_subset": subset,
            "turn_count": len(raw_messages),
            "has_reasoning": True,
        },
        "added": None,
        "created": None,  # Not provided in source
        "version": VERSION,
    }
