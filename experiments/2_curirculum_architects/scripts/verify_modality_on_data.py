import io
import os
import sys

import pyarrow.parquet as pq
from curriculum_extractor.metrics.modality import ModalityMetric

# Force UTF-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to path
sys.path.append(os.getcwd())


# Mock config since ModalityMetric might expect it, though it usually just needs it for init
class MockConfig:
    pass


def main():
    parquet_path = "d:/CAPSTONE/LLM/experiments/2_curirculum_architects/data/source/v1_5r2_sample-0002.json.parquet"

    print(f"Reading parquet from: {parquet_path}")
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as e:
        print(f"Error opening parquet: {e}")
        return

    print(f"Parquet metadata: {parquet_file.metadata}")

    records = []
    # Read batches until we have ~5000 records
    for batch in parquet_file.iter_batches(batch_size=1000):
        records.extend(batch.to_pylist())
        if len(records) >= 5000:
            break

    # slice to exact 5000
    records = records[:5000]
    print(f"Loaded {len(records)} records for processing.")

    metric = ModalityMetric(MockConfig())

    stats = {
        "has_code": 0,
        "has_math": 0,
        "has_reasoning": 0,
        "has_agentic": 0,
        "has_research_paper": 0,
    }

    examples = {
        "has_code": [],
        "has_math": [],
        "has_reasoning": [],
        "has_agentic": [],
        "has_research_paper": [],
    }

    print("Processing records...")
    for record in records:
        # content column usually holds text
        text = record.get("text", record.get("content", ""))
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)

        # Custom compute to extract match strings
        res = {}

        # Code
        match = metric.CODE_PATTERN.search(text)
        if match:
            res["has_code"] = match.group(0)

        # Math
        match = metric.MATH_PATTERN.search(text)
        if match:
            res["has_math"] = match.group(0)

        # Reasoning
        match = metric.REASONING_PATTERN.search(text)
        if match:
            res["has_reasoning"] = match.group(0)

        # Agentic
        match = metric.AGENTIC_PATTERN.search(text)
        if match:
            res["has_agentic"] = match.group(0)

        # Research
        match = metric.RE_RESEARCH_PAPER.search(text)
        if match:
            res["has_research_paper"] = match.group(0)

        for key in stats:
            if key in res:
                stats[key] += 1
                if len(examples[key]) < 5:  # Keep 5 examples now
                    # Format: "Match: [XYZ] | Context: ...text..."
                    snippet = text[:200].replace("\n", " ")
                    examples[key].append(f"MATCH: '{res[key]}' | TEXT: {snippet}")

    print("\n" + "=" * 50)
    print(f"RESULTS (N={len(records)})")
    print("=" * 50)

    for key, count in stats.items():
        pct = (count / len(records)) * 100
        print(f"{key}: {count} ({pct:.2f}%)")
        if examples[key]:
            print("  Examples:")
            for i, ex in enumerate(examples[key]):
                print(f"    {i+1}. {ex}...")  # Safe print handled by sys.stdout wrap
        print("-" * 30)


if __name__ == "__main__":
    main()
