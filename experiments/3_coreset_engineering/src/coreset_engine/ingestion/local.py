import glob
import json
import os
from typing import Any, Dict, Iterator, Optional

from .abstract import DataSource

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class JsonlDataSource(DataSource):
    """
    Reads a directory of JSONL files.
    """

    def __init__(self, data_path: str):
        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        else:
            self.files = [data_path]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    try:
                        record = json.loads(line)
                        # Add metadata
                        record["file_path"] = os.path.abspath(file_path)
                        record["file_line"] = line_idx + 1  # 1-indexed

                        # Ensure required keys exist (defaulting if necessary for robustness)
                        if "domain" not in record:
                            record["domain"] = "unknown"
                        if "token_count" not in record:
                            # Simple heuristic fallback if pre-computed count is missing
                            # 1 token ~= 4 chars
                            record["token_count"] = len(record.get("text", "")) // 4

                        yield record
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

    def count(self) -> Optional[int]:
        # Counting JSONL lines without reading is expensive, return None or implement wc -l
        return None


class ParquetDataSource(DataSource):
    """
    Reads a directory of Parquet files using PyArrow.
    """

    def __init__(self, data_path: str):
        if not HAS_PYARROW:
            print(
                "Warning: PyArrow not installed. ParquetDataSource will fail if used."
            )

        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        else:
            self.files = [data_path]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not HAS_PYARROW:
            raise ImportError("pyarrow not installed. Cannot read parquet files.")

        for file_path in self.files:
            try:
                parquet_file = pq.ParquetFile(file_path)
                # Iterate over row groups to avoid loading massive file into RAM
                for batch in parquet_file.iter_batches():
                    df = batch.to_pandas()
                    records = df.to_dict("records")
                    for idx, record in enumerate(records):
                        record["file_path"] = os.path.abspath(file_path)
                        # Parquet doesn't have "lines", but we can track record index within file
                        # Ideally we'd track row_group_offset + idx, but here we simplify
                        record["file_line"] = idx

                        if "domain" not in record:
                            record["domain"] = "unknown"
                        if "token_count" not in record:
                            # Simple heuristic fallback
                            record["token_count"] = len(record.get("text", "")) // 4

                        yield record
            except Exception as e:
                # Log error and skip file in production, for now just print
                print(f"Error reading {file_path}: {e}")
                continue

    def count(self) -> Optional[int]:
        if not HAS_PYARROW:
            return 0
        total = 0
        for f in self.files:
            try:
                meta = pq.read_metadata(f)
                total += meta.num_rows
            except (OSError, IOError, ValueError):
                pass
        return total
