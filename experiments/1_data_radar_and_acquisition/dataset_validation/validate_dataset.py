import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


@dataclass
class FileResult:
    file: str
    ok_parquet: bool
    ok_schema: bool
    ok_utf8: bool
    rows: int
    error: Optional[str] = None
    schema_mismatch: Optional[str] = None
    utf8_issues: Optional[str] = None


def parse_schema_json(schema_path: str) -> Dict[str, str]:
    """
    Expected schema JSON format:
    {
      "columns": {
        "colA": "string",
        "colB": "int64",
        "colC": "float32",
        ...
      },
      "required_non_null": ["colA", "colB"]
    }
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    if "columns" not in schema:
        raise ValueError("schema JSON must include 'columns' mapping.")
    schema.setdefault("required_non_null", [])
    return schema


def arrow_type_to_simple(t: pa.DataType) -> str:
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_int8(t):
        return "int8"
    if pa.types.is_int16(t):
        return "int16"
    if pa.types.is_int32(t):
        return "int32"
    if pa.types.is_int64(t):
        return "int64"
    if pa.types.is_uint8(t):
        return "uint8"
    if pa.types.is_uint16(t):
        return "uint16"
    if pa.types.is_uint32(t):
        return "uint32"
    if pa.types.is_uint64(t):
        return "uint64"
    if pa.types.is_float16(t):
        return "float16"
    if pa.types.is_float32(t):
        return "float32"
    if pa.types.is_float64(t):
        return "float64"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_date32(t) or pa.types.is_date64(t):
        return "date"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "binary"
    # fallback to Arrow's string
    return str(t)


def validate_schema(
    table_schema: pa.Schema,
    expected: Dict[str, str],
    required_non_null: List[str],
    nullable_fields: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    nullable_fields = nullable_fields or []
    actual_cols = {name: arrow_type_to_simple(table_schema.field(name).type) for name in table_schema.names}

    # Check required columns exist and have expected type
    mismatches = []
    for col, exp_type in expected.items():
        if col not in actual_cols:
            mismatches.append(f"missing column: {col}")
        else:
            act_type = actual_cols[col]
            if act_type != exp_type:
                if not (col in nullable_fields and act_type == "null"):
                    mismatches.append(f"type mismatch: {col} expected={exp_type} actual={act_type}")

    # Extra columns are usually OK; if you want strict mode, enforce here
    # extras = sorted(set(actual_cols) - set(expected))

    # required_non_null just checks presence here; null-rate check happens during sample scan
    for col in required_non_null:
        if col not in actual_cols:
            mismatches.append(f"required_non_null column missing: {col}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, None


def check_utf8_on_sample(table: pa.Table, sample_rows: int, string_cols: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Ensures string values can be encoded as UTF-8 (catches invalid surrogates).
    Arrow strings are UTF-8 by definition, but this catches cases where data came in malformed.
    """
    if sample_rows <= 0 or not string_cols:
        return True, None

    sample = table.slice(0, min(sample_rows, table.num_rows))
    issues = []
    for col in string_cols:
        arr = sample[col]
        # Convert to Python list (None or str)
        pyvals = arr.to_pylist()
        for i, v in enumerate(pyvals):
            if v is None:
                continue
            try:
                # If v is already str, this should succeed unless it contains invalid surrogates
                v.encode("utf-8")
            except Exception as e:
                issues.append(f"{col}[row={i}] utf8-encode-fail: {repr(v)} ({e})")
                if len(issues) >= 20:
                    break
        if len(issues) >= 20:
            break

    if issues:
        return False, " | ".join(issues[:20])
    return True, None


def parquet_integrity_check(file_path: str) -> Tuple[bool, int, Optional[str], Optional[pa.Schema]]:
    """
    Confirms parquet file opens, metadata reads, and we can read a small batch.
    Returns (ok, rows, error, schema)
    """
    try:
        pf = pq.ParquetFile(file_path)
        # metadata access
        md = pf.metadata
        rows = md.num_rows if md else 0

        # try reading a small batch (forces decode)
        # read first row group / first batch
        it = pf.iter_batches(batch_size=1024)
        _ = next(it, None)

        # get schema
        schema = pf.schema_arrow
        return True, rows, None, schema
    except Exception as e:
        return False, 0, str(e), None


def list_parquet_files(path: str) -> List[str]:
    files = []
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    for root, _, fnames in os.walk(path):
        for f in fnames:
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    return sorted(files)


def hf_count(dataset_name: str, config: Optional[str], split: str) -> int:
    if load_dataset is None:
        raise RuntimeError("datasets library not installed. `pip install datasets`")
    ds = load_dataset(dataset_name, config, split=split)
    return len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Folder containing parquet files (or a single parquet file).")
    ap.add_argument("--schema_json", required=True, help="Path to expected schema JSON.")
    ap.add_argument("--sample_rows", type=int, default=1000, help="Rows per file to sample for non-null + utf8 checks.")
    ap.add_argument("--out_dir", default="validation_out", help="Output directory for reports.")
    ap.add_argument("--hf_dataset", default=None, help="Hugging Face dataset name (e.g., ai4bharat/indiccorp).")
    ap.add_argument("--hf_config", default=None, help="HF config/subset name if required.")
    ap.add_argument("--hf_split", default="train", help="HF split to count (default: train).")
    ap.add_argument("--min_match", type=float, default=0.98, help="Min ratio result_count/hf_count to pass.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    schema_cfg = parse_schema_json(args.schema_json)
    expected_cols = schema_cfg["columns"]
    required_non_null = schema_cfg.get("required_non_null", [])

    parquet_files = list_parquet_files(args.data_path)
    if not parquet_files:
        print(f"No parquet files found under: {args.data_path}")
        sys.exit(2)

    results: List[FileResult] = []
    total_rows = 0

    for fp in parquet_files:
        ok_parquet, rows, err, arrow_schema = parquet_integrity_check(fp)
        ok_schema = False
        ok_utf8 = False
        schema_mismatch = None
        utf8_issues = None

        if ok_parquet and arrow_schema is not None:
            ok_schema, schema_mismatch = validate_schema(
                arrow_schema,
                expected_cols,
                required_non_null,
                schema_cfg.get("nullable_fields", []),
            )

            # Load a small sample table for utf8 + required non-null checks
            try:
                # table = pq.read_table(fp, max_rows=max(args.sample_rows, 1))
                pf = pq.ParquetFile(fp)
                batches = []
                rows_needed = max(args.sample_rows, 1)
                rows_seen = 0

                for batch in pf.iter_batches(batch_size=min(1024, rows_needed)):
                    batches.append(batch)
                    rows_seen += batch.num_rows
                    if rows_seen >= rows_needed:
                        break

                table = pa.Table.from_batches(batches)

                # Required non-null check on sample
                null_issues = []
                for col in required_non_null:
                    if col in table.column_names:
                        arr = table[col]
                        if arr.null_count > 0:
                            null_issues.append(f"{col} nulls_in_sample={arr.null_count}")
                    else:
                        null_issues.append(f"{col} missing_in_file")

                # UTF8 check on string columns
                string_cols = [
                    name for name in table.column_names
                    if pa.types.is_string(table.schema.field(name).type) or pa.types.is_large_string(table.schema.field(name).type)
                ]
                ok_utf8, utf8_issues = check_utf8_on_sample(table, args.sample_rows, string_cols)

                if null_issues:
                    ok_schema = False
                    schema_mismatch = (schema_mismatch + "; " if schema_mismatch else "") + " | ".join(null_issues)

            except Exception as e:
                ok_utf8 = False
                utf8_issues = f"sample_read_failed: {e}"

        total_rows += rows

        results.append(FileResult(
            file=fp,
            ok_parquet=ok_parquet,
            ok_schema=ok_schema,
            ok_utf8=ok_utf8,
            rows=rows,
            error=err,
            schema_mismatch=schema_mismatch,
            utf8_issues=utf8_issues,
        ))

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(os.path.join(args.out_dir, "file_validation_summary.csv"), index=False)

    # Aggregate status
    all_ok = bool(df["ok_parquet"].all() and df["ok_schema"].all() and df["ok_utf8"].all())
    report = {
        "data_path": args.data_path,
        "num_files": len(parquet_files),
        "total_rows": int(total_rows),
        "all_files_ok": all_ok,
        "failed_files": df.loc[~(df.ok_parquet & df.ok_schema & df.ok_utf8), "file"].tolist(),
    }

    # HF count comparison
    if args.hf_dataset:
        expected = hf_count(args.hf_dataset, args.hf_config, args.hf_split)
        ratio = (total_rows / expected) if expected else 0.0
        report["hf_dataset"] = args.hf_dataset
        report["hf_config"] = args.hf_config
        report["hf_split"] = args.hf_split
        report["hf_expected_rows"] = int(expected)
        report["result_rows"] = int(total_rows)
        report["match_ratio"] = float(ratio)
        report["min_match_required"] = float(args.min_match)
        report["match_pass"] = bool(ratio >= args.min_match)

    with open(os.path.join(args.out_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print a quick CLI summary
    print(json.dumps(report, indent=2))
    if not all_ok:
        sys.exit(1)
    if args.hf_dataset and not report.get("match_pass", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
