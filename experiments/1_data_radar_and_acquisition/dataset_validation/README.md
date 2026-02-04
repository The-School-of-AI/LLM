# Dataset Validation README

This README describes how to validate:
1) Parquet file integrity (valid format, not corrupted)  
2) Schema conformance (columns + types + required fields)  
3) UTF-8 compliance for string fields (sample-based)  
4) Optional: Record count match vs Hugging Face dataset (>= 98%)

---

## Prerequisites

- Python 3.9+
- Recommended: virtual environment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyarrow pandas datasets
```

Verify versions:

```bash
python -c "import pyarrow as pa; import pandas as pd; print('pyarrow', pa.__version__); print('pandas', pd.__version__)"
```

---

## Files

- `validate_dataset.py` : validation script  
- `schema.json` : expected schema definition  
- Output folder (example): `validation_out/`

---

## schema.json (Standard Schema)

Create `schema.json` in the same folder as `validate_dataset.py`.

> Note: Some Parquet writers may store nullable columns as Arrow `null` type when an entire column is null in a file.  
> To support `string/null` fields, we keep expected type as `string` but list them under `nullable_fields`.

```json
{
  "columns": {
    "id": "string",
    "hash": "string",
    "dataset": "string",
    "domain": "string",
    "source": "string",
    "text": "string",
    "language": "string",
    "metadata": "string",
    "added": "string",
    "created": "string",
    "version": "string"
  },
  "required_non_null": ["id", "hash", "dataset", "domain", "text", "language"],
  "nullable_fields": ["source", "added", "created", "version"]
}
```

---

## Run Validation (Parquet + Schema + UTF-8)

```bash
python validate_dataset.py \
  --data_path /path/to/result_parquets \
  --schema_json schema.json \
  --sample_rows 2000 \
  --out_dir validation_out
```

### What it checks
- Parquet integrity (file opens and decodes)
- Schema conformance
- UTF-8 encoding on string fields
- Required non-null fields (sample-based)

---

## Output Artifacts

- `validation_out/file_validation_summary.csv`
- `validation_out/validation_report.json`

These serve as validation evidence.

---

## Optional: Hugging Face Record Count Validation

```bash
python validate_dataset.py \
  --data_path /path/to/result_parquets \
  --schema_json schema.json \
  --sample_rows 2000 \
  --out_dir validation_out \
  --hf_dataset "ORG/DATASET_NAME" \
  --hf_split "train" \
  --min_match 0.98
```

---

## Common Issues

### Parquet nullable columns inferred as null
- Acceptable for fields listed under `nullable_fields`

### metadata stored as string instead of struct
- Allowed if schema specifies `"metadata": "string"`

### pyarrow max_rows error
- Script should use `ParquetFile.iter_batches()` instead of `read_table(max_rows=...)`

---

## Exit Codes

- `0` : Validation passed
- `1` : Validation failed
- `2` : No parquet files found

---

## Validation Checklist

- [ ] All Parquet files valid and readable
- [ ] Schema matches standard
- [ ] UTF-8 validation completed
- [ ] (Optional) HF record count >= 98%
- [ ] Evidence artifacts generated
