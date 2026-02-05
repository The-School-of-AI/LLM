# Dataset Validation README

## Overview
This repository provides a lightweight and auditable validation pipeline for large-scale text datasets stored in **Parquet** format.  
The validation focuses on **data correctness and integrity**, not model quality.

---

## What This Validation Covers

The validation process ensures the dataset meets the following requirements:

### 1. Parquet File Integrity
- All files open successfully
- Metadata is readable
- At least one batch can be decoded (corruption check)

### 2. Schema Validation
- Columns match the expected schema (`schema.json`)
- Data types are validated
- Nullable fields are explicitly allowed
- Required columns are present

### 3. UTF-8 Encoding Validation
- String fields are validated for UTF-8 encoding
- Validation performed on a sampled subset of records
- **Status: Passed (sampled)**

### 4. Record Counting
- Total record counts are computed from Parquet metadata
- Optional **language-scoped counting** using a column filter (e.g., `language = English`)

---

## Hugging Face Comparison

**Status:** Not Applicable

---

## Usage

### Recommended (Count-only, Lightweight)
Use this mode for large datasets or local machines:

```bash
python validate_dataset.py \
  --data_path . \
  --schema_json schema.json \
  --sample_rows 0 \
  --out_dir validation_out/english_validation \
  --language_column language \
  --language_value English
```

---

### UTF-8 Sample Validation (Executed)

```bash
python validate_dataset.py \
  --data_path . \
  --schema_json schema.json \
  --sample_rows 2000 \
  --out_dir validation_out/utf8_sample \
  --language_column language \
  --language_value English
```

---

## Outputs

- `file_validation_summary.csv`
- `validation_report.json`

---

## Exit Codes

- `0` — Validation passed
- `1` — One or more files failed validation

---

## Validation Status (Current)

- Dataset language: **English**
- Parquet integrity: ✅ Passed
- Schema validation: ✅ Passed
- UTF-8 validation: ✅ **Passed (sampled)**
- Hugging Face comparison: **Not Applicable**
