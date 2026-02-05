# Parquet Schema Validation Tool

## Overview
This tool validates random records from parquet files stored in AWS S3 against a predefined schema. It samples 20-25 records from 5-10 random parquet files and generates a detailed validation report.

## Features
- 🔍 Random sampling from multiple parquet files
- ✅ Schema validation against defined structure
- 📊 Detailed validation report in Markdown format
- 🔐 AWS S3 integration with credentials

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the validation script:
```bash
python validate_parquet_schema.py
```

The script will:
1. Connect to S3 using provided credentials
2. List all parquet files in `s3://t1-dataacquisition-datasets/datasets_prod/sangraha/hin/`
3. Randomly select 5-10 files
4. Sample 20-25 records total
5. Validate each record against the schema
6. Generate `validation_result.md` report

## Expected Schema

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique record identifier |
| hash | string | SHA-256 hash of text content (for deduplication) |
| dataset | string | Source dataset name |
| domain | string | Content domain (web, literature, education) |
| source | string/null | Source identifier (for Dolma) |
| text | string | Main text content |
| language | string | Full language name |
| metadata | dict | Additional dataset-specific fields |
| added | string/null | ISO timestamp when added |
| created | string/null | ISO timestamp of creation |
| version | string/null | Dataset version |

## Output

The tool generates `validation_result.md` containing:
- Validation summary statistics
- Schema definition
- Detailed validation results for each record
- Field-level validation details
- Record data previews
- Issues and errors found

## Configuration

You can modify the sampling parameters in the `main()` function:
- `num_files`: Number of random files to sample (default: 7)
- `total_records`: Total number of records to validate (default: 25)

## Security Note

⚠️ AWS credentials are currently hardcoded in the script. For production use, consider using:
- Environment variables
- AWS credentials file (~/.aws/credentials)
- IAM roles (for EC2/Lambda)
