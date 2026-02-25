# Dataset Download Project

This Python project downloads datasets from Hugging Face and uploads them directly to Amazon S3 without storing them locally, using the "Direct Transfer" method. This is ideal for large datasets where local disk space is limited.

## Requirements

- Python (managed by uv)
- AWS credentials configured (e.g., via AWS CLI or environment variables)
- An S3 bucket

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure your AWS credentials. You can do this by:
   - Running `aws configure`
   - Setting environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

3. Update the `bucket` variable in `sangraha_download.py` with your S3 bucket name.

## Usage

Run the script to download specific parts of the sangraha dataset and upload to S3.

For verified data:
```bash
uv run python sangraha_download.py --category verified
```

For language-specific data (use a 3-character language code, e.g., 'hin' for Hindi, 'eng' for English):
```bash
uv run python sangraha_download.py --category language --lang hin
```

Replace `hin` with the desired 3-character language code as per ISO 639-2 (e.g., `eng` for English, `spa` for Spanish, etc.).

The script will:
- List all files in the specified Hugging Face dataset repository
- Download each file temporarily
- Upload it to S3
- Delete the local copy immediately

## Configuration

Edit `sangraha_download.py` to change:
- `repo_id`: The Hugging Face repository ID
- `bucket`: Your S3 bucket name
- `s3_key` prefix: The key structure in S3

## Dependencies

- `huggingface_hub`: For downloading from Hugging Face
- `boto3`: For uploading to S3
