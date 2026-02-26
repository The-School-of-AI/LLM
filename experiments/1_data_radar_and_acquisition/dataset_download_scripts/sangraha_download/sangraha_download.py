import argparse
from io import BytesIO

import boto3
import requests
from huggingface_hub import list_repo_files


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face and upload to S3"
    )
    parser.add_argument(
        "--category",
        choices=["verified", "synthetic", "unverified"],
        required=True,
        help="Choose the category: verified, synthetic, or unverified",
    )
    parser.add_argument(
        "--lang", help="Specify the language if category is language (e.g., hindi)"
    )
    args = parser.parse_args()

    # Configuration
    repo_id = "ai4bharat/sangraha"
    bucket = "t1-dataacquisition-datasets/"  # Replace with your S3 bucket name
    repo_type = "dataset"

    # Determine prefix
    if args.category == "verified":
        prefix = "verified/"
    else:
        parser.error("Invalid category")
    if not args.lang:
        parser.error("--lang is required ")
    else:
        prefix = prefix + args.lang + "/"

    print("Starting the download and upload process...")
    print(
        f"Category: {args.category}, Language: {args.lang if args.lang else 'N/A'}, Prefix: {prefix}"
    )

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Get list of files in the repo and filter by prefix
    all_files = list_repo_files(repo_id, repo_type=repo_type)
    files = [f for f in all_files if f.startswith(prefix)]

    if not files:
        print(f"No files found for prefix {prefix}")
        return

    for filename in files:
        print(f"Streaming and uploading {filename}...")
        try:
            # Construct the raw file URL for Hugging Face datasets
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_obj = BytesIO()
            for chunk in response.iter_content(chunk_size=1048576):
                if chunk:
                    file_obj.write(chunk)
            file_obj.seek(0)
            # Upload to S3 from memory
            s3_key = f"huggingface_sangraha/{filename}"
            s3.upload_fileobj(file_obj, bucket, s3_key)
            print(f"Successfully uploaded {filename} to s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
