import boto3
from huggingface_hub import hf_hub_download, list_repo_files
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face and upload to S3")
    parser.add_argument('--category', choices=['verified', 'synthetic','unverified'], required=True, help='Choose the category: verified, synthetic, or unverified')
    parser.add_argument('--lang', help='Specify the language if category is language (e.g., hindi)')
    args = parser.parse_args()

    # Configuration
    repo_id = "ai4bharat/sangraha"
    bucket = "t1-dataacquisition-datasets/"  # Replace with your S3 bucket name
    repo_type = "dataset"

    # Determine prefix
    if args.category == 'verified':
        prefix = 'verified/'
    else:
        parser.error("Invalid category")
    if not args.lang:
            parser.error("--lang is required ")
    else:
        prefix = prefix+ args.lang + '/'
    
    print ("Starting the download and upload process...")
    print (f"Category: {args.category}, Language: {args.lang if args.lang else 'N/A'}, Prefix: {prefix}")

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Get list of files in the repo and filter by prefix
    all_files = list_repo_files(repo_id, repo_type=repo_type)
    files = [f for f in all_files if f.startswith(prefix)]

    if not files:
        print(f"No files found for prefix {prefix}")
        return

    for filename in files:
        print(f"Downloading and uploading {filename}...")
        try:
            # Download the file
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            
            # Upload to S3
            s3_key = f"huggingface_sangraha/{args.lang}/{filename}"  # Adjust the key as needed
            s3.upload_file(local_path, bucket, s3_key)
            
            # Clean up local file
            os.remove(local_path)
            print(f"Successfully uploaded {filename} to s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
