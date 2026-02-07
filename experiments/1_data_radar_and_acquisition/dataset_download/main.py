import boto3
from huggingface_hub import hf_hub_download, list_repo_files
import os

def main():
    # Configuration
    repo_id = "ai4bharat/sangraha"
    bucket = "t1-dataacquisition-datasets"  # Replace with your S3 bucket name
    repo_type = "dataset"

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Get list of files in the repo
    files = list_repo_files(repo_id, repo_type=repo_type)

    for filename in files:
        print(f"Downloading and uploading {filename}...")
        try:
            # Download the file
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            
            # Upload to S3
            s3_key = f"sangraha/{filename}"  # Adjust the key as needed
            s3.upload_file(local_path, bucket, s3_key)
            
            # Clean up local file
            os.remove(local_path)
            print(f"Successfully uploaded {filename} to s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
