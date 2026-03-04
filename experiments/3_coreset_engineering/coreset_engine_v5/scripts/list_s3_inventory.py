import boto3
import csv
import sys
import os

# Configuration
BUCKET_NAME = "t2-datacurriculum-353"
PREFIX = "processed_dataset/curriculum_data/"
OUTPUT_CSV = "s3_counts_with_size.csv"
AWS_PROFILE = "default"  # Change to your preferred profile if needed

def get_s3_inventory():
    print(f"📊 Starting S3 count and size inventory for bucket: {BUCKET_NAME}")
    print(f"📂 Prefix: {PREFIX}")
    
    try:
        session = boto3.Session(profile_name=AWS_PROFILE)
        s3 = session.client('s3')
    except Exception as e:
        print(f"❌ Error initializing AWS session: {e}")
        sys.exit(1)
    
    # List top-level 'source=' folders
    paginator = s3.get_paginator('list_objects_v2')
    sources = []
    
    result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/')
    if 'CommonPrefixes' in result:
        for prefix in result['CommonPrefixes']:
            folder = prefix['Prefix']
            # Should look like: processed_dataset/curriculum_data/source=cc_head/
            folder_parts = folder.strip('/').split('/')
            if folder_parts:
                folder_name = folder_parts[-1]
                if folder_name.startswith('source='):
                    sources.append(folder)

    inventory = []

    for source_prefix in sources:
        source_name = source_prefix.strip('/').split('/')[-1].replace('source=', '')
        print(f"🔍 Inventorying source: {source_name}")
        
        # List bands inside each source
        bands_prefix = f"{source_prefix}bands/"
        result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bands_prefix, Delimiter='/')
        
        if 'CommonPrefixes' in result:
            for band_prefix_obj in result['CommonPrefixes']:
                band_prefix = band_prefix_obj['Prefix']
                # Looks like: .../source=cc_head/bands/band=B0/
                band_folder = band_prefix.strip('/').split('/')[-1]
                
                if band_folder.startswith('band='):
                    band_number = band_folder.replace('band=B', '')
                    
                    count = 0
                    total_bytes = 0
                    
                    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=band_prefix):
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                # Skip the folder prefix itself
                                if not obj['Key'].endswith('/'):
                                    count += 1
                                    total_bytes += obj['Size']
                    
                    # Calculate size in GB (1 GB = 1024^3 bytes)
                    size_gb = total_bytes / (1024**3)
                    
                    print(f"   - Band {band_number}: {count} files, {size_gb:.4f} GB")
                    inventory.append({
                        'source_name': source_name,
                        'band_number': band_number,
                        'files_count': count,
                        'total_size_gb': f"{size_gb:.4f}"
                    })

    # Write results to CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_name', 'band_number', 'files_count', 'total_size_gb'])
        writer.writeheader()
        writer.writerows(inventory)

    print(f"\n✅ Done! Inventory saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    try:
        import boto3
    except ImportError:
        print("❌ Error: 'boto3' not found. Please run 'pip install boto3'")
        sys.exit(1)
        
    get_s3_inventory()
