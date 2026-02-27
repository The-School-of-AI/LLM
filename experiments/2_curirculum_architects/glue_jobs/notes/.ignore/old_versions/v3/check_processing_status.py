"""
Processing Status Monitor

Checks status of all dataset processing jobs and generates summary reports.

Usage:
    # Check current status
    python check_processing_status.py
    
    # Check specific dataset
    python check_processing_status.py --dataset dolma_RefineWeb_v1_7
    
    # Generate detailed report
    python check_processing_status.py --detailed
    
    # Check rejection statistics
    python check_processing_status.py --rejection-stats
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError

# Configuration
S3_BUCKET = "t1-dataacquisition-datasets"
METRICS_BASE = f"s3://{S3_BUCKET}/metrics"
CHECKPOINT_FILE = "processing_checkpoint.json"

# Initialize AWS clients
s3_client = boto3.client('s3', region_name='us-east-1')
glue_client = boto3.client('glue', region_name='us-east-1')
athena_client = boto3.client('athena', region_name='us-east-1')


def load_checkpoint() -> Dict:
    """Load processing checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "in_progress": None}


def check_s3_dataset_exists(dataset_name: str, prefix: str) -> tuple[bool, int, float]:
    """
    Check if dataset exists in S3 and get file count/size
    
    Returns:
        (exists, file_count, size_gb)
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Extract bucket and key from S3 path
        s3_path = f"{prefix}/{dataset_name}/"
        bucket = S3_BUCKET
        key_prefix = s3_path.replace(f"s3://{bucket}/", "")
        
        total_size = 0
        file_count = 0
        
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        total_size += obj['Size']
                        file_count += 1
        
        size_gb = total_size / (1024 ** 3)
        exists = file_count > 0
        
        return exists, file_count, size_gb
        
    except ClientError as e:
        print(f"Error checking S3: {e}")
        return False, 0, 0.0


def get_rejection_stats(dataset_name: str) -> Optional[Dict]:
    """
    Query metrics Parquet files to get rejection statistics
    Uses PyArrow to read Parquet directly (faster than Athena for spot checks)
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow.fs as fs
        
        s3_path = f"{METRICS_BASE}/{dataset_name}"
        
        # Read Parquet files
        s3_fs = fs.S3FileSystem()
        dataset = pq.ParquetDataset(s3_path.replace("s3://", ""), filesystem=s3_fs)
        
        # Read only rejection columns (column pruning)
        table = dataset.read(columns=['is_rejected', 'rejection_reason'])
        
        # Convert to pandas for aggregation
        df = table.to_pandas()
        
        total = len(df)
        rejected = df['is_rejected'].sum()
        accepted = total - rejected
        
        # Top rejection reasons
        rejection_reasons = df[df['is_rejected']].groupby('rejection_reason').size()
        top_reasons = rejection_reasons.sort_values(ascending=False).head(10).to_dict()
        
        return {
            'total': total,
            'accepted': accepted,
            'rejected': rejected,
            'acceptance_rate': accepted / total if total > 0 else 0,
            'top_rejection_reasons': top_reasons
        }
        
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return None


def print_status_table(datasets_status: List[Dict]):
    """Print formatted status table"""
    
    # Header
    print("\n" + "=" * 120)
    print(f"{'Dataset':<40} {'Status':<15} {'Files':<10} {'Size (GB)':<12} {'Acceptance':<12}")
    print("=" * 120)
    
    # Rows
    for ds in datasets_status:
        name = ds['name'][:39]  # Truncate long names
        status = ds['status']
        files = ds.get('file_count', 0)
        size = ds.get('size_gb', 0.0)
        acceptance = ds.get('acceptance_rate', 0.0)
        
        # Color coding for status
        status_symbol = {
            'completed': '✅',
            'failed': '❌',
            'in_progress': '⏳',
            'not_started': '⏹️'
        }.get(status, '❓')
        
        print(f"{name:<40} {status_symbol} {status:<13} {files:<10} {size:<12.2f} {acceptance:<12.1%}")
    
    print("=" * 120)


def generate_summary_report(datasets_status: List[Dict]):
    """Generate summary statistics"""
    
    total = len(datasets_status)
    completed = sum(1 for ds in datasets_status if ds['status'] == 'completed')
    failed = sum(1 for ds in datasets_status if ds['status'] == 'failed')
    in_progress = sum(1 for ds in datasets_status if ds['status'] == 'in_progress')
    not_started = sum(1 for ds in datasets_status if ds['status'] == 'not_started')
    
    total_size = sum(ds.get('size_gb', 0) for ds in datasets_status if ds['status'] == 'completed')
    total_records = sum(ds.get('total_records', 0) for ds in datasets_status if ds['status'] == 'completed')
    total_accepted = sum(ds.get('accepted_records', 0) for ds in datasets_status if ds['status'] == 'completed')
    total_rejected = sum(ds.get('rejected_records', 0) for ds in datasets_status if ds['status'] == 'completed')
    
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Datasets:       {total}")
    print(f"  ✅ Completed:       {completed} ({completed/total*100:.1f}%)")
    print(f"  ❌ Failed:          {failed} ({failed/total*100:.1f}%)")
    print(f"  ⏳ In Progress:     {in_progress}")
    print(f"  ⏹️  Not Started:     {not_started} ({not_started/total*100:.1f}%)")
    print()
    print(f"Total Data Processed: {total_size:.1f} GB")
    print(f"Total Records:        {total_records:,}")
    print(f"  Accepted:           {total_accepted:,} ({total_accepted/total_records*100:.1f}%)")
    print(f"  Rejected:           {total_rejected:,} ({total_rejected/total_records*100:.1f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Check dataset processing status")
    parser.add_argument('--csv', default='Datasets_details.csv',
                       help='Path to Datasets_details.csv')
    parser.add_argument('--dataset', help='Check specific dataset')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--rejection-stats', action='store_true',
                       help='Show rejection statistics for completed datasets')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Load datasets from CSV
    import csv
    datasets = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row['Dataset'])
    
    # Filter to specific dataset
    if args.dataset:
        if args.dataset not in datasets:
            print(f"ERROR: Dataset not found: {args.dataset}")
            sys.exit(1)
        datasets = [args.dataset]
    
    # Check status of each dataset
    datasets_status = []
    
    print("Checking dataset status...")
    
    for dataset_name in datasets:
        # Determine status
        if dataset_name in checkpoint.get('completed', []):
            status = 'completed'
        elif dataset_name in checkpoint.get('failed', []):
            status = 'failed'
        elif checkpoint.get('in_progress') == dataset_name:
            status = 'in_progress'
        else:
            status = 'not_started'
        
        # Check S3 for output
        team1_exists, team1_files, team1_size = check_s3_dataset_exists(
            dataset_name, f"s3://{S3_BUCKET}/processed"
        )
        team2_exists, team2_files, team2_size = check_s3_dataset_exists(
            dataset_name, f"s3://{S3_BUCKET}/metrics"
        )
        
        ds_info = {
            'name': dataset_name,
            'status': status,
            'team1_exists': team1_exists,
            'team1_file_count': team1_files,
            'team1_size_gb': team1_size,
            'team2_exists': team2_exists,
            'team2_file_count': team2_files,
            'team2_size_gb': team2_size,
            'file_count': team2_files,
            'size_gb': team2_size,
        }
        
        # Get rejection stats if requested
        if args.rejection_stats and status == 'completed' and team2_exists:
            print(f"  Reading metrics for {dataset_name}...")
            rejection_stats = get_rejection_stats(dataset_name)
            if rejection_stats:
                ds_info.update({
                    'total_records': rejection_stats['total'],
                    'accepted_records': rejection_stats['accepted'],
                    'rejected_records': rejection_stats['rejected'],
                    'acceptance_rate': rejection_stats['acceptance_rate'],
                    'top_rejection_reasons': rejection_stats['top_rejection_reasons']
                })
        
        datasets_status.append(ds_info)
    
    # Print status table
    print_status_table(datasets_status)
    
    # Print summary
    generate_summary_report(datasets_status)
    
    # Print detailed rejection stats if requested
    if args.rejection_stats:
        print("\n" + "=" * 80)
        print("REJECTION STATISTICS BY DATASET")
        print("=" * 80)
        
        for ds in datasets_status:
            if ds['status'] == 'completed' and 'top_rejection_reasons' in ds:
                print(f"\n{ds['name']}:")
                print(f"  Total: {ds['total_records']:,}")
                print(f"  Accepted: {ds['accepted_records']:,} ({ds['acceptance_rate']:.1%})")
                print(f"  Rejected: {ds['rejected_records']:,}")
                print(f"  Top rejection reasons:")
                
                for reason, count in list(ds['top_rejection_reasons'].items())[:5]:
                    pct = count / ds['rejected_records'] * 100
                    print(f"    • {reason}: {count:,} ({pct:.1f}%)")
        
        print("=" * 80)
    
    # Print detailed info if requested
    if args.detailed:
        print("\n" + "=" * 80)
        print("DETAILED DATASET INFORMATION")
        print("=" * 80)
        
        for ds in datasets_status:
            print(f"\n{ds['name']}:")
            print(f"  Status: {ds['status']}")
            print(f"  Team 1 (Transformed Data):")
            print(f"    Exists: {ds['team1_exists']}")
            print(f"    Files: {ds['team1_file_count']}")
            print(f"    Size: {ds['team1_size_gb']:.2f} GB")
            print(f"  Team 2 (Metrics):")
            print(f"    Exists: {ds['team2_exists']}")
            print(f"    Files: {ds['team2_file_count']}")
            print(f"    Size: {ds['team2_size_gb']:.2f} GB")
            
            if 'acceptance_rate' in ds:
                print(f"  Quality:")
                print(f"    Records: {ds['total_records']:,}")
                print(f"    Acceptance Rate: {ds['acceptance_rate']:.1%}")
        
        print("=" * 80)


if __name__ == "__main__":
    main()
