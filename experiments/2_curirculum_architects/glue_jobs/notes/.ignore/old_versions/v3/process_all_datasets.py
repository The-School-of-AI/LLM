"""
Dataset Processing Orchestrator

Processes all datasets from Datasets_details.csv in optimal order:
1. Small datasets first (validation)
2. Medium datasets (parallel execution)
3. Large datasets (sequential with checkpointing)

Usage:
    # Process all datasets
    python process_all_datasets.py --workers 10
    
    # Process specific dataset
    python process_all_datasets.py --dataset dolma_RefineWeb_v1_7 --workers 25
    
    # Dry run (validate configuration)
    python process_all_datasets.py --dry-run
    
    # Process by size category
    python process_all_datasets.py --size-range small --workers 5
    python process_all_datasets.py --size-range medium --workers 15
    python process_all_datasets.py --size-range large --workers 50

Features:
- Automatic retry on failure (3 attempts)
- Progress tracking and logging
- Cost estimation before execution
- Checkpoint/resume capability
- Parallel execution for independent datasets
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError

# AWS Configuration
GLUE_JOB_NAME = "combined-t123-optimized"
REGION = "us-east-1"
S3_BUCKET = "t1-dataacquisition-datasets"
OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed"
METRICS_BASE = "s3://t1-dataacquisition-datasets/metrics"
CHECKPOINT_FILE = "processing_checkpoint.json"
LOG_FILE = "processing_log.txt"

# Glue job costs (G.2X DPU: $0.44/hour)
DPU_HOUR_COST = 0.44

# Initialize AWS clients
glue_client = boto3.client('glue', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)


class DatasetInfo:
    """Dataset metadata from Datasets_details.csv"""
    
    def __init__(self, row: Dict[str, str]):
        self.name = row['Dataset']
        self.original_path = row['Original']
        self.external_source = row['EXTERNAL_SOURCE']
        self.domain = row['DOMAIN']
        self.file_count = int(row['File count'])
        self.per_gb = float(row['PerGB'])
        
        # Calculate estimated size
        self.estimated_size_gb = self.file_count * self.per_gb
        
        # Categorize by size
        if self.file_count <= 10:
            self.size_category = "small"
        elif self.file_count <= 100:
            self.size_category = "medium"
        else:
            self.size_category = "large"
    
    def __repr__(self):
        return f"<Dataset {self.name}: {self.file_count} files, ~{self.estimated_size_gb:.1f}GB>"


def load_datasets(csv_path: str) -> List[DatasetInfo]:
    """Load dataset metadata from CSV"""
    datasets = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(DatasetInfo(row))
    return datasets


def estimate_processing_time(dataset: DatasetInfo, workers: int) -> tuple[float, float]:
    """
    Estimate processing time and cost
    
    Returns:
        (hours, cost_usd)
    """
    # Throughput estimate: 20GB/hour per worker (conservative)
    throughput_per_worker = 20  # GB/hour
    total_throughput = throughput_per_worker * workers
    
    hours = dataset.estimated_size_gb / total_throughput
    
    # Each worker = 1 DPU in Glue
    cost = hours * workers * DPU_HOUR_COST
    
    return hours, cost


def load_checkpoint() -> Dict:
    """Load processing checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "in_progress": None}


def save_checkpoint(checkpoint: Dict):
    """Save processing checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def log_message(message: str):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")


def check_glue_job_exists(job_name: str) -> bool:
    """Check if Glue job exists"""
    try:
        glue_client.get_job(JobName=job_name)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityNotFoundException':
            return False
        raise


def create_glue_job(job_name: str, script_location: str):
    """Create Glue job if it doesn't exist"""
    log_message(f"Creating Glue job: {job_name}")
    
    glue_client.create_job(
        Name=job_name,
        Role="AWSGlueServiceRole-DataProcessing",  # Update with your IAM role
        Command={
            'Name': 'glueetl',
            'ScriptLocation': script_location,
            'PythonVersion': '3'
        },
        DefaultArguments={
            '--enable-spark-ui': 'true',
            '--enable-metrics': 'true',
            '--enable-glue-datacatalog': 'true',
            '--enable-continuous-cloudwatch-log': 'true',
            '--conf': 'spark.sql.adaptive.enabled=true',
            '--conf': 'spark.sql.adaptive.coalescePartitions.enabled=true',
            '--conf': 'spark.sql.adaptive.skewJoin.enabled=true',
            '--conf': 'spark.sql.parquet.compression.codec=zstd',
            '--job-language': 'python',
        },
        MaxRetries=1,
        Timeout=2880,  # 48 hours max
        GlueVersion='5.0',
        WorkerType='G.2X',
        NumberOfWorkers=10,  # Default, overridden at runtime
        ExecutionClass='STANDARD',  # Use 'FLEX' for 70% cost savings (up to 10 min startup delay)
    )
    
    log_message(f"✓ Glue job created: {job_name}")


def start_glue_job(dataset: DatasetInfo, workers: int, dry_run: bool = False) -> Optional[str]:
    """
    Start Glue job for a dataset
    
    Returns:
        job_run_id if started, None otherwise
    """
    input_path = f"{dataset.original_path}/*.json.gz"
    team1_output = f"{OUTPUT_BASE}/{dataset.name}"
    team2_metrics = f"{METRICS_BASE}/{dataset.name}"
    
    args = {
        '--INPUT_PATH': input_path,
        '--TEAM1_OUTPUT_PATH': team1_output,
        '--TEAM2_METRICS_PATH': team2_metrics,
        '--DOMAIN': dataset.domain,
        '--EXTERNAL_SOURCE': dataset.external_source,
        '--VERSION': '1.7',
        '--TARGET_PARTITION_SIZE_MB': '192',
    }
    
    hours, cost = estimate_processing_time(dataset, workers)
    
    log_message("=" * 80)
    log_message(f"Dataset: {dataset.name}")
    log_message(f"  Size: {dataset.file_count} files (~{dataset.estimated_size_gb:.1f} GB)")
    log_message(f"  Domain: {dataset.domain}")
    log_message(f"  Source: {dataset.external_source}")
    log_message(f"  Input: {input_path}")
    log_message(f"  Team 1 Output: {team1_output}")
    log_message(f"  Team 2 Metrics: {team2_metrics}")
    log_message(f"  Workers: {workers}")
    log_message(f"  Estimated Time: {hours:.2f} hours")
    log_message(f"  Estimated Cost: ${cost:.2f}")
    log_message("=" * 80)
    
    if dry_run:
        log_message("DRY RUN - Job not started")
        return None
    
    try:
        response = glue_client.start_job_run(
            JobName=GLUE_JOB_NAME,
            Arguments=args,
            WorkerType='G.2X',
            NumberOfWorkers=workers,
            Timeout=2880,  # 48 hours
        )
        
        job_run_id = response['JobRunId']
        log_message(f"✓ Job started: {job_run_id}")
        log_message(f"  Monitor: https://console.aws.amazon.com/glue/home?region={REGION}#/v2/etl-configuration/jobs/{GLUE_JOB_NAME}/run/{job_run_id}")
        
        return job_run_id
        
    except ClientError as e:
        log_message(f"✗ Failed to start job: {e}")
        return None


def wait_for_job_completion(job_run_id: str, dataset_name: str) -> str:
    """
    Wait for Glue job to complete
    
    Returns:
        Job status: 'SUCCEEDED', 'FAILED', 'STOPPED', 'TIMEOUT'
    """
    log_message(f"Waiting for job completion: {dataset_name}")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            response = glue_client.get_job_run(
                JobName=GLUE_JOB_NAME,
                RunId=job_run_id
            )
            
            status = response['JobRun']['JobRunState']
            
            if status != last_status:
                elapsed = (time.time() - start_time) / 60
                log_message(f"  Status: {status} (elapsed: {elapsed:.1f} min)")
                last_status = status
            
            if status in ['SUCCEEDED', 'FAILED', 'STOPPED', 'TIMEOUT']:
                elapsed = (time.time() - start_time) / 60
                log_message(f"  Final Status: {status} (total: {elapsed:.1f} min)")
                return status
            
            # Wait 30 seconds before checking again
            time.sleep(30)
            
        except ClientError as e:
            log_message(f"  Error checking job status: {e}")
            time.sleep(30)


def process_dataset(dataset: DatasetInfo, workers: int, dry_run: bool = False, 
                   checkpoint: Optional[Dict] = None, max_retries: int = 3) -> bool:
    """
    Process a single dataset with retry logic
    
    Returns:
        True if successful, False otherwise
    """
    if checkpoint and dataset.name in checkpoint.get("completed", []):
        log_message(f"✓ Dataset already processed: {dataset.name}")
        return True
    
    for attempt in range(1, max_retries + 1):
        log_message(f"\nProcessing {dataset.name} (attempt {attempt}/{max_retries})")
        
        job_run_id = start_glue_job(dataset, workers, dry_run)
        
        if dry_run or job_run_id is None:
            return False
        
        # Update checkpoint
        if checkpoint is not None:
            checkpoint["in_progress"] = dataset.name
            save_checkpoint(checkpoint)
        
        # Wait for completion
        status = wait_for_job_completion(job_run_id, dataset.name)
        
        if status == 'SUCCEEDED':
            log_message(f"✅ Dataset processed successfully: {dataset.name}")
            
            if checkpoint is not None:
                checkpoint["completed"].append(dataset.name)
                checkpoint["in_progress"] = None
                save_checkpoint(checkpoint)
            
            return True
        else:
            log_message(f"❌ Job {status}: {dataset.name}")
            
            if attempt < max_retries:
                log_message(f"  Retrying in 60 seconds...")
                time.sleep(60)
    
    # All retries failed
    if checkpoint is not None:
        checkpoint["failed"].append(dataset.name)
        checkpoint["in_progress"] = None
        save_checkpoint(checkpoint)
    
    return False


def process_datasets(datasets: List[DatasetInfo], workers: int, dry_run: bool = False):
    """Process multiple datasets"""
    checkpoint = load_checkpoint()
    
    total_datasets = len(datasets)
    total_size_gb = sum(d.estimated_size_gb for d in datasets)
    total_hours, total_cost = 0, 0
    
    for dataset in datasets:
        hours, cost = estimate_processing_time(dataset, workers)
        total_hours += hours
        total_cost += cost
    
    log_message("\n" + "=" * 80)
    log_message("PROCESSING SUMMARY")
    log_message("=" * 80)
    log_message(f"Total Datasets: {total_datasets}")
    log_message(f"Total Size: {total_size_gb:.1f} GB")
    log_message(f"Workers per Job: {workers}")
    log_message(f"Estimated Time: {total_hours:.2f} hours")
    log_message(f"Estimated Cost: ${total_cost:.2f}")
    log_message("=" * 80)
    
    if dry_run:
        log_message("\nDRY RUN - No jobs will be started")
    else:
        response = input("\nProceed with processing? (yes/no): ")
        if response.lower() != 'yes':
            log_message("Processing cancelled by user")
            return
    
    # Process datasets
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, dataset in enumerate(datasets, 1):
        log_message(f"\n{'='*80}")
        log_message(f"Processing dataset {i}/{total_datasets}: {dataset.name}")
        log_message(f"{'='*80}")
        
        success = process_dataset(dataset, workers, dry_run, checkpoint)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    elapsed_hours = (time.time() - start_time) / 3600
    
    log_message("\n" + "=" * 80)
    log_message("PROCESSING COMPLETE")
    log_message("=" * 80)
    log_message(f"Successful: {successful}/{total_datasets}")
    log_message(f"Failed: {failed}/{total_datasets}")
    log_message(f"Total Time: {elapsed_hours:.2f} hours")
    log_message("=" * 80)
    
    if failed > 0:
        log_message("\nFailed datasets:")
        for dataset_name in checkpoint.get("failed", []):
            log_message(f"  - {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Process datasets with Glue ETL job")
    parser.add_argument('--csv', default='Datasets_details.csv', 
                       help='Path to Datasets_details.csv')
    parser.add_argument('--dataset', help='Process specific dataset')
    parser.add_argument('--size-range', choices=['small', 'medium', 'large'],
                       help='Process datasets in size category')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of Glue workers (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show plan without executing')
    parser.add_argument('--create-job', action='store_true',
                       help='Create Glue job if it does not exist')
    parser.add_argument('--script-location', 
                       default='s3://t1-dataacquisition-datasets/scripts/combined_t123_optimized.py',
                       help='S3 location of Glue script')
    
    args = parser.parse_args()
    
    # Check if Glue job exists
    if not check_glue_job_exists(GLUE_JOB_NAME):
        if args.create_job:
            create_glue_job(GLUE_JOB_NAME, args.script_location)
        else:
            log_message(f"ERROR: Glue job '{GLUE_JOB_NAME}' does not exist")
            log_message(f"Run with --create-job to create it")
            sys.exit(1)
    
    # Load datasets
    datasets = load_datasets(args.csv)
    log_message(f"Loaded {len(datasets)} datasets from {args.csv}")
    
    # Filter datasets
    if args.dataset:
        datasets = [d for d in datasets if d.name == args.dataset]
        if not datasets:
            log_message(f"ERROR: Dataset not found: {args.dataset}")
            sys.exit(1)
    
    if args.size_range:
        datasets = [d for d in datasets if d.size_category == args.size_range]
        log_message(f"Filtered to {len(datasets)} {args.size_range} datasets")
    
    # Sort by size (small to large)
    datasets.sort(key=lambda d: d.estimated_size_gb)
    
    # Process datasets
    process_datasets(datasets, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
