"""Common utilities for dataset downloading."""

import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

# Base data directory (can be overridden via set_data_dir)
_DATA_DIR = Path(".data")


def get_data_dir() -> Path:
    """Get the current data directory."""
    return _DATA_DIR


def set_data_dir(path: str) -> None:
    """Set the data directory.
    
    Args:
        path: Path to use as the data directory
    """
    global _DATA_DIR
    _DATA_DIR = Path(path)

# Default chunk size (records per file)
DEFAULT_CHUNK_SIZE = 10000  # 10K records per file

# Recommended chunk sizes by scope
SCOPE_CHUNK_SIZES = {
    "test": 10,            # Match test record count
    "validate": 1000,      # Medium chunks for validation
    "pre-prod": 5000,      # Larger chunks for pre-production
    "production": 10000,   # Optimal chunks for production
}

# Supported output formats
OutputFormat = Literal["json", "parquet"]

# Supported storage types
StorageType = Literal["local", "s3"]


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text content.
    
    Args:
        text: Text content to hash
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class S3Storage:
    """Manages S3 bucket operations for dataset storage."""
    
    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        prefix: str = "datasets",
    ):
        """Initialize S3 storage.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region for bucket creation
            prefix: Prefix path within bucket (default: 'datasets')
        """
        self.bucket_name = bucket_name
        self.region = region
        self.prefix = prefix
        self.s3_client = boto3.client("s3", region_name=region)
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Using existing S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                self._create_bucket()
            else:
                raise
    
    def _create_bucket(self):
        """Create S3 bucket with proper configuration."""
        try:
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region}
                )
            print(f"Created S3 bucket: {self.bucket_name} in {self.region}")
            
            # Enable versioning for data safety
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={"Status": "Enabled"}
            )
            print(f"Enabled versioning on bucket: {self.bucket_name}")
        except ClientError as e:
            raise RuntimeError(f"Failed to create bucket: {e}")
    
    def get_key(self, dataset: str, subfolder: str, filename: str, output_format: OutputFormat = "json") -> str:
        """Get S3 object key with proper structure.
        
        Structure: {prefix}/{dataset}/{subfolder}/{format}/{filename}
        Example: datasets/sangraha/hin/parquet/records_00000.parquet
        
        Args:
            dataset: Dataset name
            subfolder: Subfolder name (source/language)
            filename: Output filename
            output_format: Output format ('json' or 'parquet')
        """
        return f"{self.prefix}/{dataset}/{subfolder}/{output_format}/{filename}"
    
    def upload_json(self, key: str, data: list, use_custom_serializer: bool = False):
        """Upload JSON data to S3."""
        if use_custom_serializer:
            content = json.dumps(data, ensure_ascii=False, indent=2, default=json_serializer)
        else:
            content = json.dumps(data, ensure_ascii=False, indent=2)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="application/json"
        )
    
    def upload_parquet(self, key: str, records: list):
        """Upload Parquet data to S3."""
        # Flatten metadata dict to JSON string for Parquet compatibility
        # Also convert datetime objects to ISO strings
        records_for_parquet = []
        for record in records:
            record_copy = record.copy()
            if "metadata" in record_copy and isinstance(record_copy["metadata"], dict):
                record_copy["metadata"] = json.dumps(record_copy["metadata"], ensure_ascii=False, default=json_serializer)
            # Convert datetime fields to ISO strings
            for key_field, value in record_copy.items():
                if isinstance(value, datetime):
                    record_copy[key_field] = value.isoformat()
            records_for_parquet.append(record_copy)
        
        df = pd.DataFrame(records_for_parquet)
        table = pa.Table.from_pandas(df)
        
        # Write to bytes buffer
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        buffer.seek(0)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream"
        )
    
    def download_json(self, key: str) -> Optional[dict]:
        """Download JSON data from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False
    
    def delete_object(self, key: str):
        """Delete object from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
        except ClientError:
            pass
    
    def list_objects(self, prefix: str) -> list:
        """List objects with given prefix."""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append(obj["Key"])
        return objects


def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_download_path(dataset: str, subfolder: str, filename: str, output_format: OutputFormat = "json") -> Path:
    """Get the download path with proper directory structure.
    
    Structure: .data/{dataset}/{subfolder}/{format}/{filename}
    Example: .data/sangraha/hin/parquet/records_00000.parquet
    
    Args:
        dataset: Dataset name ('dolma' or 'sangraha')
        subfolder: Subfolder name (source for dolma, language for sangraha)
        filename: Output filename
        output_format: Output format ('json' or 'parquet')
        
    Returns:
        Path object for the output file
    """
    output_dir = get_data_dir() / dataset / subfolder / output_format
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def get_chunk_filename(base_name: str, chunk_index: int, output_format: OutputFormat = "json") -> str:
    """Generate chunk filename with index.
    
    Args:
        base_name: Base filename (e.g., 'records')
        chunk_index: Chunk index (0-based)
        output_format: Output format ('json' or 'parquet')
        
    Returns:
        Chunk filename (e.g., 'records_00001.json' or 'records_00001.parquet')
    """
    # Remove any extension from base_name
    name = base_name.rsplit(".", 1)[0] if "." in base_name else base_name
    ext = "parquet" if output_format == "parquet" else "json"
    return f"{name}_{chunk_index:05d}.{ext}"


class ChunkedWriter:
    """Manages writing records to chunked files (JSON or Parquet) - local or S3."""
    
    def __init__(
        self,
        dataset: str,
        subfolder: str,
        base_filename: str = "records",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        use_custom_serializer: bool = False,
        resume_from_record: int = 0,
        output_format: OutputFormat = "json",
        storage_type: StorageType = "local",
        s3_storage: Optional[S3Storage] = None,
    ):
        """Initialize chunked writer.
        
        Args:
            dataset: Dataset name
            subfolder: Subfolder name
            base_filename: Base filename for chunks (without extension)
            chunk_size: Number of records per chunk file
            use_custom_serializer: Whether to use custom JSON serializer
            resume_from_record: Number of records already downloaded (for resume)
            output_format: Output format ('json' or 'parquet')
            storage_type: Storage type ('local' or 's3')
            s3_storage: S3Storage instance (required if storage_type='s3')
        """
        self.dataset = dataset
        self.subfolder = subfolder
        self.base_filename = base_filename
        self.chunk_size = chunk_size
        self.use_custom_serializer = use_custom_serializer
        self.output_format = output_format
        self.storage_type = storage_type
        self.s3_storage = s3_storage
        
        if storage_type == "s3" and s3_storage is None:
            raise ValueError("s3_storage is required when storage_type='s3'")
        
        self.current_chunk = []
        self.total_records = 0
        self.files_written = []
        
        # Calculate starting chunk index based on resume point
        if resume_from_record > 0:
            # Start from the next chunk after existing records
            self.chunk_index = resume_from_record // chunk_size
            self.total_records = resume_from_record
            print(f"Writer resuming: starting at chunk {self.chunk_index} (after {resume_from_record} records)")
        else:
            self.chunk_index = 0
    
    def _write_chunk(self):
        """Write current chunk to file."""
        if not self.current_chunk:
            return
        
        chunk_filename = get_chunk_filename(self.base_filename, self.chunk_index, self.output_format)
        
        if self.storage_type == "s3":
            self._write_chunk_s3(chunk_filename)
        else:
            self._write_chunk_local(chunk_filename)
        
        self.files_written.append(chunk_filename)
        self.chunk_index += 1
        self.current_chunk = []
    
    def _write_chunk_local(self, chunk_filename: str):
        """Write chunk to local file system."""
        output_path = get_download_path(self.dataset, self.subfolder, chunk_filename, self.output_format)
        
        if self.output_format == "parquet":
            self._write_parquet_local(output_path)
        else:
            self._write_json_local(output_path)
        
        print(f"Saved {len(self.current_chunk)} records to {output_path}")
    
    def _write_chunk_s3(self, chunk_filename: str):
        """Write chunk to S3."""
        key = self.s3_storage.get_key(self.dataset, self.subfolder, chunk_filename, self.output_format)
        
        if self.output_format == "parquet":
            self.s3_storage.upload_parquet(key, self.current_chunk)
        else:
            self.s3_storage.upload_json(key, self.current_chunk, self.use_custom_serializer)
        
        print(f"Saved {len(self.current_chunk)} records to s3://{self.s3_storage.bucket_name}/{key}")
    
    def _write_json_local(self, output_path: Path):
        """Write chunk as JSON file to local storage."""
        with open(output_path, "w", encoding="utf-8") as f:
            if self.use_custom_serializer:
                json.dump(self.current_chunk, f, ensure_ascii=False, indent=2, default=json_serializer)
            else:
                json.dump(self.current_chunk, f, ensure_ascii=False, indent=2)
    
    def _write_parquet_local(self, output_path: Path):
        """Write chunk as Parquet file to local storage."""
        # Flatten metadata dict to JSON string for Parquet compatibility
        # Also convert datetime objects to ISO strings
        records_for_parquet = []
        for record in self.current_chunk:
            record_copy = record.copy()
            if "metadata" in record_copy and isinstance(record_copy["metadata"], dict):
                record_copy["metadata"] = json.dumps(record_copy["metadata"], ensure_ascii=False, default=json_serializer)
            # Convert datetime fields to ISO strings
            for key, value in record_copy.items():
                if isinstance(value, datetime):
                    record_copy[key] = value.isoformat()
            records_for_parquet.append(record_copy)
        
        df = pd.DataFrame(records_for_parquet)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression="snappy")
    
    def add_record(self, record: dict):
        """Add a record, automatically creating new chunk when needed.
        
        Args:
            record: Record dictionary to add
        """
        self.current_chunk.append(record)
        self.total_records += 1
        
        if len(self.current_chunk) >= self.chunk_size:
            self._write_chunk()
    
    def flush(self):
        """Write any remaining records to file."""
        if self.current_chunk:
            self._write_chunk()
    
    def finalize(self):
        """Finalize writing."""
        self.flush()
        storage_info = f"s3://{self.s3_storage.bucket_name}" if self.storage_type == "s3" else "local"
        print(f"Completed: {self.total_records} total records in {len(self.files_written)} files ({storage_info})")
    
    def get_stats(self) -> dict:
        """Get writer statistics."""
        return {
            "total_records": self.total_records,
            "files_written": len(self.files_written),
            "chunk_size": self.chunk_size,
            "current_chunk_size": len(self.current_chunk),
            "storage_type": self.storage_type,
        }


def save_records(records, output_path, use_custom_serializer=False):
    """Save records to a JSON file.
    
    Args:
        records: List of record dictionaries
        output_path: Path to save the JSON file
        use_custom_serializer: Whether to use custom JSON serializer for datetime
    """
    # Ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        if use_custom_serializer:
            json.dump(records, f, ensure_ascii=False, indent=2, default=json_serializer)
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {output_path}")


def load_records(output_path) -> list:
    """Load existing records from a JSON file.
    
    Args:
        output_path: Path to the JSON file
        
    Returns:
        List of existing records, or empty list if file doesn't exist
    """
    output_path = Path(output_path)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} existing records from {output_path}")
        return records
    return []


def get_progress_path(dataset: str, subfolder: str, output_format: OutputFormat = "json") -> Path:
    """Get the progress file path for tracking download state.
    
    Args:
        dataset: Dataset name ('dolma' or 'sangraha')
        subfolder: Subfolder name (source for dolma, language for sangraha)
        output_format: Output format ('json' or 'parquet')
        
    Returns:
        Path object for the progress file
    """
    return get_download_path(dataset, subfolder, ".progress.json", output_format)


def save_progress(
    dataset: str,
    subfolder: str,
    downloaded_count: int,
    total_count: int,
    output_format: OutputFormat = "json",
    s3_storage: Optional[S3Storage] = None,
):
    """Save download progress to resume later.
    
    Args:
        dataset: Dataset name
        subfolder: Subfolder name
        downloaded_count: Number of records downloaded so far
        total_count: Total number of records to download
        output_format: Output format ('json' or 'parquet')
        s3_storage: Optional S3Storage for S3 persistence
    """
    progress = {
        "downloaded_count": downloaded_count,
        "total_count": total_count,
        "last_updated": datetime.now().isoformat()
    }
    
    if s3_storage:
        key = s3_storage.get_key(dataset, subfolder, ".progress.json", output_format)
        s3_storage.upload_json(key, progress)
    else:
        progress_path = get_progress_path(dataset, subfolder, output_format)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)


def load_progress(
    dataset: str,
    subfolder: str,
    output_format: OutputFormat = "json",
    s3_storage: Optional[S3Storage] = None,
) -> dict:
    """Load download progress for resuming.
    
    Args:
        dataset: Dataset name
        subfolder: Subfolder name
        output_format: Output format ('json' or 'parquet')
        s3_storage: Optional S3Storage for S3 persistence
        
    Returns:
        Progress dict with downloaded_count, or None if no progress file
    """
    if s3_storage:
        key = s3_storage.get_key(dataset, subfolder, ".progress.json", output_format)
        return s3_storage.download_json(key)
    else:
        progress_path = get_progress_path(dataset, subfolder, output_format)
        if progress_path.exists():
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


def clear_progress(
    dataset: str,
    subfolder: str,
    output_format: OutputFormat = "json",
    s3_storage: Optional[S3Storage] = None,
):
    """Clear progress file after successful completion.
    
    Args:
        dataset: Dataset name
        subfolder: Subfolder name
        output_format: Output format ('json' or 'parquet')
        s3_storage: Optional S3Storage for S3 persistence
    """
    if s3_storage:
        key = s3_storage.get_key(dataset, subfolder, ".progress.json", output_format)
        s3_storage.delete_object(key)
        print(f"Cleared S3 progress: s3://{s3_storage.bucket_name}/{key}")
    else:
        progress_path = get_progress_path(dataset, subfolder, output_format)
        if progress_path.exists():
            progress_path.unlink()
            print(f"Cleared progress file: {progress_path}")
