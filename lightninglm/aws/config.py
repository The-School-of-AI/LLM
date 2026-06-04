"""
AWS Configuration for S3 Checkpoint Management.

This module contains all AWS-related configuration and settings
for uploading checkpoints to S3.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class S3Config:
    """Configuration for S3 checkpoint storage."""

    # S3 Bucket Configuration
    bucket_name: str
    s3_prefix: str
    region: str = "us-east-1"

    # Local Storage Configuration
    local_checkpoint_dir: str = "./checkpoints"

    # Upload Configuration
    max_retries: int = 3
    retry_backoff_base: int = 2  # Exponential backoff base in seconds
    upload_timeout: int = 3600  # Timeout for uploads in seconds

    # Checkpoint Management
    keep_last_n_checkpoints: int = 3  # Number of local checkpoints to keep
    cleanup_after_upload: bool = (
        False  # Whether to delete local checkpoints after upload
    )

    # AWS Credentials (optional - uses boto3 default credential chain if not specified)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # Advanced S3 Configuration
    use_multipart_upload: bool = True
    multipart_threshold: int = 256 * 1024 * 1024  # 256MB
    multipart_chunksize: int = (
        256 * 1024 * 1024
    )  # 256MB — fewer, larger parts = less overhead
    max_concurrency: int = 40  # Max concurrent multipart parts per file
    max_file_parallelism: int = 8  # Max files uploaded in parallel

    # Logging Configuration
    verbose: bool = True
    log_upload_progress: bool = True

    @classmethod
    def from_env(cls, **kwargs) -> "S3Config":
        """
        Create S3Config from environment variables.

        Environment variables:
            S3_BUCKET_NAME: S3 bucket name
            S3_PREFIX: S3 prefix/folder path
            S3_REGION: AWS region (default: us-east-1)
            LOCAL_CHECKPOINT_DIR: Local checkpoint directory
            AWS_ACCESS_KEY_ID: AWS access key (optional)
            AWS_SECRET_ACCESS_KEY: AWS secret key (optional)
            AWS_SESSION_TOKEN: AWS session token (optional)
            KEEP_LAST_N_CHECKPOINTS: Number of checkpoints to keep locally

        Args:
            **kwargs: Override specific config values

        Returns:
            S3Config instance

        Example:
            >>> config = S3Config.from_env(bucket_name="my-bucket")
        """
        config_dict = {
            "bucket_name": os.getenv("S3_BUCKET_NAME", ""),
            "s3_prefix": os.getenv("S3_PREFIX", "training/checkpoints"),
            "region": os.getenv("S3_REGION", "us-east-1"),
            "local_checkpoint_dir": os.getenv("LOCAL_CHECKPOINT_DIR", "./checkpoints"),
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
            "keep_last_n_checkpoints": int(os.getenv("KEEP_LAST_N_CHECKPOINTS", "3")),
        }

        # Override with provided kwargs
        config_dict.update(kwargs)

        # Remove None values for optional fields
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if v is not None
            or k in ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
        }

        return cls(**config_dict)

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.bucket_name:
            raise ValueError("bucket_name is required")

        if not self.s3_prefix:
            raise ValueError("s3_prefix is required")

        if self.keep_last_n_checkpoints < 1:
            raise ValueError("keep_last_n_checkpoints must be at least 1")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.multipart_threshold < 5 * 1024 * 1024:
            raise ValueError(
                "multipart_threshold must be at least 5MB (AWS requirement)"
            )

    def get_boto3_config(self) -> dict:
        """
        Get boto3 client configuration.

        Returns:
            Dictionary with boto3 client configuration
        """
        from botocore.config import Config

        config = {
            "region_name": self.region,
        }

        # Add credentials if provided
        if self.aws_access_key_id:
            config["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            config["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            config["aws_session_token"] = self.aws_session_token

        # Add advanced configuration
        if self.use_multipart_upload:
            config["config"] = Config(
                max_pool_connections=self.max_concurrency,
                retries={"max_attempts": self.max_retries, "mode": "adaptive"},
            )

        return config

    def get_transfer_config(self):
        """
        Get boto3 TransferConfig for uploads — always full speed.

        Returns:
            boto3.s3.transfer.TransferConfig instance
        """
        from boto3.s3.transfer import TransferConfig

        return TransferConfig(
            multipart_threshold=self.multipart_threshold,
            multipart_chunksize=self.multipart_chunksize,
            max_concurrency=self.max_concurrency,
            use_threads=True,
            # No max_bandwidth — always full speed
        )

    def __repr__(self) -> str:
        """String representation of config (hides sensitive data)."""
        return (
            f"S3Config(\n"
            f"  bucket=s3://{self.bucket_name}/{self.s3_prefix}\n"
            f"  region={self.region}\n"
            f"  local_dir={self.local_checkpoint_dir}\n"
            f"  keep_checkpoints={self.keep_last_n_checkpoints}\n"
            f"  max_retries={self.max_retries}\n"
            f"  credentials={'configured' if self.aws_access_key_id else 'from_env'}\n"
            f")"
        )


# Default configuration presets
DEFAULT_CONFIGS = {
    "development": S3Config(
        bucket_name="dev-training-checkpoints",
        s3_prefix="experiments/dev",
        region="us-east-1",
        keep_last_n_checkpoints=2,
        verbose=True,
    ),
    "production": S3Config(
        bucket_name="prod-training-checkpoints",
        s3_prefix="experiments/prod",
        region="us-east-1",
        keep_last_n_checkpoints=5,
        cleanup_after_upload=False,
        verbose=False,
    ),
    "test": S3Config(
        bucket_name="test-training-checkpoints",
        s3_prefix="experiments/test",
        region="us-east-1",
        keep_last_n_checkpoints=1,
        verbose=True,
    ),
}


def get_default_config(preset: str = "development") -> S3Config:
    """
    Get a default configuration preset.

    Args:
        preset: Configuration preset name ("development", "production", "test")

    Returns:
        S3Config instance

    Example:
        >>> config = get_default_config("production")
    """
    if preset not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. Available: {list(DEFAULT_CONFIGS.keys())}"
        )

    return DEFAULT_CONFIGS[preset]
