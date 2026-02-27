import logging
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
import yaml


def setup_glue_logger():
    """Sets up a logger that works well with AWS Glue/CloudWatch."""
    logger = logging.getLogger("glue_logger")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    # Handle S3 config paths
    if config_path.startswith("s3://"):
        parsed = urlparse(config_path)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return yaml.safe_load(obj["Body"].read())

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_sources_from_s3(bucket: str, base_prefix: str) -> List[str]:
    """Dynamically discovers sources by listing S3 prefixes."""
    s3 = boto3.client("s3")
    # Use delimiter to only get the top-level folders under base_prefix
    if not base_prefix.endswith("/"):
        base_prefix += "/"

    logger = logging.getLogger("glue_logger")
    logger.info(f"Discovering sources in s3://{bucket}/{base_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    sources = []

    for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            # Prefix is like 'processed_dataset/curriculum_data/source=arxiv/'
            folder_name = prefix.get("Prefix").split("/")[-2]
            if folder_name.startswith("source="):
                source_name = folder_name.split("=")[-1]
                sources.append(source_name)

    logger.info(f"Found {len(sources)} sources: {sources}")
    return sources


class CheckpointManager:
    """Simple manager for job checkpoints in S3."""

    def __init__(self, spark, checkpoint_path: str):
        self.spark = spark
        self.checkpoint_path = checkpoint_path

    def is_finished(self, identifier: str) -> bool:
        """Checks if a particular source/unit has been processed."""
        try:
            # In a real Glue scenario, this would check a small metadata file in S3
            path = f"{self.checkpoint_path}/{identifier}.done"
            return self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.spark._jsc.hadoopConfiguration()
            ).exists(self.spark._jvm.org.apache.hadoop.fs.Path(path))
        except Exception:
            return False

    def mark_finished(self, identifier: str):
        """Marks a unit as processed by creating a .done file."""
        path = f"{self.checkpoint_path}/{identifier}.done"
        self.spark.range(1).write.mode("overwrite").text(path)
