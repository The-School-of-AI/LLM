"""
Curriculum Data Glue Job V1.0
===================================================
AWS Glue job for consolidating, transforming, and deduplicating 
curriculum data from multiple sources.

Key Features:
- Dynamic source discovery from S3
- Deterministic exact deduplication across bands
- Multi-band processing for individual sources
- S3 based checkpoint management

Author: Glue Migration
Date: 2026-02-12
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
import yaml
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# =========================================================================
# CONFIGURATION & CONSTANTS
# =========================================================================

config = {
    "s3": {
        "bucket": "t2-datacurriculum-353",
        "base_prefix": "processed_dataset/curriculum_data",
        "output_prefix": "processed_dataset/curriculum_pyspark_output",
        "checkpoint_path": "processed_dataset/checkpoints/curriculum_pyspark",
    },
    "processing": {
        "parallelism": 200,
        "batch_size": 1000,
        "default_bands": [
            "B0",
            "B1",
            "B2",
            "B3",
            "B4",
        ],
    },
    "glue_config": {
        "worker_type": "G.1X",
        "num_workers": 10,
        "max_retries": 0,
        "timeout": 2880,
    },
    "schema": {
        "rename_columns": {"uuid": "chunk_id"},
        "drop_columns": ["id", "text", "hash", "metadata", "assigned_band"],
    },
}

# =========================================================================
# LOGGING SETUP
# =========================================================================


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


logger = setup_glue_logger()


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


# =========================================================================
# HADOOP/SPARK UTILS & CHECKPOINTING
# =========================================================================


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


# =========================================================================
# SPARK DATA PROCESSOR
# =========================================================================


class SparkDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process_source_group(
        self, spark: SparkSession, source_name: str, band_paths: List[str]
    ) -> DataFrame:
        """
        Loads all bands for a single source, transforms them,
        and performs deduplication across the entire consolidated source data.
        """
        logger.info(
            f"🚀 Processing Source Group: {source_name} (Bands: {len(band_paths)})"
        )

        # 1. Load all bands for this source into one DataFrame
        # band_paths is a list of s3://bucket/prefix/source=XYZ/bands/band=B0/ etc.
        df_list = []
        for path in band_paths:
            # Extract band name from path for metadata
            # Example path: s3://bucket/prefix/source=arxiv/bands/band=B0/
            band_name = path.split("band=")[-1].strip("/")

            logger.info(f"  Reading Band: {band_name} from {path}")
            df = spark.read.parquet(path)

            # Add metadata columns early
            df = df.withColumn("band", F.lit(band_name))
            df = df.withColumn("source_url", F.lit(path))
            df_list.append(df)

        # Union all bands
        consolidated_df = df_list[0]
        for df in df_list[1:]:
            consolidated_df = consolidated_df.unionByName(df, allowMissingColumns=True)

        # 2. Transform columns based on config
        transformed_df = self._transform_schema(consolidated_df)

        # 3. Deterministic Exact Deduplication on 'hash' column
        # Requirement: "exact dedup shud happend for eah sources but all bands combine"
        logger.info(f"  Performing deterministic deduplication for {source_name}...")

        # We use a window function to pick the "best" record for a hash.
        # For example, if a document exists in B0 and B1, we might prefer B0.
        window_spec = Window.partitionBy("hash").orderBy(F.col("band"))

        unique_df = (
            transformed_df.withColumn("row_num", F.row_number().over(window_spec))
            .filter(F.col("row_num") == 1)
            .drop("row_num")
        )

        # Optimize parallelism before save
        # Repartition to target number of files (parallelism config)
        num_partitions = self.config["processing"].get("parallelism", 10)
        unique_df = unique_df.repartition(num_partitions)

        return unique_df

    def _transform_schema(self, df: DataFrame) -> DataFrame:
        """Applies renames and column drops defined in config."""
        rename_map = self.config["schema"]["rename_columns"]
        drop_cols = self.config["schema"]["drop_columns"]

        # 1. Handle dynamic band scores (band_p_<BAND> -> band_score)
        # Note: In PySpark, we can use coalesce to find the first non-null band_p column
        # or handle it during the union phase. Since we unioned multiple bands,
        # each row should have its matching band_p_<BAND> col.

        # Create 'band_score' column dynamically
        band_p_cols = [c for c in df.columns if c.startswith("band_p_")]
        if band_p_cols:
            df = df.withColumn(
                "band_score", F.coalesce(*[F.col(c) for c in band_p_cols])
            )

        # 2. Apply renames
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # 3. Drop columns (keep hash for dedup, but we might drop it in final output save)
        # We only drop columns that AREN'T needed for dedup logic.
        cols_to_drop = [c for c in drop_cols if c != "hash" and c in df.columns]
        df = df.drop(*cols_to_drop)

        return df

    def save_output(self, df: DataFrame, output_path: str, source_name: str):
        """Saves the processed DataFrame as JSONL with partitioning."""
        logger.info(f"  Saving deduplicated data for {source_name} to {output_path}")

        # We keep 'source' as a column for partitioning in S3
        # This creates a folder structure like: output/source=arxiv/
        # which is much better for downstream querying (Athena/Glue Catalog).
        df.write.mode("append").partitionBy("source").json(output_path)


# =========================================================================
# MAIN ORCHESTRATION
# =========================================================================


def main():
    # 1. Initialize Glue Context
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "config_path"])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    logger.info("Initializing PySpark Glue Job for Curriculum Data")
    logger.info(f"Job Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Load Config
    args["config_path"]
    global config

    bucket = config["s3"]["bucket"]
    base_prefix = config["s3"]["base_prefix"]
    output_prefix = config["s3"]["output_prefix"]
    checkpoint_path = f"s3://{bucket}/{config['s3']['checkpoint_path']}"
    output_path = f"s3://{bucket}/{output_prefix}"

    processor = SparkDataProcessor(config)
    checkpoint_mgr = CheckpointManager(spark, checkpoint_path)

    # 3. Discover Structure Dynamically
    sources = discover_sources_from_s3(bucket, base_prefix)
    target_bands = config["processing"]["default_bands"]

    # Professional Tip: Set shuffle partitions based on parallelism config
    # This prevents the default 200 partitions if data is small/large
    spark.conf.set(
        "spark.sql.shuffle.partitions", config["processing"].get("parallelism", 200)
    )

    # 4. Orchestrate Processing
    for source in sources:
        if checkpoint_mgr.is_finished(source):
            logger.info(f"Skipping already processed source: {source}")
            continue

        try:
            # Construct paths for all bands for this source
            band_paths = [
                f"s3://{bucket}/{base_prefix}/source={source}/bands/band={band}/"
                for band in target_bands
            ]

            # Process and Dedup
            unique_df = processor.process_source_group(spark, source, band_paths)

            # Save results
            processor.save_output(unique_df, output_path, source)

            # Mark Checkpoint
            checkpoint_mgr.mark_finished(source)
            logger.info(f"✅ Finished source: {source}")

        except Exception as e:
            logger.error(f"❌ Failed processing source {source}: {str(e)}")
            # In a real job, you might want to stop or continue based on config
            continue

    job.commit()


if __name__ == "__main__":
    main()
