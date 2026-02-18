"""
T3 Final Processing - EMR Serverless Version
============================================
Consolidates, transforms, and deduplicates curriculum data from multiple sources.
Converted from Glue T3FinalProcessing.py - EXACT LOGIC PRESERVED.

Key Features:
- Dynamic source discovery from S3
- Deterministic exact deduplication across bands
- Multi-band processing for individual sources
- S3 based checkpoint management

Author: EMR Migration
Date: 2026-02-12
"""

import sys
import argparse
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

import logging
import boto3
from typing import Dict, Any, List
from datetime import datetime

# =========================================================================
# CONFIGURATION & CONSTANTS
# =========================================================================

DEFAULT_CONFIG = {
    "s3": {
        "bucket": "t2-datacurriculum-353",
        "base_prefix": "processed_dataset/curriculum_data",
        "output_prefix": "processed_dataset/curriculum_pyspark_output",
        "checkpoint_path": "processed_dataset/checkpoints/curriculum_pyspark",
    },
    "processing": {
        "parallelism": 200,
        "default_bands": ["B0", "B1", "B2", "B3", "B4"],
    },
    "schema": {
        "rename_columns": {"uuid": "chunk_id"},
        "drop_columns": ["id", "text", "hash", "metadata", "assigned_band"],
    },
}

# =========================================================================
# LOGGING SETUP
# =========================================================================


def setup_logger():
    logger = logging.getLogger("t3_emr_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def discover_sources_from_s3(bucket: str, base_prefix: str) -> List[str]:
    """Dynamically discovers sources by listing S3 prefixes."""
    s3 = boto3.client("s3")
    if not base_prefix.endswith("/"):
        base_prefix += "/"

    logger.info(f"Discovering sources in s3://{bucket}/{base_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    sources = []

    for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            folder_name = prefix.get("Prefix").split("/")[-2]
            if folder_name.startswith("source="):
                source_name = folder_name.split("=")[-1]
                sources.append(source_name)

    logger.info(f"Found {len(sources)} sources: {sources}")
    return sources


from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast


def generate_distribution_stats(df, stats_output_path):
    """
    Lightweight aggregation after dedup + chunking.
    Uses existing word_count and token_count_estimate.
    """

    print("Starting distribution stats aggregation...")

    required_cols = [
        "source",
        "band",
        "domain",
        "language",
        "word_count",
        "token_count_estimate",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column for stats: {col}")

    # Filter null critical fields
    df_stats = df.filter(
        F.col("band").isNotNull()
        & F.col("domain").isNotNull()
        & F.col("language").isNotNull()
    )

    # Select minimal columns (CRITICAL for shuffle efficiency)
    df_stats = df_stats.select(
        "source",
        "band",
        "domain",
        "language",
        F.col("word_count"),
        F.col("token_count_estimate").alias("token_count"),
    )

    # Aggregate
    agg_df = df_stats.groupBy("source", "band", "domain", "language").agg(
        F.count("*").alias("doc_count"),
        F.sum("token_count").alias("total_tokens"),
        F.sum("word_count").alias("total_words"),
    )

    # Derived metrics
    agg_df = agg_df.withColumn(
        "avg_tokens_per_doc", F.col("total_tokens") / F.col("doc_count")
    ).withColumn("avg_words_per_doc", F.col("total_words") / F.col("doc_count"))

    # Percent per source
    source_totals = agg_df.groupBy("source").agg(
        F.sum("total_tokens").alias("source_total_tokens")
    )

    final_stats_df = (
        agg_df.join(broadcast(source_totals), "source")
        .withColumn(
            "pct_of_source_tokens", F.col("total_tokens") / F.col("source_total_tokens")
        )
        .drop("source_total_tokens")
    )

    print("Writing distribution CSV...")

    (
        final_stats_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(stats_output_path)
    )

    print("Distribution stats written successfully.")


# =========================================================================
# CHECKPOINT MANAGER (S3-based for EMR Serverless)
# =========================================================================


class CheckpointManager:
    """S3-based checkpoint manager (no Hadoop dependency)."""

    def __init__(self, bucket: str, checkpoint_path: str):
        self.bucket = bucket
        self.base_path = checkpoint_path.rstrip("/")
        self.s3 = boto3.client("s3")

    def is_finished(self, identifier: str) -> bool:
        """Checks if a source has been processed."""
        key = f"{self.base_path}/{identifier}.done"
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

    def mark_finished(self, identifier: str):
        """Marks a source as processed by creating a .done file."""
        key = f"{self.base_path}/{identifier}.done"
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=b"", ContentType="text/plain"
        )


# =========================================================================
# SPARK DATA PROCESSOR
# =========================================================================


class SparkDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process_source_group(
        self, spark: SparkSession, source_name: str, band_paths: List[str]
    ) -> DataFrame:
        """Loads all bands for a source, transforms, and deduplicates."""
        logger.info(
            f"Processing Source Group: {source_name} (Bands: {len(band_paths)})"
        )

        df_list = []
        for path in band_paths:
            band_name = path.split("band=")[-1].strip("/")
            logger.info(f"  Reading Band: {band_name} from {path}")

            try:
                df = spark.read.parquet(path)
            except Exception as e:
                logger.warning(f"  Skipping band {band_name} (path may not exist): {e}")
                continue

            df = df.withColumn("band", F.lit(band_name))
            df = df.withColumn("source_url", F.lit(path))
            df_list.append(df)

        if not df_list:
            raise ValueError(f"No bands found for source {source_name}")

        consolidated_df = df_list[0]
        for df in df_list[1:]:
            consolidated_df = consolidated_df.unionByName(df, allowMissingColumns=True)

        transformed_df = self._transform_schema(consolidated_df)

        # Deduplication - keep hash for partition, drop after
        window_spec = Window.partitionBy("hash").orderBy(F.col("band"))
        unique_df = (
            transformed_df.withColumn("row_num", F.row_number().over(window_spec))
            .filter(F.col("row_num") == 1)
            .drop("row_num")
        )

        # Drop hash before save (if in drop_columns)
        if (
            "hash" in unique_df.columns
            and "hash" in self.config["schema"]["drop_columns"]
        ):
            unique_df = unique_df.drop("hash")

        num_partitions = self.config["processing"].get("parallelism", 200)
        unique_df = unique_df.repartition(num_partitions)

        return unique_df

    def _transform_schema(self, df: DataFrame) -> DataFrame:
        """Applies renames and column drops."""
        rename_map = self.config["schema"]["rename_columns"]
        drop_cols = self.config["schema"]["drop_columns"]

        band_p_cols = [c for c in df.columns if c.startswith("band_p_")]
        if band_p_cols:
            df = df.withColumn(
                "band_score", F.coalesce(*[F.col(c) for c in band_p_cols])
            )

        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # Don't drop hash yet - needed for dedup
        cols_to_drop = [c for c in drop_cols if c != "hash" and c in df.columns]
        df = df.drop(*cols_to_drop)

        return df

    def save_output(self, df: DataFrame, output_path: str, source_name: str):
        """Saves as JSONL partitioned by source."""
        logger.info(f"  Saving deduplicated data for {source_name} to {output_path}")
        df.write.mode("append").partitionBy("source").json(output_path)


# =========================================================================
# MAIN
# =========================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="T3 Final Processing - EMR Serverless")
    parser.add_argument("--BUCKET", default="t2-datacurriculum-353", help="S3 bucket")
    parser.add_argument(
        "--BASE_PREFIX",
        default="processed_dataset/curriculum_data",
        help="Input base prefix",
    )
    parser.add_argument(
        "--OUTPUT_PREFIX",
        default="processed_dataset/curriculum_pyspark_output",
        help="Output prefix",
    )
    parser.add_argument(
        "--PARALLELISM", type=int, default=200, help="Shuffle/output partitions"
    )
    parser.add_argument(
        "--SOURCE",
        default=None,
        help="Optional: process only this source (e.g. redpajama-arxiv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "s3": {
            "bucket": args.BUCKET,
            "base_prefix": args.BASE_PREFIX,
            "output_prefix": args.OUTPUT_PREFIX,
            "checkpoint_path": "processed_dataset/checkpoints/curriculum_pyspark",
        },
        "processing": {
            "parallelism": args.PARALLELISM,
            "default_bands": ["B0", "B1", "B2", "B3", "B4"],
        },
        "schema": DEFAULT_CONFIG["schema"],
    }

    bucket = config["s3"]["bucket"]
    base_prefix = config["s3"]["base_prefix"]
    output_prefix = config["s3"]["output_prefix"]
    output_path = f"s3://{bucket}/{output_prefix}"

    logger.info("=" * 60)
    logger.info("T3 Final Processing - EMR Serverless")
    logger.info("=" * 60)
    logger.info(f"Input: s3://{bucket}/{base_prefix}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    spark = (
        SparkSession.builder.appName("T3_Final_Curriculum_Processing")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", config["processing"]["parallelism"])
        .getOrCreate()
    )

    processor = SparkDataProcessor(config)
    checkpoint_mgr = CheckpointManager(bucket, config["s3"]["checkpoint_path"])

    if args.SOURCE:
        sources = [args.SOURCE]
        logger.info(f"Processing single source: {args.SOURCE}")
    else:
        sources = discover_sources_from_s3(bucket, base_prefix)
    target_bands = config["processing"]["default_bands"]

    start_time = datetime.now()
    processed = 0

    for source in sources:
        if checkpoint_mgr.is_finished(source):
            logger.info(f"Skipping already processed source: {source}")
            continue

        try:
            band_paths = [
                f"s3://{bucket}/{base_prefix}/source={source}/bands/band={band}/"
                for band in target_bands
            ]

            unique_df = processor.process_source_group(spark, source, band_paths)
            processor.save_output(unique_df, output_path, source)

            stats_output_path = f"{output_path}/stats/{source}"
            generate_distribution_stats(unique_df, stats_output_path)

            checkpoint_mgr.mark_finished(source)
            processed += 1
            logger.info(f"Finished source: {source}")

        except Exception as e:
            logger.error(f"Failed processing source {source}: {e}")
            raise

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"COMPLETED: {processed} sources in {duration:.1f} seconds")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
