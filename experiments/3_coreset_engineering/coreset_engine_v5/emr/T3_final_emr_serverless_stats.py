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

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import boto3
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

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
        "default_bands": ["B0", "B1", "B2", "B3", "B4", "B5"],
    },
    "schema": {
        "rename_columns": {"id": "chunk_id"},
        "drop_columns": [
            "uuid",
            "text",
            "hash",
            "metadata",
            "assigned_band",
            "file_path",
        ],
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


def generate_distribution_stats(df, stats_output_path):
    """
    Lightweight aggregation after dedup + chunking.
    Uses existing word_count and token_count_estimate.
    """
    logger.info("Starting distribution stats aggregation...")

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
        agg_df.join(F.broadcast(source_totals), "source")
        .withColumn(
            "pct_of_source_tokens",
            F.when(
                F.col("source_total_tokens") > 0,
                F.col("total_tokens") / F.col("source_total_tokens"),
            ).otherwise(F.lit(0.0)),
        )
        .drop("source_total_tokens")
    )

    logger.info("Writing distribution Parquet...")
    (final_stats_df.coalesce(1).write.mode("overwrite").parquet(stats_output_path))
    logger.info("Distribution stats written successfully.")


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
        except Exception as e:
            logger.error(f"Error checking checkpoint: {e}")
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
    ) -> Tuple[DataFrame, Dict[str, Any]]:
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
            df = df.withColumn(
                "source_doc_id", F.element_at(F.split(F.input_file_name(), "/"), -1)
            )
            df_list.append(df)

        if not df_list:
            raise ValueError(f"No bands found for source {source_name}")

        consolidated_df = df_list[0]
        for df in df_list[1:]:
            consolidated_df = consolidated_df.unionByName(df, allowMissingColumns=True)

        # Add global source column for tracking (ensures it is in JSON)
        consolidated_df = consolidated_df.withColumn("source", F.lit(source_name))

        # 1. Thin out the data (rename and drop heavy columns before shuffle)
        transformed_df = self._transform_schema(consolidated_df)

        # Ensure assigned_band exists and is never null (coalesce with the folder-derived band)
        if "assigned_band" not in transformed_df.columns:
            transformed_df = transformed_df.withColumn("assigned_band", F.col("band"))
        else:
            transformed_df = transformed_df.withColumn(
                "assigned_band", F.coalesce(F.col("assigned_band"), F.col("band"))
            )

        # Ensure metadata columns are strings and not null
        transformed_df = transformed_df.withColumn(
            "source_url", F.coalesce(F.col("source_url"), F.lit("")).cast("string")
        )
        transformed_df = transformed_df.withColumn(
            "source_doc_id",
            F.coalesce(F.col("source_doc_id"), F.lit("")).cast("string"),
        )

        # --- CALCULATE INPUT STATS (Pre-Dedup) ---
        input_stats = transformed_df.select(
            F.count("*").alias("docs"),
            F.sum("word_count").alias("words"),
            F.sum("token_count_estimate").alias("tokens"),
        ).collect()[0]

        # 2. Deduplication on THIN data - based ONLY on the hash column
        unique_df = transformed_df.dropDuplicates(["hash"])

        # --- CALCULATE UNIQUE STATS (Post-Dedup) ---
        unique_stats = unique_df.select(
            F.count("*").alias("docs"),
            F.sum("word_count").alias("words"),
            F.sum("token_count_estimate").alias("tokens"),
        ).collect()[0]

        # Post-Dedup Logic: Band assignments and scoring
        # Final 'band' is derived from 'assigned_band'
        unique_df = unique_df.withColumn("band", F.col("assigned_band"))

        # Ensure all probability columns exist (B0 to B5) and are correctly typed
        for i in range(6):
            col_name = f"band_p_B{i}"
            if col_name not in unique_df.columns:
                unique_df = unique_df.withColumn(col_name, F.lit(0.0))
            else:
                unique_df = unique_df.withColumn(
                    col_name, F.col(col_name).cast("double")
                )

        # Calculate band_score based on the final assigned_band
        # Use trim and upper to ensure robust string matching
        unique_df = unique_df.withColumn(
            "_band_match", F.trim(F.upper(F.col("assigned_band")))
        )
        unique_df = unique_df.withColumn(
            "band_score",
            F.when(F.col("_band_match") == "B0", F.col("band_p_B0"))
            .when(F.col("_band_match") == "B1", F.col("band_p_B1"))
            .when(F.col("_band_match") == "B2", F.col("band_p_B2"))
            .when(F.col("_band_match") == "B3", F.col("band_p_B3"))
            .when(F.col("_band_match") == "B4", F.col("band_p_B4"))
            .when(F.col("_band_match") == "B5", F.col("band_p_B5"))
            .otherwise(F.lit(0.0)),
        ).drop("_band_match")

        # 4. Final Cleanup: Drop hash and assigned_band as the very last step
        final_drops = ["hash", "assigned_band"]
        unique_df = unique_df.drop(*[c for c in final_drops if c in unique_df.columns])

        num_partitions = self.config["processing"].get("parallelism", 200)
        unique_df = unique_df.coalesce(num_partitions)

        stats_dict = {
            "input": {
                "docs": input_stats["docs"],
                "words": input_stats["words"] or 0,
                "tokens": input_stats["tokens"] or 0,
            },
            "unique": {
                "docs": unique_stats["docs"],
                "words": unique_stats["words"] or 0,
                "tokens": unique_stats["tokens"] or 0,
            },
        }

        return unique_df, stats_dict

    def _transform_schema(self, df: DataFrame) -> DataFrame:
        """Applies renames and early column drops for shuffle efficiency."""
        rename_map = self.config["schema"]["rename_columns"]
        drop_cols = self.config["schema"]["drop_columns"]

        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # Early drop of heavy columns to optimize shuffle
        # MUST preserve hash, assigned_band, metadata, and probability columns for later logic
        prob_cols = [c for c in df.columns if c.startswith("band_p_")]
        cols_to_keep = {"hash", "assigned_band", "source_doc_id", "source_url"} | set(
            prob_cols
        )

        early_drop_targets = [
            c for c in drop_cols if c not in cols_to_keep and c in df.columns
        ]
        df = df.drop(*early_drop_targets)

        return df

    def save_output(self, df: DataFrame, output_path: str, source_name: str):
        """Saves as Parquet. We write to source-specific folder explicitly to keep 'source' column inside the data."""
        final_path = f"{output_path}/source={source_name}"
        logger.info(f"  Saving deduplicated data to {final_path}")
        df.write.mode("append").parquet(final_path)


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
            "default_bands": ["B0", "B1", "B2", "B3", "B4", "B5"],
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
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "auto")
        .config("spark.sql.shuffle.partitions", config["processing"]["parallelism"])
        .getOrCreate()
    )

    # Include only if needed
    # .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    # .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
    # .config("spark.sql.files.maxPartitionBytes", "128m") \
    # .config("spark.executor.memoryOverhead", "2g") \

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

            unique_df, stats = processor.process_source_group(spark, source, band_paths)

            # --- LOG DETAILED STATS ---
            logger.info(f"Source: {source} - Statistics:")
            logger.info(
                f"  Input:  {stats['input']['docs']:,} docs, {stats['input']['words']:,} words, {stats['input']['tokens']:,} tokens"
            )
            logger.info(
                f"  Unique: {stats['unique']['docs']:,} docs, {stats['unique']['words']:,} words, {stats['unique']['tokens']:,} tokens"
            )

            dropped_docs = stats["input"]["docs"] - stats["unique"]["docs"]
            dropped_words = stats["input"]["words"] - stats["unique"]["words"]
            dropped_tokens = stats["input"]["tokens"] - stats["unique"]["tokens"]

            logger.info(
                f"  Dropped (Duplicates): {dropped_docs:,} docs, {dropped_words:,} words, {dropped_tokens:,} tokens"
            )

            if stats["input"]["tokens"] > 0:
                reduction_pct = (dropped_tokens / stats["input"]["tokens"]) * 100
                logger.info(f"  Token Reduction: {reduction_pct:.2f}%")

            # Persist to avoid re-executing the entire DAG for stats
            unique_df.persist(StorageLevel.DISK_ONLY)

            processor.save_output(unique_df, output_path, source)

            stats_output_path = f"{output_path}/stats/{source}"
            generate_distribution_stats(unique_df, stats_output_path)

            # Checkpoint AFTER stats so retries regenerate both data + stats
            checkpoint_mgr.mark_finished(source)
            processed += 1
            logger.info(f"Finished source: {source}")

        except Exception as e:
            logger.error(f"Failed processing source {source}: {e}")
            raise

        finally:
            # Memory cleanup between sources
            try:
                unique_df.unpersist()
            except Exception:
                pass
            # Add only if needed due to memory issues
            # spark.catalog.clearCache()
            spark._jvm.System.gc()

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"COMPLETED: {processed} sources in {duration:.1f} seconds")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
