"""
Sangraha Data Normalization - Parquet to Normalized Document Schema
====================================================================
Purpose: Read Sangraha Parquet datasets (from Hugging Face ai4bharat/sangraha),
         normalize to document schema, and write as Parquet.
         Sangraha is already in Parquet format with doc_id and text columns.
         Output: Partitioned by source and language

Usage:
    aws glue start-job-run --job-name T1_data_normalization \\
        --region us-east-1 \
        --arguments '{
            "--DATASETS":"dolma_arxiv_v1_7,dolma_cc_news_v1_7"
        }'

    aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --arguments '{
        "--DATASETS":"dolma_algebraic_v1_7",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/shuffle_temp/"
    }'

    # Enable S3 Shuffle to avoid disk space errors on large datasets (e.g., 900GB)
    # This allows using cheaper G.1X workers for massive shuffles.
    
    # running on flex, with auto-scaling upto 20 G1X workers, and S3 shuffle enabled;
        - For >700Gb dataset, we are increasing autoscaling to 40 G2X and time out from default 8 hrs to 20hrs

"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, TimestampType

# ISO 639-1 (2-char) mapping from Sangraha 3-char language codes
LANG_3_TO_2 = {
    "asm": "as",  # Assamese
    "ben": "bn",  # Bengali
    "guj": "gu",  # Gujarati
    "hin": "hi",  # Hindi
    "kan": "kn",  # Kannada
    "mal": "ml",  # Malayalam
    "mar": "mr",  # Marathi
    "ori": "or",  # Odia
    "pan": "pa",  # Punjabi
    "tam": "ta",  # Tamil
    "tel": "te",  # Telugu
}

# -------------------------------------------------------------------------
# DATASET CONFIGURATION
# -------------------------------------------------------------------------

DATASETS_CONFIG = {
    "asm": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/asm/",
        "source": "sangraha",
        "domain": "web",
        "language": "asm",
        "file_count": 26,
        "dataset_size_in_GB": 98.8,
    },
    "ben": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/ben/",
        "source": "sangraha",
        "domain": "web",
        "language": "ben",
        "file_count": 100,
        "dataset_size_in_GB": 24,
    },
    "guj": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/guj/",
        "source": "sangraha",
        "domain": "web",
        "language": "guj",
        "file_count": 5,
        "dataset_size_in_GB": 16.5,
    },
    "hin": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/hin/",
        "source": "sangraha",
        "domain": "web",
        "language": "hin",
        "file_count": 49,
        "dataset_size_in_GB": 189.8,
    },
    "kan": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/kan/",
        "source": "sangraha",
        "domain": "web",
        "language": "kan",
        "file_count": 16,
        "dataset_size_in_GB": 10.3,
    },
    "mal": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/mal/",
        "source": "sangraha",
        "domain": "web",
        "language": "mal",
        "file_count": 4,
        "dataset_size_in_GB": 7.3,
    },
    "mar": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/mar/",
        "source": "sangraha",
        "domain": "web",
        "language": "mar",
        "file_count": 13,
        "dataset_size_in_GB": 12.1,
    },
    "ori": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/ori/",
        "source": "sangraha",
        "domain": "web",
        "language": "ori",
        "file_count": 66,
        "dataset_size_in_GB": 27.1,
    },
    "pan": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/pan/",
        "source": "sangraha",
        "domain": "web",
        "language": "pan",
        "file_count": 171,
        "dataset_size_in_GB": 266.7,
    },
    "tam": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/tam/",
        "source": "sangraha",
        "domain": "web",
        "language": "tam",
        "file_count": 500,
        "dataset_size_in_GB": 829.3,
    },
    "tel": {
        "path": "s3://t1-dataacquisition-datasets/huggingface_sangraha/verified/tel/",
        "source": "sangraha",
        "domain": "web",
        "language": "te",
        "file_count": 264,
        "dataset_size_in_GB": 28.3,
    },
}


VERSION = "1.7"
OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"

# Sangraha Parquet schema: doc_id (string), text (string)
# Target document schema: id, hash, dataset, domain, source, text, language, metadata, added, created, version


def get_glue_args():
    """Parse Glue job arguments."""
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])

    # Optional arguments with defaults
    optional_args = {}
    if "--DATASETS" in sys.argv:
        optional_args["DATASETS"] = getResolvedOptions(sys.argv, ["DATASETS"])[
            "DATASETS"
        ]
    else:
        optional_args["DATASETS"] = "all"  # Process all datasets by default

    return args, optional_args


def process_dataset(spark, dataset_name, config):
    """
    Process a single Sangraha dataset: read Parquet, normalize to document schema, write Parquet.
    Sangraha schema: doc_id, text. Target schema: id, hash, dataset, domain, source, text,
    language, metadata, added, created, version.
    """
    print(f"Processing dataset: {dataset_name}")
    print(
        f"  Source: {config['source']}, Domain: {config['domain']}, Language: {config['language']}"
    )
    target_mb = 256
    dataset_gb = config.get("dataset_size_in_GB", 1.0)
    num_partitions = max(1, int((dataset_gb * 1024) / target_mb * 1.1))

    print(f"  Dataset Size: {dataset_gb} GB")
    print(f"  Target Partitions: {num_partitions}")

    # Read Sangraha Parquet (schema: doc_id, text - Hugging Face ai4bharat/sangraha)
    df = spark.read.parquet(config["path"])

    # Map doc_id -> id (Sangraha uses doc_id)
    df_base = df.withColumn("_id", F.col("doc_id"))

    # Repartition
    df_distributed = df_base.repartition(num_partitions)

    # Normalize to document schema: add hash, metadata columns; fill missing fields with null
    df_out = (
        df_distributed.withColumn("id", F.col("_id"))
        .withColumn("hash", F.sha2(F.col("text"), 256))
        .withColumn("dataset", F.lit("sangraha"))
        .withColumn("domain", F.lit(config["domain"]))
        .withColumn(
            "source",
            F.lit(
                f"sangraha_{LANG_3_TO_2.get(config['language'], config['language'])}"
            ),
        )
        .withColumn(
            "language", F.lit(LANG_3_TO_2.get(config["language"], config["language"]))
        )
        .withColumn("metadata", F.lit(None).cast(StringType()))
        .withColumn("added", F.lit(None).cast(TimestampType()))
        .withColumn("created", F.lit(None).cast(TimestampType()))
        .withColumn("version", F.lit(VERSION))
        .select(
            "id",
            "hash",
            "dataset",
            "domain",
            "source",
            "text",
            "language",
            "metadata",
            "added",
            "created",
            "version",
        )
    )

    # Write partitioned by source only (same format as Dolma): source=sangraha_as, source=sangraha_bn, etc.
    lang_2 = LANG_3_TO_2.get(config["language"], config["language"])
    source_val = f"sangraha_{lang_2}"
    output_path = f"{OUTPUT_BASE}/source={source_val}"
    print(f"  Writing to: {output_path}")
    (
        df_out.write.mode(
            "overwrite"
        )  # overwrite since this is running sequentially at folder level
        .option("compression", "zstd")
        .parquet(output_path)
    )

    print(f"  ✓ Completed: {dataset_name}")


def main():
    """Main execution logic."""
    args, optional_args = get_glue_args()

    datasets_to_process = optional_args["DATASETS"]

    # Initialize Spark
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session

    # --- OPTIMIZATION CONFIGS ---
    # Enable AQE to handle the huge data load dynamically
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    # Target 256MB per file for S3 efficiency
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "268435456")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "268435456")
    # Optimize Parquet for large-scale writes
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")

    job = Job(glueContext)
    job.init("Sangraha Data Normalization", args)

    print("=" * 80)
    print("Sangraha Data Normalization - Starting")
    print(f"Output Base: {OUTPUT_BASE}")
    print(f"Datasets Filter: {datasets_to_process}")
    print("=" * 80)

    # Determine which datasets to process
    if datasets_to_process == "all":
        datasets = DATASETS_CONFIG.items()
    else:
        dataset_names = [d.strip() for d in datasets_to_process.split(",")]
        datasets = [
            (name, DATASETS_CONFIG[name])
            for name in dataset_names
            if name in DATASETS_CONFIG
        ]

        if not datasets:
            print(f"ERROR: No valid datasets found in filter: {datasets_to_process}")
            print(f"Available datasets: {', '.join(DATASETS_CONFIG.keys())}")
            job.commit()
            return

    # Process each dataset sequentially to avoid memory issues
    total = len(datasets)
    for idx, (name, config) in enumerate(datasets, 1):
        print(f"\n[{idx}/{total}] Processing: {name}")
        try:
            process_dataset(spark, name, config)

            # 2. CLEAR CACHE & UNPERSIST
            # Removes metadata for all cached tables/DataFrames in this session
            spark.catalog.clearCache()

            # 3. GLOBAL CLEANUP
            # Suggests to the JVM that it's a good time for garbage collection
            # (Optional, but helpful in long Glue loops)
            sc._jvm.System.gc()

        except Exception as e:
            print(f"ERROR processing {name}: {str(e)}")
            # Continue with next dataset instead of failing entire job
            continue

    print("\n" + "=" * 80)
    print("Sangraha Data Normalization - Completed")
    print("=" * 80)

    job.commit()


if __name__ == "__main__":
    main()
