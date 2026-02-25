"""
T1 Raw Data Converter - Simple Parquet Conversion with Metadata
================================================================
Purpose: Read raw JSON.gz datasets, add basic metadata columns, and write as Parquet.
Data: ~4TB | Output: Partitioned by source

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
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

# -------------------------------------------------------------------------
# DATASET CONFIGURATION
# -------------------------------------------------------------------------

DATASETS_CONFIG = {
    "dolma_Pes2o_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_Pes2o_v1_7",
        "source": "pes2o",
        "domain": "science",
        "file_count": 26,
        "dataset_size_in_GB": 98.8,
    },
    "dolma_arxiv_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/arxiv_v1_7/",
        "source": "redpajama-arxiv",
        "domain": "science",
        "file_count": 100,
        "dataset_size_in_GB": 24,
    },
    "dolmas_cc_news_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_news_v1_7",
        "source": "cc_news",
        "domain": "news",
        "file_count": 5,
        "dataset_size_in_GB": 16.5,
    },
    "dolmas_starcoder_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_starcoder_v1_7",
        "source": "Starcoder",
        "domain": "code",
        "file_count": 49,
        "dataset_size_in_GB": 189.8,
    },
    "dolmas_algebraic-stack_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_algebraic_v1_7",
        "source": "proof_pile_2-algebraic_stack",
        "domain": "math",
        "file_count": 16,
        "dataset_size_in_GB": 10.3,
    },
    "dolmas_books_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/books/",
        "source": "books",
        "domain": "literature",
        "file_count": 4,
        "dataset_size_in_GB": 7.3,
    },
    "dolmas_open-web-math_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_open-web-math-train_v1_7",
        "source": "proof_pile_2-open_web_math",
        "domain": "math",
        "file_count": 13,
        "dataset_size_in_GB": 12.1,
    },
    "dolmas_tulu_flan_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_tulu_flan_v1_7",
        "source": "flan",
        "domain": "instruction",
        "file_count": 66,
        "dataset_size_in_GB": 27.1,
    },
    "dolma_C4_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_urls_C4_v1_7",
        "source": "C4",
        "domain": "web",
        "file_count": 171,
        "dataset_size_in_GB": 266.7,
    },
    "dolma_RefineWeb_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_RefineWeb_v1_7/",
        "source": "refinedweb",
        "domain": "web",
        "file_count": 500,
        "dataset_size_in_GB": 829.3,
    },
    "dolma_megawika_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_megawika_v1_7",
        "source": "megawika",
        "domain": "encyclopedia",
        "file_count": 264,
        "dataset_size_in_GB": 28.3,
    },
    "dolma_reddit_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_reddit_v1_7",
        "source": "reddit",
        "domain": "social",
        "file_count": 78,
        "dataset_size_in_GB": 158,
    },
    "dolma_stackexchange_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_stackexchange_v1_7",
        "source": "stackexchange",
        "domain": "qa",
        "file_count": 26,
        "dataset_size_in_GB": 23.3,
    },
    "dolma_cc_en_head_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_head_v1_7/",
        "source": "cc_head",
        "domain": "web",
        "file_count": 275,
        "dataset_size_in_GB": 723.2,
    },
    "dolma_cc_en_middle_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_middle_v1_7",
        "source": "cc_middle",
        "domain": "web",
        "file_count": 379,
        "dataset_size_in_GB": 932.2,
    },
    "dolma_cc_en_tail_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_tail_v1_7",
        "source": "cc_tail",
        "domain": "web",
        "file_count": 444,
        "dataset_size_in_GB": 840.1,
    },
}

VERSION = "1.7"
OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"

# -------------------------------------------------------------------------
# SCHEMA DEFINITION
# -------------------------------------------------------------------------
INPUT_SCHEMA = StructType(
    [
        StructField("id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("metadata", StringType(), True),
        StructField("added", TimestampType(), True),
        StructField("created", TimestampType(), True),
    ]
)


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
    Process a single dataset: read JSON.gz, add metadata, write Parquet.

    Args:
        spark: SparkSession
        dataset_name: Name identifier for the dataset
        config: Dict with path, source, domain
    """
    print(f"Processing dataset: {dataset_name}")
    print(f"  Source: {config['source']}, Domain: {config['domain']}")
    target_mb = 256
    dataset_gb = config.get("dataset_size_in_GB", 1.0)  # Default to 1GB if missing

    # We use a slight multiplier (1.1) because Parquet metadata and
    # dictionary encoding can vary, and it's better to have 230MB files than 300MB.
    num_partitions = max(1, int((dataset_gb * 1024) / target_mb * 1.1))

    print(f"  Dataset Size: {dataset_gb} GB")
    print(f"  Target Partitions: {num_partitions}")

    # Read JSON.gz with explicit schema
    df = (
        spark.read.schema(INPUT_SCHEMA)
        .option("compression", "gzip")
        .json(config["path"])
    )

    # 2. Repartition immediately (Forces data to spread out)
    df_distributed = df.repartition(num_partitions)

    # Add metadata columns
    df_out = (
        df_distributed.withColumn("hash", F.sha2(F.col("text"), 256))
        .withColumn("dataset", F.lit("dolma"))
        .withColumn("domain", F.lit(config["domain"]))
        .withColumn("source", F.lit(config["source"]))
        .withColumn("language", F.lit("en"))
        .withColumn("metadata", F.col("metadata").cast("string"))
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

    # Write partitioned by source
    output_path = f"{OUTPUT_BASE}/source={config['source']}"
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
    job.init("Data Normalization - Raw Converter", args)

    print("=" * 80)
    print("T1 Raw Data Converter - Starting")
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
    print("T1 Raw Data Converter - Completed")
    print("=" * 80)

    job.commit()


if __name__ == "__main__":
    main()
