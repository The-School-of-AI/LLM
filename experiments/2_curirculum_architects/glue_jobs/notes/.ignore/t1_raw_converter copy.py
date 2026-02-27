"""
T1 Raw Data Converter - Simple Parquet Conversion with Metadata
================================================================
Purpose: Read raw JSON.gz datasets, add basic metadata columns, and write as Parquet.
Data: 4TB | Output: Partitioned by source

Usage:
    aws glue start-job-run --job-name t1-raw-converter \\
        --arguments '{
            "--DATASETS":"dolma_arxiv_v1_7,dolma_cc_news_v1_7",
            "--OUTPUT_BASE":"s3://bucket/processed_dataset/raw_data"
        }'
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# -------------------------------------------------------------------------
# DATASET CONFIGURATION
# -------------------------------------------------------------------------
DATASETS_CONFIG = {
    "dolma_Pes2o_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_Pes2o_v1_7",
        "source": "pes2o",
        "domain": "science"
    },
    "dolma_arxiv_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/arxiv_v1_7/",
        "source": "redpajama-arxiv",
        "domain": "science"
    },
    "dolmas_cc_news_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_news_v1_7",
        "source": "cc_news",
        "domain": "news"
    },
    "dolmas_starcoder_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_starcoder_v1_7",
        "source": "Starcoder",
        "domain": "code"
    },
    "dolmas_algebraic-stack_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_algebraic_v1_7",
        "source": "proof_pile_2-algebraic_stack",
        "domain": "math"
    },
    "dolmas_books_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/books/",
        "source": "books",
        "domain": "literature"
    },
    "dolmas_open-web-math_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_open-web-math-train_v1_7",
        "source": "proof_pile_2-open_web_math",
        "domain": "math"
    },
    "dolmas_tulu_flan_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_tulu_flan_v1_7",
        "source": "flan",
        "domain": "instruction"
    },
    "dolma_C4_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_urls_C4_v1_7",
        "source": "C4",
        "domain": "web"
    },
    "dolma_RefineWeb_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_RefineWeb_v1_7/",
        "source": "refinedweb",
        "domain": "web"
    },
    "dolma_megawika_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_megawika_v1_7",
        "source": "megawika",
        "domain": "encyclopedia"
    },
    "dolma_reddit_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_reddit_v1_7",
        "source": "reddit",
        "domain": "social"
    },
    "dolma_stackexchange_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_stackexchange_v1_7",
        "source": "stackexchange",
        "domain": "qa"
    },
    "dolma_cc_en_head_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_head_v1_7/",
        "source": "cc",
        "domain": "web"
    },
    "dolma_cc_en_middle_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_middle_v1_7",
        "source": "cc",
        "domain": "web"
    },
    "dolma_cc_en_tail_v1_7": {
        "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_tail_v1_7",
        "source": "cc",
        "domain": "web"
    }
}

VERSION = "1.7"

# -------------------------------------------------------------------------
# SCHEMA DEFINITION
# -------------------------------------------------------------------------
INPUT_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("metadata", StringType(), True),
    StructField("added", TimestampType(), True),
    StructField("created", TimestampType(), True)
])

def get_glue_args():
    """Parse Glue job arguments."""
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    # Optional arguments with defaults
    optional_args = {}
    if '--DATASETS' in sys.argv:
        optional_args['DATASETS'] = getResolvedOptions(sys.argv, ['DATASETS'])['DATASETS']
    else:
        optional_args['DATASETS'] = 'all'  # Process all datasets by default
    
    if '--OUTPUT_BASE' in sys.argv:
        optional_args['OUTPUT_BASE'] = getResolvedOptions(sys.argv, ['OUTPUT_BASE'])['OUTPUT_BASE']
    else:
        optional_args['OUTPUT_BASE'] = 's3://t1-dataacquisition-datasets/processed_dataset/raw_data'
    
    return args, optional_args

def process_dataset(spark, dataset_name, config, output_base):
    """
    Process a single dataset: read JSON.gz, add metadata, write Parquet.
    
    Args:
        spark: SparkSession
        dataset_name: Name identifier for the dataset
        config: Dict with path, source, domain
        output_base: S3 base path for output
    """
    print(f"Processing dataset: {dataset_name}")
    print(f"  Source: {config['source']}, Domain: {config['domain']}")
    
    # Read JSON.gz with explicit schema
    df = (
        spark.read
        .schema(INPUT_SCHEMA)
        .option("compression", "gzip")
        .json(config['path'])
    )
    
    # Add metadata columns
    df_out = (
        df
        .withColumn("hash", F.sha2(F.col("text"), 256))
        .withColumn("dataset", F.lit("dolma"))
        .withColumn("domain", F.lit(config['domain']))
        .withColumn("source", F.lit(config['source']))
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
            "version"
        )
    )
    
    # Write partitioned by source
    output_path = f"{output_base}/source={config['source']}"
    
    print(f"  Writing to: {output_path}")
    (
        df_out
        .repartition(100)  # Adjust based on dataset size
        .write
        .mode("overwrite")
        .option("compression", "zstd")
        .parquet(output_path)
    )

    print(f"  ✓ Completed: {dataset_name}")

def main():
    """Main execution logic."""
    args, optional_args = get_glue_args()
    
    output_base = optional_args['OUTPUT_BASE']
    datasets_to_process = optional_args['DATASETS']
    
    # Initialize Spark
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    spark.conf.set("spark.sql.shuffle.partitions", "1000")  # Adjust based on cluster size
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
    spark.conf.set("spark.io.compression.zstd.level", "3")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    print("=" * 80)
    print("T1 Raw Data Converter - Starting")
    print(f"Output Base: {output_base}")
    print(f"Datasets Filter: {datasets_to_process}")
    print("=" * 80)
    
    # Determine which datasets to process
    if datasets_to_process == 'all':
        datasets = DATASETS_CONFIG.items()
    else:
        dataset_names = [d.strip() for d in datasets_to_process.split(',')]
        datasets = [(name, DATASETS_CONFIG[name]) for name in dataset_names if name in DATASETS_CONFIG]
        
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
            process_dataset(spark, name, config, output_base)
        except Exception as e:
            print(f"ERROR processing {name}: {str(e)}")
            # Continue with next dataset instead of failing entire job
            continue
    
    print("\n" + "=" * 80)
    print("T1 Raw Data Converter - Completed")
    print("=" * 80)
    
    job.commit()

if __name__ == '__main__':
    main()
