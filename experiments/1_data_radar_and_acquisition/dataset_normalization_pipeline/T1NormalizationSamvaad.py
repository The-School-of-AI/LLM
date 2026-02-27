"""
Samvaad-Hi Data Normalization - Format and Convert to Normalized Document Schema
==============================================================================
Purpose: Read Samvaad-Hi (sarvamai/samvaad-hi-v1) Parquet, format conversation
         array to Option 1 flattened text ([USER] / [ASSISTANT] markers),
         normalize to document schema (same as Dolma/Sangraha/NCERT),
         write as Parquet. Output: Partitioned by source (source=samvaad_hi).

Usage:
    aws glue start-job-run --job-name Samvaad_data_normalization \\
        --region us-east-1 \\
        --arguments '{"--INPUT_PATH": "s3://.../samvaad-hi-v1/data/"}'
"""

import json
import sys
from typing import Any

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, TimestampType
from pyspark.sql.window import Window

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

VERSION = "2025"
OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
DEFAULT_INPUT_PATH = "s3://t1-dataacquisition-datasets/huggingface_sarvamai"

# Column name for conversation array (list of {content, role}). Tried in order.
CONVERSATION_COL_NAMES = ("messages", "conversation", "data", "chat")


def format_conversation_option1(messages: Any) -> str:
    """Format conversation array to Option 1: [USER] / [ASSISTANT] flattened text.

    Args:
        messages: List of dicts with 'content' and 'role' (user/assistant).
                  Can be list, or JSON string.

    Returns:
        Single string with turns separated by newlines.
    """
    if messages is None:
        return ""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return ""
    if not isinstance(messages, (list, tuple)):
        return ""
    parts = []
    for m in messages:
        if isinstance(m, dict):
            content = (m.get("content") or m.get("value") or "").strip()
            role = (m.get("role") or m.get("from") or "user").lower()
        else:
            continue
        if not content:
            continue
        label = "USER" if role in ("user", "human") else "ASSISTANT"
        parts.append(f"[{label}] {content}")
    return "\n\n".join(parts)


def get_glue_args():
    """Parse Glue job arguments."""
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    optional = {}
    if "--INPUT_PATH" in sys.argv:
        optional["INPUT_PATH"] = getResolvedOptions(sys.argv, ["INPUT_PATH"])[
            "INPUT_PATH"
        ]
    else:
        optional["INPUT_PATH"] = DEFAULT_INPUT_PATH
    return args, optional


def process_samvaad(spark, input_path: str) -> None:
    """Read Samvaad-Hi Parquet, format conversation to Option 1 text, normalize, write Parquet."""
    print(f"Reading Parquet from: {input_path}")
    df = spark.read.parquet(input_path)

    conv_col = None
    for c in CONVERSATION_COL_NAMES:
        if c in df.columns:
            conv_col = c
            break
    if conv_col is None:
        raise ValueError(
            f"Samvaad Parquet must have a conversation column (e.g. 'messages', 'conversation', 'data'). "
            f"Columns found: {df.columns}"
        )

    col_type = df.schema[conv_col].dataType
    if isinstance(col_type, ArrayType):
        df = df.withColumn("_conv_json", F.to_json(F.col(conv_col)))
    elif "string" in str(col_type).lower():
        df = df.withColumn("_conv_json", F.col(conv_col))
    else:
        df = df.withColumn("_conv_json", F.to_json(F.col(conv_col)))

    @F.udf(StringType())
    def format_conv_udf(js: str) -> str:
        return format_conversation_option1(js)

    df = df.withColumn("_formatted_text", format_conv_udf(F.col("_conv_json")))
    df = df.filter(F.length(F.col("_formatted_text")) > 0)

    df_out = (
        df.withColumn("hash", F.sha2(F.col("_formatted_text"), 256))
        .withColumn("dataset", F.lit("samvaad"))
        .withColumn("domain", F.lit("conversation"))
        .withColumn("source", F.lit("samvaad_hi"))
        .withColumn("text", F.col("_formatted_text"))
        .withColumn("language", F.lit("hi"))
        .withColumn("metadata", F.lit("{}"))
        .withColumn("added", F.lit(None).cast(TimestampType()))
        .withColumn("created", F.lit(None).cast(TimestampType()))
        .withColumn("version", F.lit(VERSION))
    )
    w = Window.orderBy(F.monotonically_increasing_id())
    df_out = df_out.withColumn(
        "id", F.concat(F.lit("samvaad_hi_"), F.row_number().over(w).cast("string"))
    ).select(
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

    record_count = df_out.count()
    print(f"Records: {record_count:,}")
    num_partitions = max(1, min(400, record_count // 5000))
    df_out = df_out.repartition(num_partitions)

    output_path = f"{OUTPUT_BASE}/source=samvaad_hi"
    print(f"Writing to: {output_path}")
    (df_out.write.mode("overwrite").option("compression", "zstd").parquet(output_path))
    print("✓ Completed Samvaad-Hi processing")


def main():
    args, optional = get_glue_args()
    input_path = optional["INPUT_PATH"]

    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "268435456")
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")

    job = Job(glue_context)
    job.init("Samvaad-Hi Data Normalization", args)

    print("=" * 80)
    print("Samvaad-Hi Data Normalization - Starting")
    print(f"Input:  {input_path}")
    print(f"Output: {OUTPUT_BASE}/source=samvaad_hi")
    print("=" * 80)

    try:
        process_samvaad(spark, input_path)
    except Exception as e:
        print(f"ERROR: {e}")
        raise

    print("\n" + "=" * 80)
    print("Samvaad-Hi Data Normalization - Completed")
    print("=" * 80)
    job.commit()


if __name__ == "__main__":
    main()
