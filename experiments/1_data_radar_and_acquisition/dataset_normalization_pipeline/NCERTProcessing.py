"""
NCERT Data Normalization - Format and Convert to Normalized Document Schema
============================================================================
Purpose: Read NCERT dataset (CSV from Hugging Face KadamParth/Ncert_dataset),
         format text from multiple fields, normalize to document schema
         (same as Dolma/Sangraha), write as Parquet.
         Output: Partitioned by source (source=ncert)

Usage:
    aws glue start-job-run --job-name NCERT_data_normalization \\
        --region us-east-1 \\
        --arguments '{"--INPUT_PATH": "s3://t1-dataacquisition-datasets/huggingface_NCERT/NCERT_Dataset.csv"}'
"""

import hashlib
import json
import sys
from typing import Any, Dict

import pandas as pd
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

DATASET_NAME = "ncert"
VERSION = "2025"
OUTPUT_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"

# Default input path (override via --INPUT_PATH)
# KadamParth/Ncert_dataset from Hugging Face - CSV format
DEFAULT_INPUT_PATH = (
    "s3://t1-dataacquisition-datasets/huggingface_NCERT/NCERT_Dataset.csv"
)


def format_text(example: Dict[str, Any]) -> str:
    """Format NCERT example fields into structured text.

    Args:
        example: Dictionary containing NCERT dataset fields

    Returns:
        Formatted text combining all fields
    """
    parts = []

    topic = example.get("Topic", example.get("topic", ""))
    if topic:
        parts.append(f"### Topic: {topic}")

    explanation = example.get("Explanation", example.get("explanation", ""))
    if explanation:
        parts.append(f"\n### Explanation:\n{explanation}")

    question = example.get("Question", example.get("question", ""))
    if question:
        parts.append(f"\n### Question:\n{question}")

    answer = example.get("Answer", example.get("answer", ""))
    if answer:
        parts.append(f"\n### Answer:\n{answer}")

    metadata_parts = []
    difficulty = example.get("Difficulty", example.get("difficulty", ""))
    if difficulty:
        metadata_parts.append(f"Difficulty: {difficulty}")

    student_level = example.get("StudentLevel", example.get("student_level", ""))
    if student_level:
        metadata_parts.append(f"Student Level: {student_level}")

    subject = example.get("subject", example.get("Subject", ""))
    if subject:
        metadata_parts.append(f"Subject: {subject}")

    grade = example.get("grade", example.get("Grade", ""))
    if grade:
        metadata_parts.append(f"Grade: {grade}")

    estimated_time = example.get("EstimatedTime", example.get("estimated_time", ""))
    if estimated_time:
        metadata_parts.append(f"Estimated Time: {estimated_time} minutes")

    prerequisites = example.get("Prerequisites", example.get("prerequisites", ""))
    if prerequisites:
        metadata_parts.append(f"Prerequisites: {prerequisites}")

    if metadata_parts:
        parts.append("\n### Metadata:\n" + "  \n".join(metadata_parts))

    return "\n".join(parts)


def build_metadata(example: Dict[str, Any]) -> str:
    """Build metadata dict and serialize to JSON string."""
    metadata = {
        "subject": str(example.get("subject", example.get("Subject", "")) or ""),
        "grade": str(example.get("grade", example.get("Grade", "")) or ""),
        "topic": str(example.get("Topic", example.get("topic", "")) or ""),
        "difficulty": str(
            example.get("Difficulty", example.get("difficulty", "")) or ""
        ),
        "student_level": str(
            example.get("StudentLevel", example.get("student_level", "")) or ""
        ),
        "question_type": str(
            example.get("QuestionType", example.get("question_type", "")) or ""
        ),
        "question_complexity": str(
            example.get("QuestionComplexity", example.get("question_complexity", ""))
            or ""
        ),
        "estimated_time": str(
            example.get("EstimatedTime", example.get("estimated_time", "")) or ""
        ),
        "prerequisites": str(
            example.get("Prerequisites", example.get("prerequisites", "")) or ""
        ),
        "source_type": "textbook",
    }
    return json.dumps(metadata)


def ncert_map_func(iterator):
    """Process NCERT rows: format text, build metadata, output document schema."""
    for pdf in iterator:
        rows = []
        for idx, row in pdf.iterrows():
            example = row.to_dict()
            formatted_text = format_text(example)
            metadata_json = build_metadata(example)

            hash_val = hashlib.sha256(formatted_text.encode("utf-8")).hexdigest()
            record_id = f"ncert_{idx}"

            rows.append(
                {
                    "id": record_id,
                    "hash": hash_val,
                    "dataset": DATASET_NAME,
                    "domain": "education",
                    "source": "ncert",
                    "text": formatted_text,
                    "language": "en",
                    "metadata": metadata_json,
                    "added": None,
                    "created": None,
                    "version": VERSION,
                }
            )
        yield pd.DataFrame(rows)


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


def process_ncert(spark, input_path: str):
    """Read NCERT data, format text, normalize, write Parquet."""
    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    output_schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("hash", StringType(), False),
            StructField("dataset", StringType(), False),
            StructField("domain", StringType(), False),
            StructField("source", StringType(), False),
            StructField("text", StringType(), False),
            StructField("language", StringType(), False),
            StructField("metadata", StringType(), True),
            StructField("added", TimestampType(), True),
            StructField("created", TimestampType(), True),
            StructField("version", StringType(), True),
        ]
    )

    print(f"Reading from: {input_path}")
    df = spark.read.option("header", "true").csv(input_path)

    record_count = df.count()
    print(f"Records: {record_count:,}")

    num_partitions = max(1, min(400, record_count // 10_000))
    df_repartitioned = df.repartition(num_partitions)

    df_out = df_repartitioned.mapInPandas(ncert_map_func, schema=output_schema)

    # Assign sequential id: ncert_1, ncert_2, ...
    w = Window.orderBy(F.monotonically_increasing_id())
    df_out = df_out.withColumn(
        "id", F.concat(F.lit("ncert_"), F.row_number().over(w).cast("string"))
    )

    output_path = f"{OUTPUT_BASE}/source=ncert"
    print(f"Writing to: {output_path}")
    (df_out.write.mode("overwrite").option("compression", "zstd").parquet(output_path))
    print("✓ Completed NCERT processing")


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
    job.init("NCERT Data Normalization", args)

    print("=" * 80)
    print("NCERT Data Normalization - Starting")
    print(f"Input:  {input_path}")
    print(f"Output: {OUTPUT_BASE}/source=ncert")
    print("=" * 80)

    try:
        process_ncert(spark, input_path)
    except Exception as e:
        print(f"ERROR: {e}")
        raise

    print("\n" + "=" * 80)
    print("NCERT Data Normalization - Completed")
    print("=" * 80)
    job.commit()


if __name__ == "__main__":
    main()
