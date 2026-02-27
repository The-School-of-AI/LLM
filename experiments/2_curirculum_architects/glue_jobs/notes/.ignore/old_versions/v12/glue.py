import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# Glue args
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "INPUT_PATH",       # s3://bucket/raw/dolma/*.json.gz
        "OUTPUT_PATH",      # s3://bucket/parquet/dolma/
        "DOMAIN",           # e.g. web
        "EXTERNAL_SOURCE",  # e.g. books
        "VERSION",          # e.g. 1.7
    ],
)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

input_path = args["INPUT_PATH"]
output_path = args["OUTPUT_PATH"]
domain = args["DOMAIN"]
external_source = args["EXTERNAL_SOURCE"]
version = args["VERSION"]

# Optional: define schema explicitly if you know it; example below
schema = (
    StructType()
    .add("id", StringType())
    .add("text", StringType())
    .add("metadata", StringType())  # or MapType/String JSON, adjust if needed
    .add("added", TimestampType())
    .add("created", TimestampType())
)

# Read Dolma JSONL.GZ from S3
df = (
    spark.read
    .schema(schema)  # or comment this out to infer schema
    .option("compression", "gzip")
    .json(input_path)
)

# Transform to match your DuckDB output
df_out = (
    df
    .withColumn("hash", F.sha2(F.col("text"), 256))
    .withColumn("dataset", F.lit("dolma"))
    .withColumn("domain", F.lit(domain))
    .withColumn("source", F.lit(external_source))
    .withColumn("language", F.lit("en"))
    .withColumn("metadata", F.col("metadata").cast("string"))
    .withColumn("version", F.lit(version))
    # Reorder/select columns explicitly
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

# Adjust partitioning/num output files as needed
# Example: partition by domain & source
(
    df_out
    .repartition(400)  # tune based on cluster size & data volume
    .write
    .mode("overwrite")
    .option("compression", "zstd")  # or "snappy" if you want faster writes
    .parquet(output_path)
)

job.commit()

