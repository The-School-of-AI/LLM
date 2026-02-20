import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from src.processor import SparkDataProcessor
from src.utils import (
    CheckpointManager,
    discover_sources_from_s3,
    load_config,
    setup_glue_logger,
)


def main():
    # 1. Initialize Glue Context
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "config_path"])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    logger = setup_glue_logger()
    logger.info("Initializing PySpark Glue Job for Curriculum Data")

    # 2. Load Config
    config_path = args["config_path"]
    config = load_config(config_path)

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
