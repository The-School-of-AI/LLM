import logging
from typing import Any, Dict, List

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logger = logging.getLogger("glue_logger")


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
