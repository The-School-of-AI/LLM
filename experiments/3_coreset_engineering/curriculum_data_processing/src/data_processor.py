import json
import logging
import os
import pyarrow.dataset as ds
from typing import Set, Dict, List, Any, Optional
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

logger = logging.getLogger("data_pipeline")


class RecordWriter:
    """Helper class to handle writing records to single, per-band, or sharded files."""

    def __init__(
        self,
        output_dir: str,
        mode: str,
        shard_size: Optional[int] = None,
        base_name: str = "output",
    ):
        self.output_dir = output_dir
        self.mode = mode
        self.shard_size = shard_size
        self.base_name = base_name

        self.current_fp = None
        self.record_count = 0
        self.shard_count = 0

        # For per_band mode with parallel processing
        self.band_fps = {}
        self.band_record_counts = defaultdict(int)
        self.band_shard_counts = defaultdict(int)

        os.makedirs(output_dir, exist_ok=True)

    def write(self, record: Dict[str, Any], band: Optional[str] = None):
        if self.mode == "single":
            self._write_single(record)
        elif self.mode == "sharded":
            self._write_sharded(record)
        else:  # per_band
            self._write_per_band(record, band)

    def _write_single(self, record: Dict[str, Any]):
        if self.current_fp is None:
            fname = os.path.join(self.output_dir, f"{self.base_name}.jsonl")
            self.current_fp = open(fname, "w", encoding="utf-8")
        self.current_fp.write(json.dumps(record) + "\n")

    def _write_sharded(self, record: Dict[str, Any]):
        if self.current_fp is None or (
            self.shard_size and self.record_count >= self.shard_size
        ):
            if self.current_fp:
                self.current_fp.close()
            fname = os.path.join(
                self.output_dir, f"{self.base_name}_shard_{self.shard_count}.jsonl"
            )
            self.current_fp = open(fname, "w", encoding="utf-8")
            self.shard_count += 1
            self.record_count = 0
        self.current_fp.write(json.dumps(record) + "\n")
        self.record_count += 1

    def _write_per_band(self, record: Dict[str, Any], band: str):
        if band not in self.band_fps or (
            self.shard_size and self.band_record_counts[band] >= self.shard_size
        ):
            if band in self.band_fps:
                self.band_fps[band].close()

            suffix = f"_shard_{self.band_shard_counts[band]}" if self.shard_size else ""
            fname = os.path.join(self.output_dir, f"{band.lower()}{suffix}.jsonl")
            self.band_fps[band] = open(fname, "w", encoding="utf-8")
            self.band_shard_counts[band] += 1
            self.band_record_counts[band] = 0

        self.band_fps[band].write(json.dumps(record) + "\n")
        self.band_record_counts[band] += 1

    def close(self):
        if self.current_fp:
            self.current_fp.close()
            self.current_fp = None
        for fp in self.band_fps.values():
            fp.close()
        self.band_fps.clear()


class DataProcessor:
    def __init__(self, config: Dict[str, Any], fs: Any):
        self.config = config
        self.fs = fs
        self.seen_hashes: Set[str] = set()
        self.stats = {
            "rows_read": 0,
            "rows_kept": 0,
            "rows_dropped": 0,
            "words_before": 0,
            "words_after": 0,
            "words_dropped": 0,
        }
        self.per_source_stats = defaultdict(
            lambda: {"read": 0, "kept": 0, "dropped": 0}
        )

    def process_all(
        self,
        target_bands: List[str],
        structure: Dict[str, List[str]],
        writer: RecordWriter,
        max_workers: int = 4,
    ):
        """Processes multiple bands in parallel using a ThreadPoolExecutor for I/O and transformation."""
        bucket = self.config["s3"]["bucket"]
        base_prefix = self.config["s3"]["base_prefix"]
        batch_size = self.config["processing"]["batch_size"]

        all_work_units = []
        for band in target_bands:
            sources_for_band = [
                src for src, bands in structure.items() if band in bands
            ]
            for source in sources_for_band:
                prefix = f"{base_prefix}/source={source}/bands/band={band}/"
                s3_path = f"{bucket}/{prefix}"
                all_work_units.append(
                    {
                        "band": band,
                        "source": source,
                        "s3_path": s3_path,
                        "full_s3_url": f"s3://{s3_path}",
                    }
                )

        logger.info(
            f"Starting parallel processing with {max_workers} workers for {len(all_work_units)} source-band combinations."
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_unit = {
                executor.submit(
                    self._fetch_and_transform_batches, unit, batch_size
                ): unit
                for unit in all_work_units
            }

            for future in concurrent.futures.as_completed(future_to_unit):
                unit = future_to_unit[future]
                try:
                    logger.info(
                        f"Processing Band: {unit['band']}, Source: {unit['source']}"
                    )
                    for transformed_batch in future.result():
                        self._deduplicate_and_write(
                            transformed_batch, writer, unit["band"], unit["source"]
                        )
                except Exception as e:
                    logger.error(
                        f"❌ Worker failed for {unit['source']} ({unit['band']}): {e}"
                    )

        writer.close()
        self.print_summary()

    def _fetch_and_transform_batches(self, unit: Dict[str, Any], batch_size: int):
        """Worker: Reads from S3 and transforms records (Parallel)."""
        transformed_batches = []
        try:
            dataset = ds.dataset(
                source=unit["s3_path"], format="parquet", filesystem=self.fs
            )
            for fragment in dataset.get_fragments():
                parquet_filename = os.path.basename(fragment.path)
                for batch in fragment.to_batches(batch_size=batch_size):
                    df = batch.to_pandas()
                    transformed_batches.append(
                        self._transform_batch(
                            df,
                            unit["band"],
                            unit["source"],
                            parquet_filename,
                            unit["full_s3_url"],
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to read from S3: {e}")
        return transformed_batches

    def _transform_batch(
        self, df: Any, band: str, source: str, filename: str, s3_url: str
    ) -> List[Dict[str, Any]]:
        """Worker: Renaming and provenance (Parallel)."""
        records = []
        rename_map = self.config["schema"]["rename_columns"]
        drop_cols = set(self.config["schema"]["drop_columns"])
        band_score_col = f"band_p_{band}"

        for _, row in df.iterrows():
            record = {
                "_raw_hash": str(row.get("hash", "")),
                "_word_count": int(row.get("word_count", 0)),
                "source_doc_id": filename,
                "source_url": s3_url,
                "band": band,
            }
            for col, val in row.items():
                if col in drop_cols:
                    continue
                if col == band_score_col:
                    record["band_score"] = val
                elif col.startswith("band_p_"):
                    continue
                elif col in rename_map:
                    record[rename_map[col]] = val
                else:
                    record[col] = val
            records.append(record)
        return records

    def _deduplicate_and_write(
        self,
        records: List[Dict[str, Any]],
        writer: RecordWriter,
        band: str,
        source: str,
    ):
        """Consumer: Global deduplication and file writing (Sequential)."""
        for record in records:
            self.stats["rows_read"] += 1
            if self.stats["rows_read"] % 1000 == 0:
                logger.info(
                    f"Progress: Processed {self.stats['rows_read']:,} rows... "
                    f"(Kept: {self.stats['rows_kept']:,}, Dropped: {self.stats['rows_dropped']:,})"
                )

            self.per_source_stats[source]["read"] += 1
            wc = record.pop("_word_count")
            self.stats["words_before"] += wc
            h = record.pop("_raw_hash")
            if h in self.seen_hashes:
                self.stats["rows_dropped"] += 1
                self.stats["words_dropped"] += wc
                self.per_source_stats[source]["dropped"] += 1
                continue
            self.seen_hashes.add(h)
            self.stats["rows_kept"] += 1
            self.stats["words_after"] += wc
            self.per_source_stats[source]["kept"] += 1
            writer.write(record, band=band)

    def print_summary(self):
        """Logs the final statistics."""
        logger.info("================= PROCESSING STATS =================")
        read = self.stats["rows_read"]
        kept = self.stats["rows_kept"]
        dropped = self.stats["rows_dropped"]
        dup_rate = (dropped / max(read, 1)) * 100
        logger.info(f"Total rows read        : {read:,}")
        logger.info(f"Unique rows kept       : {kept:,}")
        logger.info(f"Duplicate rows dropped : {dropped:,}")
        logger.info(f"Duplicate rate (%)     : {dup_rate:.2f}%")
        logger.info("--- Word Count Impact ---")
        logger.info(f"Words before dedup     : {self.stats['words_before']:,}")
        logger.info(f"Words after dedup      : {self.stats['words_after']:,}")
        logger.info("--- Per Source Breakdown ---")
        for src, s in self.per_source_stats.items():
            logger.info(
                f"{src:15s} | read={s['read']:,} kept={s['kept']:,} dropped={s['dropped']:,}"
            )
