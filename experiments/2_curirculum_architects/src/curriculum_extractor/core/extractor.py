"""Main extraction engine for processing datasets.

Key Design Principles:
- Records are READ-ONLY: never modified during extraction
- No plugin chaining: each metric sees original data only
- Early rejection: rejected records skip remaining metrics
- Level-based execution: metrics at same level can run in parallel
"""

import importlib
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import yaml

from ..utils.curriculum_loader import CurriculumConfig
from .plugin import MetricPlugin, ReadOnlyRecord
from .state_manager import StateManager
from .writers import (
    MetadataRecord,
    MetadataWriter,
    RejectionRecord,
    RejectionWriter,
    generate_uuid,
)


class MetricTiming:
    """Track timing information for metrics."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)

    def record(self, metric_name: str, elapsed: float):
        """Record a timing measurement."""
        self.timings[metric_name].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics per metric."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "total_seconds": sum(times),
                    "mean_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
        return stats


class CurriculumExtractor:
    """Main engine for extracting curriculum metadata from datasets.

    Features:
    - Auto-discovers and loads metrics from metrics_config.yaml
    - Read-only record handling (immutable)
    - Early rejection: stops processing on first rejection
    - Level-based execution for potential parallelism
    - State management for incremental processing
    - Outputs flattened columns (no nested structs)

    Note: Band assignment is NOT performed during extraction.
    Use the band_assignment script to assign bands after extraction.
    """

    def __init__(
        self,
        curriculum_path: str | Path,
        metrics: Optional[List[MetricPlugin]] = None,
        metrics_config_path: Optional[str | Path] = None,
        state_manager: Optional[StateManager] = None,
        metadata_output_path: Optional[str | Path] = None,
        rejection_output_path: Optional[str | Path] = None,
        filesystem: Optional[Any] = None,
        track_timing: bool = False,
    ):
        """Initialize extractor with curriculum config and metrics.

        Args:
            curriculum_path: Path to curriculum YAML file
            metrics: List of metric instances (applied in order by level).
                    If None, auto-loads from metrics_config.yaml
            metrics_config_path: Path to metrics config file.
                    Defaults to metrics_config.yaml in same dir as curriculum
            state_manager: Optional state manager for incremental processing
            metadata_output_path: Path for metadata layer output
            rejection_output_path: Path for rejection layer output
            filesystem: Optional s3fs filesystem for S3 support
            track_timing: Whether to track timing per metric
        """
        self.config = CurriculumConfig(curriculum_path)
        self.plugins = (
            metrics
            if metrics is not None
            else self._load_metrics(curriculum_path, metrics_config_path)
        )

        # Sort plugins by level for ordered execution
        self.plugins = sorted(self.plugins, key=lambda p: p.level)

        # Group plugins by level for potential parallel execution
        self._plugins_by_level: Dict[int, List[MetricPlugin]] = defaultdict(list)
        for plugin in self.plugins:
            self._plugins_by_level[plugin.level].append(plugin)

        self.state_manager = state_manager
        self.fs = filesystem
        self.track_timing = track_timing
        self.timing = MetricTiming() if track_timing else None

        # Initialize writers if paths provided
        self.metadata_writer = (
            MetadataWriter(metadata_output_path, filesystem)
            if metadata_output_path
            else None
        )
        self.rejection_writer = (
            RejectionWriter(rejection_output_path, filesystem)
            if rejection_output_path
            else None
        )

    def _get_builtin_defaults(self) -> List[MetricPlugin]:
        """Return list of essential default metrics if no config found."""
        from ..metrics.difficulty import DifficultyMetric
        from ..metrics.modality import ModalityMetric
        from ..metrics.readability import ReadabilityMetric

        return [
            DifficultyMetric(self.config),
            ModalityMetric(self.config),
            ReadabilityMetric(self.config),
        ]

    def _load_metrics(
        self, curriculum_path: str | Path, metrics_config_path: Optional[str | Path]
    ) -> List[MetricPlugin]:
        """Auto-discover and load metrics from config file.

        Note: BandAssignmentMetric is excluded as band assignment
        happens in a separate post-processing step.
        """
        # Determine metrics config path
        if metrics_config_path is None:
            curriculum_dir = Path(curriculum_path).parent
            metrics_config_path = curriculum_dir / "metrics_config.yaml"

        metrics_config_path = Path(metrics_config_path)

        if not metrics_config_path.exists():
            return self._get_builtin_defaults()

        with open(metrics_config_path) as f:
            config = yaml.safe_load(f)

        metrics_list = config.get("metrics", [])
        loaded_metrics = []

        for metric_def in metrics_list:
            if not metric_def.get("enabled", True):
                continue

            class_name = metric_def.get("class")
            if not class_name:
                continue

            # Skip band assignment - it's done in post-processing
            if class_name == "BandAssignmentMetric":
                continue

            try:
                module_name = metric_def.get("module")

                if not module_name:
                    base_name = class_name
                    if base_name.endswith("Metric"):
                        base_name = base_name[:-6]
                    module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()

                module_path = f"curriculum_extractor.metrics.{module_name}"

                module = importlib.import_module(module_path)
                metric_class = getattr(module, class_name)

                metric_instance = metric_class(self.config)

                # Override level if specified in config
                if "level" in metric_def:
                    metric_instance.level = metric_def["level"]

                loaded_metrics.append(metric_instance)

            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load metric {class_name}: {e}")
                continue

        if not loaded_metrics:
            raise ValueError("No valid metrics loaded from configuration.")
        return loaded_metrics

    def extract_record(
        self,
        record: Dict[str, Any],
        source_file: str = "unknown",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[RejectionRecord]]:
        """Extract metadata from a single record.

        The record is wrapped in ReadOnlyRecord to prevent modifications.
        Metrics are executed in level order. If any metric rejects,
        processing stops immediately and the rejection is returned.

        Args:
            record: Data record with 'text' and other fields (NOT modified)
            source_file: Source file path for tracking

        Returns:
            Tuple of (metadata_dict, rejection_record)
            - If not rejected: (metadata_dict, None)
            - If rejected: (None, RejectionRecord)
        """
        record_id = record.get("id", "unknown")
        record_uuid = generate_uuid(record_id, source_file)

        # Wrap record as read-only to prevent modifications
        readonly_record = ReadOnlyRecord(record)

        # Collect all flattened metrics
        all_metrics: Dict[str, Any] = {}

        # Process metrics level by level
        sorted_levels = sorted(self._plugins_by_level.keys())

        for level in sorted_levels:
            level_plugins = self._plugins_by_level[level]

            # Execute plugins at this level
            # (Could be parallelized in future, but sequential for now)
            for plugin in level_plugins:
                try:
                    start_time = time.perf_counter() if self.timing else 0

                    result = plugin.extract(readonly_record)

                    if self.timing:
                        elapsed = time.perf_counter() - start_time
                        self.timing.record(plugin.name, elapsed)

                    # Check for rejection - stop immediately
                    if result.rejected:
                        return None, RejectionRecord(
                            uuid=record_uuid,
                            id=record_id,
                            file_path=source_file,
                            rejected_reason=result.rejection_reason or "Unknown",
                            rejected_at=plugin.name,
                        )

                    # Flatten metrics for output
                    flattened = plugin.flatten_metrics(result.metrics)
                    all_metrics.update(flattened)

                except Exception as e:
                    # On error, add error info but continue
                    all_metrics[f"{plugin.name}_error"] = str(e)

        # Add metadata version
        all_metrics["curriculum_version"] = self.config.version

        return all_metrics, None

    def process_parquet(
        self,
        input_path: str | Path,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Process local parquet file and extract metadata.

        Args:
            input_path: Input parquet file
            batch_size: Number of rows per batch
            progress_callback: Optional callback for progress

        Returns:
            Statistics about processing
        """
        input_path = Path(input_path)
        source_file = str(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Mark as in progress if state manager exists
        if self.state_manager:
            if self.state_manager.is_completed(source_file):
                return {
                    "status": "skipped",
                    "reason": "already_processed",
                    "file": source_file,
                }
            self.state_manager.mark_in_progress(source_file)

        try:
            parquet_file = pq.ParquetFile(input_path)

            metadata_records: List[MetadataRecord] = []
            rejection_records: List[RejectionRecord] = []

            total_rows = 0
            error_count = 0

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                records = batch.to_pylist()

                for record in records:
                    metadata, rejection = self.extract_record(record, source_file)

                    if rejection:
                        rejection_records.append(rejection)
                    else:
                        record_id = record.get("id", "unknown")
                        metadata_records.append(
                            MetadataRecord(
                                uuid=generate_uuid(record_id, source_file),
                                id=record_id,
                                file_path=source_file,
                                metrics=metadata,
                            )
                        )

                    total_rows += 1

                if progress_callback:
                    progress_callback(total_rows)

            # Write metadata layer
            if self.metadata_writer and metadata_records:
                self.metadata_writer.write_records(metadata_records, source_file)

            # Write rejection layer
            if self.rejection_writer and rejection_records:
                self.rejection_writer.write_records(rejection_records, source_file)

            # Mark as completed
            if self.state_manager:
                self.state_manager.mark_completed(
                    source_file,
                    rows_processed=len(metadata_records),
                    rows_rejected=len(rejection_records),
                )

            result = {
                "status": "completed",
                "total_rows": total_rows,
                "processed_rows": len(metadata_records),
                "rejected_rows": len(rejection_records),
                "error_count": error_count,
            }

            if self.timing:
                result["timing"] = self.timing.get_stats()

            return result

        except Exception as e:
            if self.state_manager:
                self.state_manager.mark_failed(source_file, str(e))
            raise

    def process_parquet_s3(
        self,
        input_path: str,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Process S3 parquet file and extract metadata.

        Args:
            input_path: S3 path to input parquet (e.g., s3://bucket/path/file.parquet)
            batch_size: Number of rows per batch
            progress_callback: Optional progress callback

        Returns:
            Statistics about processing
        """
        if not self.fs:
            raise ValueError(
                "Filesystem not configured. Pass filesystem to constructor."
            )

        source_file = input_path

        if not self.fs.exists(input_path):
            raise FileNotFoundError(f"S3 input not found: {input_path}")

        # Mark as in progress
        if self.state_manager:
            if self.state_manager.is_completed(source_file):
                return {
                    "status": "skipped",
                    "reason": "already_processed",
                    "file": source_file,
                }
            self.state_manager.mark_in_progress(source_file)

        try:
            parquet_file = pq.ParquetFile(input_path, filesystem=self.fs)

            metadata_records: List[MetadataRecord] = []
            rejection_records: List[RejectionRecord] = []

            total_rows = 0

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                records = batch.to_pylist()

                for record in records:
                    metadata, rejection = self.extract_record(record, source_file)

                    if rejection:
                        rejection_records.append(rejection)
                    else:
                        record_id = record.get("id", "unknown")
                        metadata_records.append(
                            MetadataRecord(
                                uuid=generate_uuid(record_id, source_file),
                                id=record_id,
                                file_path=source_file,
                                metrics=metadata,
                            )
                        )

                    total_rows += 1

                if progress_callback:
                    progress_callback(total_rows)

            # Write metadata layer
            if self.metadata_writer and metadata_records:
                self.metadata_writer.write_records(metadata_records, source_file)

            # Write rejection layer
            if self.rejection_writer and rejection_records:
                self.rejection_writer.write_records(rejection_records, source_file)

            # Mark as completed
            if self.state_manager:
                self.state_manager.mark_completed(
                    source_file,
                    rows_processed=len(metadata_records),
                    rows_rejected=len(rejection_records),
                )

            result = {
                "status": "completed",
                "total_rows": total_rows,
                "processed_rows": len(metadata_records),
                "rejected_rows": len(rejection_records),
            }

            if self.timing:
                result["timing"] = self.timing.get_stats()

            return result

        except Exception as e:
            if self.state_manager:
                self.state_manager.mark_failed(source_file, str(e))
            raise

    def process_batch(
        self,
        records: List[Dict[str, Any]],
        source_file: str = "batch",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a batch of records in memory.

        Args:
            records: List of data records (NOT modified)
            source_file: Source identifier

        Returns:
            Tuple of (processed_records, rejected_records)
        """
        processed = []
        rejected = []

        for record in records:
            metadata, rejection = self.extract_record(record, source_file)

            if rejection:
                rejected.append(
                    {
                        "uuid": rejection.uuid,
                        "id": rejection.id,
                        "file_path": rejection.file_path,
                        "rejected_reason": rejection.rejected_reason,
                        "rejected_at": rejection.rejected_at,
                    }
                )
            else:
                record_id = record.get("id", "unknown")
                processed.append(
                    {
                        "uuid": generate_uuid(record_id, source_file),
                        "id": record_id,
                        "file_path": source_file,
                        **metadata,
                    }
                )

        return processed, rejected

    def get_timing_stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get timing statistics if tracking is enabled."""
        if self.timing:
            return self.timing.get_stats()
        return None


def create_extractor_from_config(
    curriculum_path: str | Path,
    state_path: Optional[str | Path] = None,
    metadata_output_path: Optional[str | Path] = None,
    rejection_output_path: Optional[str | Path] = None,
    s3_bucket: Optional[str] = None,
    track_timing: bool = False,
) -> CurriculumExtractor:
    """Factory function to create an extractor with common configuration.

    Args:
        curriculum_path: Path to curriculum YAML
        state_path: Path for state management
        metadata_output_path: Path for metadata layer
        rejection_output_path: Path for rejection layer
        s3_bucket: Optional S3 bucket for S3 mode
        track_timing: Whether to track timing per metric

    Returns:
        Configured CurriculumExtractor instance
    """
    filesystem = None
    if s3_bucket:
        import s3fs

        filesystem = s3fs.S3FileSystem()

    state_manager = None
    if state_path:
        state_manager = StateManager(state_path, filesystem)

    return CurriculumExtractor(
        curriculum_path=curriculum_path,
        state_manager=state_manager,
        metadata_output_path=metadata_output_path,
        rejection_output_path=rejection_output_path,
        filesystem=filesystem,
        track_timing=track_timing,
    )
