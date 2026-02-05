"""Main tagging engine for processing datasets."""

import importlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..core.plugin import MetricPlugin
from ..utils.curriculum_loader import CurriculumConfig


class CurriculumTagger:
    """Main engine for tagging datasets with curriculum metadata.

    Auto-discovers and loads metrics from metrics_config.yaml.
    """

    def __init__(
        self,
        curriculum_path: str | Path,
        metrics: Optional[List[MetricPlugin]] = None,
        metrics_config_path: Optional[str | Path] = None,
    ):
        """Initialize tagger with curriculum config and metrics.

        Args:
            curriculum_path: Path to curriculum YAML file
            metrics: List of metric instances (applied in order).
                    If None, auto-loads from metrics_config.yaml
            metrics_config_path: Path to metrics config file.
                    Defaults to metrics_config.yaml in same dir as curriculum
        """
        self.config = CurriculumConfig(curriculum_path)
        self.plugins = (
            metrics
            if metrics is not None
            else self._load_metrics(curriculum_path, metrics_config_path)
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
        """Auto-discover and load metrics from config file."""
        # Determine metrics config path
        if metrics_config_path is None:
            curriculum_dir = Path(curriculum_path).parent
            metrics_config_path = curriculum_dir / "metrics_config.yaml"

        metrics_config_path = Path(metrics_config_path)

        if not metrics_config_path.exists():
            # Fallback to built-in defaults if no config file
            return self._get_builtin_defaults()

        # Load metrics config
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

            try:
                # Auto-import from metrics package
                # If module is explicitly provided in config, use it
                module_name = metric_def.get("module")

                if not module_name:
                    # Fallback convention: class DifficultyMetric in difficulty.py
                    # Simple heuristic: remove "Metric" and lowercase
                    # This covers standard cases like DifficultyMetric -> difficulty
                    # Complex cases should specify 'module' in config
                    base_name = class_name
                    if base_name.endswith("Metric"):
                        base_name = base_name[:-6]
                    # module_name = base_name.lower()

                    # Convert CamelCase to snake_case for module name
                    module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()

                module_path = f"curriculum_tags.metrics.{module_name}"

                module = importlib.import_module(module_path)
                metric_class = getattr(module, class_name)

                # Instantiate metric with curriculum config
                metric_instance = metric_class(self.config)
                loaded_metrics.append(metric_instance)

            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load metric {class_name}: {e}")
                continue
        if not loaded_metrics:
            raise ValueError("No valid metrics loaded from configuration.")
        return loaded_metrics

    def tag_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add curriculum tags to a single sample.

        Args:
            sample: Data sample with 'text' and other fields

        Returns:
            Sample with added 'curriculum_tags' field
        """
        # Initialize curriculum_tags if not present
        if "curriculum_tags" not in sample:
            sample["curriculum_tags"] = {}

        # Run plugins in order - each sees accumulated tags
        for plugin in self.plugins:
            try:
                tags = plugin.compute(sample)
                print(f"id: {sample['id']}, tags: {tags}")
                sample["curriculum_tags"][plugin.name] = tags

                # SKIP: If rejection_policy rejects, stop processing
                if plugin.name == "rejection_policy" and tags.get("rejected", False):
                    break

            except Exception as e:
                sample["curriculum_tags"][plugin.name] = {"error": str(e)}

        # Add metadata
        sample["curriculum_tags"]["version"] = self.config.version

        return sample

    def process_parquet(
        self,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Process parquet file and add curriculum tags.

        Args:
            input_path: Input parquet file
            output_path: Output parquet file
            batch_size: Number of rows per batch
            progress_callback: Optional callback for progress (total_rows)

        Returns:
            Statistics about processing (rows, errors, etc.)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Prepare metadata parquet output path
        metadata_path = output_path.with_suffix(".metadata.parquet")

        # Read parquet file
        parquet_file = pq.ParquetFile(input_path)

        # Process in batches
        output_batches = []
        metadata_batches = []
        total_rows = 0
        error_count = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Convert to Python dicts
            records = batch.to_pylist()

            # Tag each record
            tagged_records = []
            meta_records = []
            for record in records:
                try:
                    tagged = self.tag_sample(record)
                    tagged_records.append(tagged)
                except Exception as e:
                    # Keep original record but mark error
                    record["curriculum_tags"] = {
                        "version": self.config.version,
                        "error": str(e),
                    }
                    tagged_records.append(record)
                    error_count += 1

            # Prepare metadata records (id, curriculum_tags)
            for tagged in tagged_records:
                meta_records.append(
                    {
                        "id": tagged.get("id"),
                        "curriculum_tags": tagged.get("curriculum_tags", {}),
                    }
                )

            # Convert back to Arrow tables
            tagged_batch = pa.Table.from_pylist(tagged_records)
            meta_batch = pa.Table.from_pylist(meta_records)
            output_batches.append(tagged_batch)
            metadata_batches.append(meta_batch)

            total_rows += len(records)

            if progress_callback:
                progress_callback(total_rows)

        # Combine and write
        output_table = pa.concat_tables(output_batches)
        pq.write_table(output_table, output_path)

        # Combine and write metadata table
        metadata_table = pa.concat_tables(metadata_batches)
        pq.write_table(metadata_table, metadata_path)

        return {
            "total_rows": total_rows,
            "error_count": error_count,
            "output_file": str(output_path),
        }

    def process_parquet_s3(
        self,
        input_path: str,
        output_path: str,
        filesystem,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        """Process S3 parquet file and add curriculum tags + metadata parquet."""

        # ---- Validate paths (S3 equivalent of Path.exists()) ----
        if not filesystem.exists(input_path):
            raise FileNotFoundError(f"S3 input not found: {input_path}")

        output_prefix = output_path.rsplit("/", 1)[0]
        if not filesystem.exists(output_prefix):
            raise FileNotFoundError(f"S3 output prefix not found: {output_prefix}")

        # Metadata path (string version of .with_suffix())
        metadata_path = output_path.replace(".parquet", ".metadata.parquet")

        parquet_file = pq.ParquetFile(input_path, filesystem=filesystem)

        output_batches = []
        metadata_batches = []
        total_rows = 0
        error_count = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Convert to Python dicts
            records = batch.to_pylist()

            # Tag each record
            tagged_records = []
            meta_records = []
            for record in records:
                try:
                    tagged = self.tag_sample(record)
                    tagged_records.append(tagged)
                except Exception as e:
                    # Keep original record but mark error
                    record["curriculum_tags"] = {
                        "version": self.config.version,
                        "error": str(e),
                    }
                    tagged_records.append(record)
                    error_count += 1

            # Prepare metadata records (id, curriculum_tags)
            for tagged in tagged_records:
                meta_records.append(
                    {
                        "id": tagged.get("id"),
                        "curriculum_tags": tagged.get("curriculum_tags", {}),
                    }
                )

            # Convert back to Arrow tables
            tagged_batch = pa.Table.from_pylist(tagged_records)
            meta_batch = pa.Table.from_pylist(meta_records)
            output_batches.append(tagged_batch)
            metadata_batches.append(meta_batch)

            total_rows += len(records)

            if progress_callback:
                progress_callback(total_rows)

        # Combine and write
        output_table = pa.concat_tables(output_batches)
        metadata_table = pa.concat_tables(metadata_batches)

        # Atomic write main parquet to s3
        tmp_output = output_path + ".tmp"
        with filesystem.open(tmp_output, "wb") as f:
            pq.write_table(output_table, f)
        filesystem.mv(tmp_output, output_path)

        # Atomic write metadata parquet to s3
        tmp_meta = metadata_path + ".tmp"
        with filesystem.open(tmp_meta, "wb") as f:
            pq.write_table(metadata_table, f)
        filesystem.mv(tmp_meta, metadata_path)

        return {
            "total_rows": total_rows,
            "error_count": error_count,
            "output_file": output_path,
            "metadata_file": metadata_path,
        }

    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples in memory.

        Args:
            samples: List of data samples

        Returns:
            List of tagged samples
        """
        return [self.tag_sample(sample) for sample in samples]
