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
        self.plugins = metrics if metrics is not None else self._load_metrics(curriculum_path, metrics_config_path)

    def _get_builtin_defaults(self) -> List[MetricPlugin]:
        """Return list of essential default metrics if no config found."""
        from ..metrics.difficulty import DifficultyMetric
        from ..metrics.modality import ModalityMetric
        from ..metrics.readability import ReadabilityMetric

        return [DifficultyMetric(self.config), ModalityMetric(self.config), ReadabilityMetric(self.config)]

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
                sample["curriculum_tags"][plugin.name] = tags
            except Exception as e:
                sample["curriculum_tags"][plugin.name] = {"error": str(e)}

        # Add metadata
        sample["curriculum_tags"]["version"] = self.config.version

        return sample

    def process_parquet(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
        output_csv_path: Optional[str | Path] = None,
        rejected_csv_path: Optional[str | Path] = None,
        write_parquet: bool = False,
    ) -> Dict[str, Any]:
        """Process parquet file and add curriculum tags.

        By default only CSV output is written (main + rejected). Parquet output
        is optional and when enabled is pass-through (original rows, no curriculum_tags).

        Args:
            input_path: Input parquet file
            output_path: Output parquet file (required only when write_parquet=True)
            batch_size: Number of rows per batch
            progress_callback: Optional callback for progress (total_rows)
            output_csv_path: If set, write flat main CSV and rejected CSV
            rejected_csv_path: If set, path for rejected log CSV; else derived from output_csv_path
            write_parquet: If True, write one Parquet at output_path with original rows
                (no curriculum_tags). No metadata Parquet is written. Default False.

        Returns:
            Statistics: total_rows, error_count; output_file only if write_parquet;
            main_csv_path, rejected_csv_path, main_csv_rows, rejected_csv_rows if CSV written.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if write_parquet and output_path is None:
            raise ValueError("output_path is required when write_parquet=True")

        if output_path is not None:
            output_path = Path(output_path)

        # Read parquet file
        parquet_file = pq.ParquetFile(input_path)

        all_tagged_records: List[Dict[str, Any]] = []
        all_original_records: List[Dict[str, Any]] = []  # rows without curriculum_tags
        total_rows = 0
        error_count = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            records = batch.to_pylist()

            tagged_records = []
            for record in records:
                try:
                    tagged = self.tag_sample(record)
                    tagged_records.append(tagged)
                except Exception as e:
                    record["curriculum_tags"] = {
                        "version": self.config.version,
                        "error": str(e),
                    }
                    tagged_records.append(record)
                    error_count += 1

            all_tagged_records.extend(tagged_records)
            # Pass-through: rows without curriculum_tags for optional Parquet output
            for tagged in tagged_records:
                all_original_records.append({k: v for k, v in tagged.items() if k != "curriculum_tags"})

            total_rows += len(records)

            if progress_callback:
                progress_callback(total_rows)

        result: Dict[str, Any] = {
            "total_rows": total_rows,
            "error_count": error_count,
        }

        if write_parquet and output_path is not None:
            output_table = pa.Table.from_pylist(all_original_records)
            pq.write_table(output_table, output_path)
            result["output_file"] = str(output_path)

        if output_csv_path is not None:
            from ..output.csv_writer import write_csv_output

            csv_stats = write_csv_output(
                all_tagged_records,
                file_path=str(input_path),
                output_csv_path=Path(output_csv_path),
                rejected_csv_path=Path(rejected_csv_path) if rejected_csv_path is not None else None,
            )
            result["main_csv_path"] = csv_stats["main_csv_path"]
            result["rejected_csv_path"] = csv_stats["rejected_csv_path"]
            result["main_csv_rows"] = csv_stats["main_row_count"]
            result["rejected_csv_rows"] = csv_stats["rejected_row_count"]

        return result

    def process_parquet_s3(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        filesystem=None,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[int], None]] = None,
        output_csv_path: Optional[str | Path] = None,
        rejected_csv_path: Optional[str | Path] = None,
        write_parquet: bool = False,
    ) -> dict:
        """Process S3 parquet file; by default only CSV output (main + rejected).

        When write_parquet=True, writes one pass-through Parquet (original rows,
        no curriculum_tags) to output_path. No metadata Parquet is written.
        """

        if not filesystem.exists(input_path):
            raise FileNotFoundError(f"S3 input not found: {input_path}")

        if write_parquet and output_path is None:
            raise ValueError("output_path is required when write_parquet=True")

        if write_parquet and output_path is not None:
            output_prefix = output_path.rsplit("/", 1)[0]
            if not filesystem.exists(output_prefix):
                raise FileNotFoundError(f"S3 output prefix not found: {output_prefix}")

        parquet_file = pq.ParquetFile(input_path, filesystem=filesystem)

        all_tagged_records: List[Dict[str, Any]] = []
        all_original_records: List[Dict[str, Any]] = []
        total_rows = 0
        error_count = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            records = batch.to_pylist()

            tagged_records = []
            for record in records:
                try:
                    tagged = self.tag_sample(record)
                    tagged_records.append(tagged)
                except Exception as e:
                    record["curriculum_tags"] = {
                        "version": self.config.version,
                        "error": str(e),
                    }
                    tagged_records.append(record)
                    error_count += 1

            all_tagged_records.extend(tagged_records)
            for tagged in tagged_records:
                all_original_records.append({k: v for k, v in tagged.items() if k != "curriculum_tags"})

            total_rows += len(records)

            if progress_callback:
                progress_callback(total_rows)

        result: Dict[str, Any] = {
            "total_rows": total_rows,
            "error_count": error_count,
        }

        if write_parquet and output_path is not None:
            output_table = pa.Table.from_pylist(all_original_records)
            tmp_output = output_path + ".tmp"
            with filesystem.open(tmp_output, "wb") as f:
                pq.write_table(output_table, f)
            filesystem.mv(tmp_output, output_path)
            result["output_file"] = output_path

        if output_csv_path is not None:
            from ..output.csv_writer import write_csv_output

            csv_stats = write_csv_output(
                all_tagged_records,
                file_path=input_path,
                output_csv_path=Path(output_csv_path),
                rejected_csv_path=Path(rejected_csv_path) if rejected_csv_path is not None else None,
            )
            result["main_csv_path"] = csv_stats["main_csv_path"]
            result["rejected_csv_path"] = csv_stats["rejected_csv_path"]
            result["main_csv_rows"] = csv_stats["main_row_count"]
            result["rejected_csv_rows"] = csv_stats["rejected_row_count"]

        return result

    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples in memory.

        Args:
            samples: List of data samples

        Returns:
            List of tagged samples
        """
        return [self.tag_sample(sample) for sample in samples]
