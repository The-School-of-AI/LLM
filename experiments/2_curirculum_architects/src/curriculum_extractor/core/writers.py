"""Output writers for metadata and rejection layers."""

import uuid as uuid_lib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Define the 5 optional columns for future metrics
OPTIONAL_METRIC_COLUMNS = [
    "opt_metric_1",
    "opt_metric_2",
    "opt_metric_3",
    "opt_metric_4",
    "opt_metric_5",
]


@dataclass
class MetadataRecord:
    """A single metadata record for the output layer."""

    uuid: str
    id: str  # Original record ID from JSONL
    file_path: str  # Source parquet file
    metrics: Dict[str, Any]  # Flattened metrics


@dataclass
class RejectionRecord:
    """A single rejection record."""

    uuid: str
    id: str  # Original record ID from JSONL
    file_path: str  # Source parquet file
    rejected_reason: str
    rejected_at: str  # Metric name that caused rejection


class OutputWriter:
    """Base class for output writers."""

    def __init__(
        self,
        output_path: str | Path,
        filesystem: Optional[Any] = None,
        partition_by_file: bool = True,
    ):
        """Initialize output writer.

        Args:
            output_path: Base path for output files
            filesystem: Optional s3fs filesystem for S3 support
            partition_by_file: Whether to partition output by source file
        """
        self.output_path = str(output_path)
        self.fs = filesystem
        self.partition_by_file = partition_by_file

    def _ensure_dir(self, path: str) -> None:
        """Ensure directory exists."""
        if self.fs:
            if not self.fs.exists(path):
                self.fs.makedirs(path, exist_ok=True)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)


class MetadataWriter(OutputWriter):
    """Writer for the metadata layer.

    Output schema:
    - uuid: string (generated UUID for this metadata record)
    - id: string (original record ID from JSONL)
    - file_path: string (source parquet file path)
    - <flattened_metrics>: various types (one column per metric value)
    - opt_metric_1..5: nullable columns for future metrics
    """

    def __init__(
        self,
        output_path: str | Path,
        filesystem: Optional[Any] = None,
        partition_by_file: bool = True,
        known_columns: Optional[List[str]] = None,
    ):
        """Initialize metadata writer.

        Args:
            output_path: Base path for metadata files
            filesystem: Optional s3fs for S3 support
            partition_by_file: Partition by source file name
            known_columns: List of expected metric column names
        """
        super().__init__(output_path, filesystem, partition_by_file)
        self.known_columns = known_columns or []
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 10000

    def _get_file_partition(self, source_file: str) -> str:
        """Extract file name for partitioning."""
        # Get just the file name without extension
        name = Path(source_file).stem
        # Clean up any problematic characters
        name = name.replace("/", "_").replace("\\", "_")
        return name

    def write_records(
        self,
        records: List[MetadataRecord],
        source_file: str,
    ) -> str:
        """Write metadata records to output.

        Args:
            records: List of metadata records to write
            source_file: Source file path (for partitioning)

        Returns:
            Path to written file
        """
        if not records:
            return ""

        # Build rows with flattened schema
        rows = []
        for record in records:
            row = {
                "uuid": record.uuid,
                "id": record.id,
                "file_path": record.file_path,
            }
            # Add all flattened metrics
            row.update(record.metrics)

            # Add optional columns (None for now)
            for opt_col in OPTIONAL_METRIC_COLUMNS:
                if opt_col not in row:
                    row[opt_col] = None

            rows.append(row)

        # Convert to Arrow table
        table = pa.Table.from_pylist(rows)

        # Determine output path
        if self.partition_by_file:
            partition = self._get_file_partition(source_file)
            output_dir = f"{self.output_path}/file_name={partition}"
            self._ensure_dir(output_dir)
            output_file = f"{output_dir}/metadata.parquet"
        else:
            self._ensure_dir(self.output_path)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            output_file = f"{self.output_path}/metadata_{timestamp}.parquet"

        # Write with atomic semantics
        if self.fs:
            tmp_file = output_file + ".tmp"
            with self.fs.open(tmp_file, "wb") as f:
                pq.write_table(table, f)
            self.fs.mv(tmp_file, output_file)
        else:
            tmp_file = output_file + ".tmp"
            pq.write_table(table, tmp_file)
            Path(tmp_file).rename(output_file)

        return output_file

    def append_record(self, record: MetadataRecord, source_file: str) -> Optional[str]:
        """Append a record to buffer, flush when full.

        Args:
            record: Metadata record to append
            source_file: Source file for partitioning

        Returns:
            Path to written file if buffer was flushed, None otherwise
        """
        self._buffer.append((record, source_file))

        if len(self._buffer) >= self._buffer_size:
            return self.flush()
        return None

    def flush(self) -> Optional[str]:
        """Flush buffered records to disk."""
        if not self._buffer:
            return None

        # Group by source file
        by_file: Dict[str, List[MetadataRecord]] = {}
        for record, source_file in self._buffer:
            if source_file not in by_file:
                by_file[source_file] = []
            by_file[source_file].append(record)

        self._buffer.clear()

        # Write each group
        written_files = []
        for source_file, records in by_file.items():
            path = self.write_records(records, source_file)
            if path:
                written_files.append(path)

        return written_files[0] if len(written_files) == 1 else None


class RejectionWriter(OutputWriter):
    """Writer for the rejection layer.

    Output schema:
    - uuid: string (generated UUID)
    - id: string (original record ID)
    - file_path: string (source parquet file)
    - rejected_reason: string (reason for rejection)
    - rejected_at: string (metric name that caused rejection)
    """

    def __init__(
        self,
        output_path: str | Path,
        filesystem: Optional[Any] = None,
        partition_by_file: bool = True,
    ):
        super().__init__(output_path, filesystem, partition_by_file)
        self._buffer: List[tuple] = []
        self._buffer_size = 10000

    def _get_file_partition(self, source_file: str) -> str:
        """Extract file name for partitioning."""
        name = Path(source_file).stem
        name = name.replace("/", "_").replace("\\", "_")
        return name

    def write_records(
        self,
        records: List[RejectionRecord],
        source_file: str,
    ) -> str:
        """Write rejection records to output.

        Args:
            records: List of rejection records
            source_file: Source file path (for partitioning)

        Returns:
            Path to written file
        """
        if not records:
            return ""

        rows = [
            {
                "uuid": r.uuid,
                "id": r.id,
                "file_path": r.file_path,
                "rejected_reason": r.rejected_reason,
                "rejected_at": r.rejected_at,
            }
            for r in records
        ]

        table = pa.Table.from_pylist(rows)

        # Determine output path
        if self.partition_by_file:
            partition = self._get_file_partition(source_file)
            output_dir = f"{self.output_path}/file_name={partition}"
            self._ensure_dir(output_dir)
            output_file = f"{output_dir}/rejections.parquet"
        else:
            self._ensure_dir(self.output_path)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            output_file = f"{self.output_path}/rejections_{timestamp}.parquet"

        # Write with atomic semantics
        if self.fs:
            tmp_file = output_file + ".tmp"
            with self.fs.open(tmp_file, "wb") as f:
                pq.write_table(table, f)
            self.fs.mv(tmp_file, output_file)
        else:
            tmp_file = output_file + ".tmp"
            pq.write_table(table, tmp_file)
            Path(tmp_file).rename(output_file)

        return output_file

    def append_record(self, record: RejectionRecord, source_file: str) -> Optional[str]:
        """Append a record to buffer."""
        self._buffer.append((record, source_file))

        if len(self._buffer) >= self._buffer_size:
            return self.flush()
        return None

    def flush(self) -> Optional[str]:
        """Flush buffered records."""
        if not self._buffer:
            return None

        by_file: Dict[str, List[RejectionRecord]] = {}
        for record, source_file in self._buffer:
            if source_file not in by_file:
                by_file[source_file] = []
            by_file[source_file].append(record)

        self._buffer.clear()

        written_files = []
        for source_file, records in by_file.items():
            path = self.write_records(records, source_file)
            if path:
                written_files.append(path)

        return written_files[0] if len(written_files) == 1 else None


def generate_uuid(record_id: str = "unknown", source_file: str = "unknown") -> str:
    """Generate a deterministic UUID for records.

    Uses UUID5 with a hash of record_id and source_file to ensure
    reproducibility across multiple runs.

    Args:
        record_id: Original record identifier from JSONL
        source_file: Source file path for tracking

    Returns:
        Deterministic UUID string based on record_id and source_file
    """
    # Create a stable namespace UUID for curriculum extraction
    namespace = uuid_lib.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_DNS

    # Create a deterministic seed from record_id and source_file
    seed_str = f"{record_id}#{source_file}"

    # Use UUID5 for deterministic generation
    return str(uuid_lib.uuid5(namespace, seed_str))
