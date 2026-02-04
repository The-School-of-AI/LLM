"""Metadata reader for the curriculum metadata layer."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds


class MetadataReader:
    """Reader for the curriculum metadata layer.

    Provides efficient access to the partitioned metadata parquet files
    with support for both local and S3 storage.

    The metadata layer is partitioned by file_name for efficient querying.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        filesystem: Optional[Any] = None,
    ):
        """Initialize metadata reader.

        Args:
            metadata_path: Base path to metadata layer
            filesystem: Optional s3fs filesystem for S3 support
        """
        self.metadata_path = str(metadata_path)
        self.fs = filesystem
        self._dataset: Optional[ds.Dataset] = None

    @property
    def dataset(self) -> ds.Dataset:
        """Lazy-load the Arrow dataset."""
        if self._dataset is None:
            if self.fs:
                self._dataset = ds.dataset(
                    self.metadata_path,
                    filesystem=self.fs,
                    format="parquet",
                    partitioning="hive",
                )
            else:
                self._dataset = ds.dataset(
                    self.metadata_path,
                    format="parquet",
                    partitioning="hive",
                )
        return self._dataset

    def get_schema(self) -> pa.Schema:
        """Get the schema of the metadata layer."""
        return self.dataset.schema

    def get_column_names(self) -> List[str]:
        """Get list of all column names."""
        return self.dataset.schema.names

    def get_partitions(self) -> List[str]:
        """Get list of partition values (file names)."""
        # Read the partitioning from the dataset
        partitions = set()
        for fragment in self.dataset.get_fragments():
            # Extract partition value from path
            path = str(fragment.path)
            if "file_name=" in path:
                start = path.find("file_name=") + len("file_name=")
                end = path.find("/", start)
                if end == -1:
                    partition = path[start:]
                else:
                    partition = path[start:end]
                partitions.add(partition)
        return sorted(partitions)

    def count_rows(self, filter_expr: Optional[Any] = None) -> int:
        """Count total rows in the dataset.

        Args:
            filter_expr: Optional pyarrow filter expression

        Returns:
            Number of rows
        """
        if filter_expr is not None:
            return self.dataset.filter(filter_expr).count_rows()
        return self.dataset.count_rows()

    def read_all(
        self,
        columns: Optional[List[str]] = None,
        filter_expr: Optional[Any] = None,
    ) -> pa.Table:
        """Read entire dataset into memory.

        Args:
            columns: Optional list of columns to read
            filter_expr: Optional filter expression

        Returns:
            Arrow table with data
        """
        scanner = self.dataset.scanner(
            columns=columns,
            filter=filter_expr,
        )
        return scanner.to_table()

    def read_partition(
        self,
        partition_value: str,
        columns: Optional[List[str]] = None,
    ) -> pa.Table:
        """Read data from a specific partition.

        Args:
            partition_value: The file_name partition value
            columns: Optional list of columns to read

        Returns:
            Arrow table with partition data
        """
        filter_expr = ds.field("file_name") == partition_value
        return self.read_all(columns=columns, filter_expr=filter_expr)

    def iter_batches(
        self,
        batch_size: int = 10000,
        columns: Optional[List[str]] = None,
        filter_expr: Optional[Any] = None,
    ) -> Iterator[pa.RecordBatch]:
        """Iterate over dataset in batches.

        Args:
            batch_size: Number of rows per batch
            columns: Optional list of columns
            filter_expr: Optional filter expression

        Yields:
            Arrow RecordBatches
        """
        scanner = self.dataset.scanner(
            columns=columns,
            filter=filter_expr,
            batch_size=batch_size,
        )
        for batch in scanner.to_batches():
            yield batch

    def sample(
        self,
        n: int = 100,
        columns: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> pa.Table:
        """Get a random sample of records.

        Args:
            n: Number of records to sample
            columns: Optional list of columns
            seed: Random seed for reproducibility

        Returns:
            Arrow table with sampled records
        """
        import random

        if seed is not None:
            random.seed(seed)

        total_rows = self.count_rows()
        if total_rows <= n:
            return self.read_all(columns=columns)

        # Sample row indices
        indices = sorted(random.sample(range(total_rows), n))

        # Read in batches and collect sampled rows
        sampled_rows = []
        current_idx = 0
        idx_set = set(indices)

        for batch in self.iter_batches(columns=columns):
            batch_size = len(batch)
            # Check if any sampled indices fall in this batch
            for i in range(batch_size):
                if current_idx + i in idx_set:
                    sampled_rows.append(batch.slice(i, 1))
            current_idx += batch_size

            # Early exit if we've got all samples
            if len(sampled_rows) >= n:
                break

        if sampled_rows:
            return pa.Table.from_batches(sampled_rows)
        return pa.table({})

    def get_record_by_id(
        self,
        record_id: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific record by its ID.

        Args:
            record_id: The record ID to look up
            columns: Optional list of columns

        Returns:
            Record as dictionary or None if not found
        """
        filter_expr = ds.field("id") == record_id
        table = self.read_all(columns=columns, filter_expr=filter_expr)

        if len(table) == 0:
            return None
        return table.to_pylist()[0]

    def get_record_by_uuid(
        self,
        uuid: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific record by its UUID.

        Args:
            uuid: The UUID to look up
            columns: Optional list of columns

        Returns:
            Record as dictionary or None if not found
        """
        filter_expr = ds.field("uuid") == uuid
        table = self.read_all(columns=columns, filter_expr=filter_expr)

        if len(table) == 0:
            return None
        return table.to_pylist()[0]

    def query(
        self,
        filter_expr: Any,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Query the metadata with a filter expression.

        Args:
            filter_expr: PyArrow filter expression
            columns: Optional list of columns
            limit: Optional maximum number of rows

        Returns:
            Filtered Arrow table

        Example:
            >>> reader.query(
            ...     (ds.field("difficulty_score") > 0.5) &
            ...     (ds.field("modality_primary_modality") == "code")
            ... )
        """
        scanner = self.dataset.scanner(
            columns=columns,
            filter=filter_expr,
        )

        if limit:
            return scanner.head(limit)
        return scanner.to_table()

    def to_pandas(
        self,
        columns: Optional[List[str]] = None,
        filter_expr: Optional[Any] = None,
    ):
        """Convert to pandas DataFrame.

        Args:
            columns: Optional list of columns
            filter_expr: Optional filter expression

        Returns:
            Pandas DataFrame
        """
        table = self.read_all(columns=columns, filter_expr=filter_expr)
        return table.to_pandas()


class RejectionReader(MetadataReader):
    """Reader for the rejection layer.

    Schema:
    - uuid: string
    - id: string
    - file_path: string
    - rejected_reason: string
    - rejected_at: string (metric name)
    """

    def get_rejection_counts_by_metric(self) -> Dict[str, int]:
        """Get rejection counts grouped by metric name."""
        table = self.read_all(columns=["rejected_at"])
        df = table.to_pandas()
        return df["rejected_at"].value_counts().to_dict()

    def get_rejection_counts_by_reason(self) -> Dict[str, int]:
        """Get rejection counts grouped by reason."""
        table = self.read_all(columns=["rejected_reason"])
        df = table.to_pandas()
        return df["rejected_reason"].value_counts().to_dict()

    def get_rejections_for_file(self, file_path: str) -> pa.Table:
        """Get all rejections for a specific source file."""
        filter_expr = ds.field("file_path") == file_path
        return self.read_all(filter_expr=filter_expr)
