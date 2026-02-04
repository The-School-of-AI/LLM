"""Deterministic batch creation for training data loading.

This module provides reproducible batch creation from the metadata layer,
suitable for training 70B LLMs with coreset-based data loading.

Key features:
- Deterministic at record level (same seed = same record order)
- Deterministic at batch level (same seed + batch_number = same batch)
- Auto-incrementing batch number support
- Efficient streaming without loading full dataset
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import xxhash

from .reader import MetadataReader


@dataclass
class BatchConfig:
    """Configuration for batch creation.

    Attributes:
        batch_size: Number of records per batch
        seed: Random seed for deterministic ordering
        shuffle: Whether to shuffle records (if False, uses natural order)
        columns: Optional list of columns to include
        filter_expr: Optional filter expression for subsetting
        stratify_by: Optional column to stratify batches by (e.g., 'band_assignment_band')
    """

    batch_size: int = 1024
    seed: int = 42
    shuffle: bool = True
    columns: Optional[List[str]] = None
    filter_expr: Optional[Any] = None
    stratify_by: Optional[str] = None


@dataclass
class BatchState:
    """State for tracking batch iteration.

    Stored to enable resumption and deterministic replay.
    """

    current_batch: int = 0
    total_records: int = 0
    total_batches: int = 0
    seed: int = 42
    config_hash: str = ""


class BatchCreator:
    """Creates deterministic batches from metadata layer.

    Provides reproducible batch creation for training data loading.
    Given the same seed and batch number, returns the same records.

    Implementation uses xxhash for fast, deterministic ordering:
    1. Each record gets a deterministic hash: xxhash(seed + uuid)
    2. Records are sorted by hash
    3. Batches are sliced from sorted order

    This ensures:
    - Same seed → same global ordering
    - Same seed + batch_number → same batch content
    - Efficient: no need to load full dataset for a single batch
    """

    STATE_FILE_NAME = "_batch_state.json"

    def __init__(
        self,
        reader: MetadataReader,
        config: BatchConfig,
        state_path: Optional[str | Path] = None,
    ):
        """Initialize batch creator.

        Args:
            reader: MetadataReader instance
            config: Batch configuration
            state_path: Optional path for persisting batch state
        """
        self.reader = reader
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self._state: Optional[BatchState] = None
        self._sorted_uuids: Optional[List[str]] = None

    @property
    def state(self) -> BatchState:
        """Get current batch state."""
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _get_config_hash(self) -> str:
        """Get hash of config for state validation."""
        config_dict = {
            "batch_size": self.config.batch_size,
            "seed": self.config.seed,
            "shuffle": self.config.shuffle,
            "columns": self.config.columns,
            "stratify_by": self.config.stratify_by,
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _load_state(self) -> BatchState:
        """Load state from disk if available."""
        if self.state_path:
            state_file = self.state_path / self.STATE_FILE_NAME
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data = json.load(f)
                    # Validate config hasn't changed
                    if data.get("config_hash") == self._get_config_hash():
                        return BatchState(
                            current_batch=data.get("current_batch", 0),
                            total_records=data.get("total_records", 0),
                            total_batches=data.get("total_batches", 0),
                            seed=data.get("seed", self.config.seed),
                            config_hash=data.get("config_hash", ""),
                        )
                except Exception:
                    pass

        # Initialize fresh state
        total = self._count_total_records()
        return BatchState(
            current_batch=0,
            total_records=total,
            total_batches=(total + self.config.batch_size - 1)
            // self.config.batch_size,
            seed=self.config.seed,
            config_hash=self._get_config_hash(),
        )

    def _save_state(self) -> None:
        """Save state to disk."""
        if self.state_path and self._state:
            self.state_path.mkdir(parents=True, exist_ok=True)
            state_file = self.state_path / self.STATE_FILE_NAME
            with open(state_file, "w") as f:
                json.dump(
                    {
                        "current_batch": self._state.current_batch,
                        "total_records": self._state.total_records,
                        "total_batches": self._state.total_batches,
                        "seed": self._state.seed,
                        "config_hash": self._state.config_hash,
                    },
                    f,
                )

    def _count_total_records(self) -> int:
        """Count total records matching filter."""
        if self.config.filter_expr:
            return self.reader.count_rows(self.config.filter_expr)
        return self.reader.count_rows()

    def _compute_record_hash(self, uuid: str) -> int:
        """Compute deterministic hash for a record."""
        # Combine seed with UUID for deterministic ordering
        key = f"{self.config.seed}:{uuid}"
        return xxhash.xxh64(key.encode()).intdigest()

    def _get_sorted_uuids(self) -> List[str]:
        """Get all UUIDs sorted by their deterministic hash.

        This is cached for efficiency when iterating through batches.
        """
        if self._sorted_uuids is not None:
            return self._sorted_uuids

        # Read all UUIDs
        table = self.reader.read_all(
            columns=["uuid"],
            filter_expr=self.config.filter_expr,
        )
        uuids = table.column("uuid").to_pylist()

        if self.config.shuffle:
            # Sort by deterministic hash
            uuid_with_hash = [(uuid, self._compute_record_hash(uuid)) for uuid in uuids]
            uuid_with_hash.sort(key=lambda x: x[1])
            self._sorted_uuids = [uuid for uuid, _ in uuid_with_hash]
        else:
            self._sorted_uuids = uuids

        return self._sorted_uuids

    def get_batch(
        self,
        batch_number: Optional[int] = None,
        include_batch_info: bool = True,
    ) -> pa.Table:
        """Get a specific batch of records.

        Args:
            batch_number: Batch number (0-indexed). If None, returns next batch.
            include_batch_info: Whether to add batch metadata columns

        Returns:
            Arrow table with batch records

        Example:
            >>> creator.get_batch(0)  # First batch
            >>> creator.get_batch()   # Auto-increment to next batch
        """
        if batch_number is None:
            batch_number = self.state.current_batch
            # Auto-increment for next call
            self._state.current_batch = (batch_number + 1) % max(
                1, self.state.total_batches
            )
            self._save_state()

        # Validate batch number
        if batch_number < 0 or (
            self.state.total_batches > 0 and batch_number >= self.state.total_batches
        ):
            raise ValueError(
                f"Batch number {batch_number} out of range [0, {self.state.total_batches})"
            )

        # Get UUIDs for this batch
        sorted_uuids = self._get_sorted_uuids()
        start_idx = batch_number * self.config.batch_size
        end_idx = min(start_idx + self.config.batch_size, len(sorted_uuids))
        batch_uuids = sorted_uuids[start_idx:end_idx]

        if not batch_uuids:
            return pa.table({})

        # Fetch records by UUID
        filter_expr = ds.field("uuid").isin(batch_uuids)
        if self.config.filter_expr:
            filter_expr = filter_expr & self.config.filter_expr

        table = self.reader.read_all(
            columns=self.config.columns,
            filter_expr=filter_expr,
        )

        # Reorder to match our sorted order
        uuid_to_idx = {uuid: i for i, uuid in enumerate(batch_uuids)}
        records = table.to_pylist()
        records.sort(key=lambda r: uuid_to_idx.get(r.get("uuid"), 0))

        result = pa.Table.from_pylist(records)

        if include_batch_info:
            # Add batch metadata
            n_rows = len(result)
            result = result.append_column(
                "_batch_number", pa.array([batch_number] * n_rows, type=pa.int32())
            )
            result = result.append_column(
                "_batch_seed", pa.array([self.config.seed] * n_rows, type=pa.int64())
            )

        return result

    def get_batch_as_list(
        self,
        batch_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get batch as list of dictionaries.

        Args:
            batch_number: Batch number (0-indexed). If None, returns next batch.

        Returns:
            List of record dictionaries
        """
        table = self.get_batch(batch_number, include_batch_info=False)
        return table.to_pylist()

    def iter_batches(
        self,
        start_batch: int = 0,
        end_batch: Optional[int] = None,
    ) -> Iterator[Tuple[int, pa.Table]]:
        """Iterate through batches.

        Args:
            start_batch: Starting batch number
            end_batch: Ending batch number (exclusive). None means all batches.

        Yields:
            Tuples of (batch_number, table)
        """
        end = end_batch if end_batch is not None else self.state.total_batches

        for batch_num in range(start_batch, end):
            yield batch_num, self.get_batch(batch_num)

    def get_total_batches(self) -> int:
        """Get total number of batches."""
        return self.state.total_batches

    def get_current_batch_number(self) -> int:
        """Get current batch number for auto-increment."""
        return self.state.current_batch

    def reset(self) -> None:
        """Reset batch state to start from beginning."""
        self._state = BatchState(
            current_batch=0,
            total_records=self.state.total_records,
            total_batches=self.state.total_batches,
            seed=self.config.seed,
            config_hash=self._get_config_hash(),
        )
        self._save_state()

    def seek(self, batch_number: int) -> None:
        """Seek to a specific batch number for auto-increment.

        Args:
            batch_number: Batch number to seek to
        """
        if batch_number < 0 or batch_number >= self.state.total_batches:
            raise ValueError(f"Batch number {batch_number} out of range")
        self._state.current_batch = batch_number
        self._save_state()


class StratifiedBatchCreator(BatchCreator):
    """Creates stratified batches based on a column value.

    Ensures each batch has proportional representation of each stratum.
    Useful for balanced training across difficulty bands.
    """

    def __init__(
        self,
        reader: MetadataReader,
        config: BatchConfig,
        state_path: Optional[str | Path] = None,
    ):
        super().__init__(reader, config, state_path)

        if not config.stratify_by:
            raise ValueError("stratify_by must be specified for StratifiedBatchCreator")

        self._stratum_uuids: Optional[Dict[str, List[str]]] = None

    def _get_stratum_uuids(self) -> Dict[str, List[str]]:
        """Get UUIDs grouped and sorted by stratum."""
        if self._stratum_uuids is not None:
            return self._stratum_uuids

        # Read UUIDs with stratification column
        table = self.reader.read_all(
            columns=["uuid", self.config.stratify_by],
            filter_expr=self.config.filter_expr,
        )

        # Group by stratum
        strata: Dict[str, List[Tuple[str, int]]] = {}
        for row in table.to_pylist():
            uuid = row["uuid"]
            stratum = str(row.get(self.config.stratify_by, "unknown"))

            if stratum not in strata:
                strata[stratum] = []

            if self.config.shuffle:
                hash_val = self._compute_record_hash(uuid)
                strata[stratum].append((uuid, hash_val))
            else:
                strata[stratum].append((uuid, 0))

        # Sort within each stratum
        self._stratum_uuids = {}
        for stratum, uuid_hashes in strata.items():
            if self.config.shuffle:
                uuid_hashes.sort(key=lambda x: x[1])
            self._stratum_uuids[stratum] = [uuid for uuid, _ in uuid_hashes]

        return self._stratum_uuids

    def get_batch(
        self,
        batch_number: Optional[int] = None,
        include_batch_info: bool = True,
    ) -> pa.Table:
        """Get a stratified batch of records.

        Each batch contains proportional samples from each stratum.
        """
        if batch_number is None:
            batch_number = self.state.current_batch
            self._state.current_batch = (batch_number + 1) % max(
                1, self.state.total_batches
            )
            self._save_state()

        stratum_uuids = self._get_stratum_uuids()

        # Calculate proportions
        total_records = sum(len(uuids) for uuids in stratum_uuids.values())
        proportions = {
            stratum: len(uuids) / total_records
            for stratum, uuids in stratum_uuids.items()
        }

        # Calculate samples per stratum for this batch
        batch_uuids = []
        for stratum, uuids in stratum_uuids.items():
            n_samples = max(1, int(self.config.batch_size * proportions[stratum]))
            start_idx = batch_number * n_samples
            end_idx = min(start_idx + n_samples, len(uuids))

            # Wrap around if needed
            if start_idx >= len(uuids):
                start_idx = start_idx % len(uuids)
                end_idx = min(start_idx + n_samples, len(uuids))

            batch_uuids.extend(uuids[start_idx:end_idx])

        if not batch_uuids:
            return pa.table({})

        # Fetch records
        filter_expr = ds.field("uuid").isin(batch_uuids)
        table = self.reader.read_all(
            columns=self.config.columns,
            filter_expr=filter_expr,
        )

        if include_batch_info:
            n_rows = len(table)
            table = table.append_column(
                "_batch_number", pa.array([batch_number] * n_rows, type=pa.int32())
            )
            table = table.append_column(
                "_batch_seed", pa.array([self.config.seed] * n_rows, type=pa.int64())
            )

        return table
