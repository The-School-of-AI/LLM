"""
Sharded Arrow IPC writer.

Writes records into a series of Arrow IPC files, each capped at a target
shard size (in MB). Files are named:
    {prefix}_{split}_{shard_idx:05d}.arrow
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


# Fields that hold plain Python lists of ints (stored as Arrow list<int32>).
_LIST_INT_FIELDS = {"input_ids", "attention_mask", "labels"}
# Fields stored as Arrow int32.
_INT32_FIELDS = {"_seq_len", "_unmasked_tokens"}


class ShardedArrowWriter:
    """
    Context-manager that buffers records and flushes Arrow shards.

    Usage::

        with ShardedArrowWriter(output_dir, prefix="sft_train", split="train",
                                shard_size_mb=512, compression="lz4") as writer:
            for record in records:
                writer.write(record)
        shard_paths = writer.shard_paths
    """

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = "sft",
        split: str = "train",
        shard_size_mb: int = 512,
        compression: str = "lz4",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._split = split
        self._shard_size_bytes = shard_size_mb * 1024 * 1024
        self._compression = compression

        self._buffer: list[dict] = []
        self._buffer_bytes: int = 0
        self._shard_idx: int = 0
        self.shard_paths: list[Path] = []

    def __enter__(self) -> "ShardedArrowWriter":
        return self

    def __exit__(self, *_) -> None:
        self.flush()

    def write(self, record: dict) -> None:
        """Add a single record. Flushes a shard when size limit is reached."""
        self._buffer.append(record)
        # Rough byte estimate: JSON length of record
        self._buffer_bytes += len(json.dumps(record, ensure_ascii=False))
        if self._buffer_bytes >= self._shard_size_bytes:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to a new shard file."""
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.ipc as ipc

        table = _records_to_table(self._buffer)
        path = self._output_dir / (
            f"{self._prefix}_{self._split}_{self._shard_idx:05d}.arrow"
        )
        opts = ipc.IpcWriteOptions(compression=self._compression)
        with ipc.new_file(str(path), table.schema, options=opts) as writer:
            writer.write_table(table)

        self.shard_paths.append(path)
        self._shard_idx += 1
        self._buffer = []
        self._buffer_bytes = 0


def read_arrow(path: str | Path) -> list[dict]:
    """Read an Arrow IPC file back into a list of Python dicts."""
    import pyarrow.ipc as ipc

    with ipc.open_file(str(path)) as reader:
        table = reader.read_all()
    return _table_to_records(table)


def iter_arrow(path: str | Path) -> Iterator[dict]:
    """Iterate over records in an Arrow IPC file without loading all into RAM."""
    import pyarrow.ipc as ipc

    with ipc.open_file(str(path)) as reader:
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            rows = batch.to_pydict()
            keys = list(rows.keys())
            n = len(rows[keys[0]]) if keys else 0
            for j in range(n):
                yield {k: rows[k][j] for k in keys}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _records_to_table(records: list[dict]):
    """Convert a list of dicts to a pyarrow Table with appropriate types."""
    import pyarrow as pa

    if not records:
        return pa.table({})

    # Collect all keys (union across records for safety)
    all_keys: list[str] = list(dict.fromkeys(k for r in records for k in r))
    columns: dict[str, list[Any]] = {k: [] for k in all_keys}
    for rec in records:
        for k in all_keys:
            columns[k].append(rec.get(k))

    arrays: dict[str, pa.Array] = {}
    for k, vals in columns.items():
        if k in _LIST_INT_FIELDS:
            arrays[k] = pa.array(
                [v if v is not None else [] for v in vals],
                type=pa.list_(pa.int32()),
            )
        elif k in _INT32_FIELDS:
            arrays[k] = pa.array(
                [int(v) if v is not None else 0 for v in vals],
                type=pa.int32(),
            )
        else:
            # Serialise non-string complex values to JSON
            str_vals = []
            for v in vals:
                if v is None:
                    str_vals.append(None)
                elif isinstance(v, str):
                    str_vals.append(v)
                else:
                    str_vals.append(json.dumps(v, ensure_ascii=False))
            arrays[k] = pa.array(str_vals, type=pa.string())

    return pa.table(arrays)


def _table_to_records(table) -> list[dict]:
    """Convert a pyarrow Table back to a list of Python dicts."""
    import pyarrow as pa

    raw = table.to_pydict()
    keys = list(raw.keys())
    n = len(raw[keys[0]]) if keys else 0
    records = []
    for i in range(n):
        rec: dict[str, Any] = {}
        for k in keys:
            v = raw[k][i]
            # Try JSON-deserialise string fields that look like JSON
            if isinstance(v, str) and v and v[0] in ("{", "[", '"'):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            rec[k] = v
        records.append(rec)
    return records
