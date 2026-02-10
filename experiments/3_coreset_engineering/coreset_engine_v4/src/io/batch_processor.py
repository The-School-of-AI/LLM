"""
Optimized batch processing utilities for large-scale coreset selection.
Handles 2 trillion+ token datasets with streaming and checkpointing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any, Iterable
from dataclasses import dataclass, asdict
import pickle

import xxhash

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    stage_name: str
    batch_num: int
    chunks_processed: int
    tokens_processed: int
    selected_chunks: int
    timestamp: str
    config_hash: str


class BatchProcessor:
    """
    Process data in batches to avoid memory overload on 2T token datasets.
    Enables streaming, checkpointing, and resumption.
    """

    def __init__(self, batch_size: int = 10_000, checkpoint_dir: Optional[str] = None):
        """
        Args:
            batch_size: Chunks to process per batch (default 10k)
            checkpoint_dir: Dir for checkpoints; if None, no checkpointing
        """
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def stream_chunks_from_jsonl(
        self,
        filepath: str,
        max_chunks: Optional[int] = None,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
        shard_key: str = "chunk_id",
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream chunks from JSONL without loading entire file into memory.
        
        Yields:
            (chunk_id, chunk_dict)
        """
        shard_id = int(shard_id)
        num_shards = int(num_shards)
        count = 0
        emitted = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_chunks and emitted >= max_chunks:
                    break
                try:
                    data = json.loads(line)
                    # Normalize the unique chunk identifier.
                    # Some datasets use uid/guid/id instead of chunk_id.
                    chunk_id = (
                        data.get('chunk_id')
                        or data.get('uid')
                        or data.get('guid')
                        or data.get('id')
                    )

                    # Optional row-level sharding (useful when input is a single huge file).
                    if num_shards > 1:
                        key_val = data.get(shard_key)
                        if key_val is None and shard_key == "chunk_id":
                            key_val = chunk_id
                        if key_val is None:
                            # Fallback: shard by line index so every row deterministically belongs
                            # to exactly one shard even if chunk_id is missing.
                            key_bytes = str(count).encode("utf-8")
                        else:
                            key_bytes = str(key_val).encode("utf-8")
                        h = xxhash.xxh64(key_bytes).intdigest()
                        if int(h % num_shards) != shard_id:
                            count += 1
                            continue

                    yield chunk_id, data
                    count += 1
                    emitted += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipped malformed JSON line {count}: {e}")
                    continue

    def batch_iterator(
        self,
        filepath: str,
        max_chunks: Optional[int] = None,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
        shard_key: str = "chunk_id",
    ) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
        """
        Iterate over chunks in batches from JSONL file.
        
        Yields:
            List of (chunk_id, chunk_dict) tuples
        """
        batch = []
        for chunk_id, data in self.stream_chunks_from_jsonl(
            filepath,
            max_chunks,
            shard_id=shard_id,
            num_shards=num_shards,
            shard_key=shard_key,
        ):
            batch.append((chunk_id, data))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining
            yield batch

    def list_input_files(self, base_path: str, format: str) -> List[Path]:
        """List input files for local filesystem datasets."""
        root = Path(base_path)
        fmt = format.lower()
        if root.is_file():
            return [root]
        if not root.exists():
            return []
        if fmt == "jsonl":
            return sorted(root.glob("**/*.jsonl"))
        if fmt == "parquet":
            return sorted(root.glob("**/*.parquet"))
        return []

    def shard_files(self, files: List[Path], shard_id: int, num_shards: int) -> List[Path]:
        """Deterministically shard files across workers using xxhash of the path."""
        if num_shards <= 1:
            return files
        out: List[Path] = []
        for p in files:
            h = xxhash.xxh64(str(p).encode("utf-8")).intdigest()
            if int(h % num_shards) == int(shard_id):
                out.append(p)
        return out

    def parquet_batch_iterator(
        self,
        path: str,
        batch_size_rows: int = 10_000,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream Parquet rows in batches using pyarrow.dataset.

        Notes:
        - Works for a single parquet file or a directory of parquet files.
        - If path is an s3:// URL and pyarrow S3 support is available, it will attempt to read it.
        """
        try:
            import pyarrow.dataset as ds
            import pyarrow as pa
        except Exception as e:
            raise RuntimeError("pyarrow is required for parquet streaming; install pyarrow") from e

        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=int(batch_size_rows))
        emitted = 0

        for record_batch in scanner.to_batches():
            table = pa.Table.from_batches([record_batch])
            rows = table.to_pylist()
            if not rows:
                continue
            if max_rows is not None:
                remaining = int(max_rows) - emitted
                if remaining <= 0:
                    break
                if len(rows) > remaining:
                    rows = rows[:remaining]

            yield rows
            emitted += len(rows)

    def save_checkpoint(self, stage_name: str, batch_num: int, 
                       state: Dict[str, Any], metadata: CheckpointMetadata) -> Path:
        """Save batch checkpoint for resumption."""
        if not self.checkpoint_dir:
            return None
        
        checkpoint_path = (
            self.checkpoint_dir 
            / f"checkpoint_{stage_name}_batch_{batch_num:06d}.pkl"
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'state': state, 'metadata': asdict(metadata)}, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, stage_name: str, batch_num: int) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Load batch checkpoint if exists."""
        if not self.checkpoint_dir:
            return None
        
        checkpoint_path = (
            self.checkpoint_dir 
            / f"checkpoint_{stage_name}_batch_{batch_num:06d}.pkl"
        )
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return data['state'], data['metadata']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def find_last_checkpoint(self, stage_name: str) -> Optional[int]:
        """Find the last completed batch checkpoint for a stage."""
        if not self.checkpoint_dir:
            return None
        
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"checkpoint_{stage_name}_batch_*.pkl"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        if checkpoints:
            batch_num = int(checkpoints[0].stem.split('_')[-1])
            logger.info(f"Found last checkpoint for {stage_name}: batch {batch_num}")
            return batch_num
        
        return None
