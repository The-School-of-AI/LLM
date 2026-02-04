"""
Optimized batch processing utilities for large-scale coreset selection.
Handles 2 trillion+ token datasets with streaming and checkpointing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any
from dataclasses import dataclass, asdict
import pickle

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

    def stream_chunks_from_jsonl(self, filepath: str, max_chunks: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream chunks from JSONL without loading entire file into memory.
        
        Yields:
            (chunk_id, chunk_dict)
        """
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_chunks and count >= max_chunks:
                    break
                try:
                    data = json.loads(line)
                    yield data.get('chunk_id'), data
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipped malformed JSON line {count}: {e}")
                    continue

    def batch_iterator(self, filepath: str, max_chunks: Optional[int] = None) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
        """
        Iterate over chunks in batches from JSONL file.
        
        Yields:
            List of (chunk_id, chunk_dict) tuples
        """
        batch = []
        for chunk_id, data in self.stream_chunks_from_jsonl(filepath, max_chunks):
            batch.append((chunk_id, data))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining
            yield batch

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
