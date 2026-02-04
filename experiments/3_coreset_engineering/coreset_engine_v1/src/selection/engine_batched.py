"""
Enhanced selection engine with batch processing support for 2T+ token scale.
Extends SelectionEngine with streaming deduplication and checkpoint-aware selection.
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Generator
import logging
from collections import defaultdict

from ..core.types import ChunkMetadata
from .engine import SelectionEngine


logger = logging.getLogger(__name__)


class BatchedSelectionEngine(SelectionEngine):
    """
    Selection engine optimized for batch processing of large datasets.
    
    Features:
    - Processes chunks in streaming batches (no full-file load)
    - Applies deduplication at batch boundaries
    - Maintains selection state across batch restarts
    - Checkpoints after each batch for fault tolerance
    """
    
    def select_for_stage_batched(
        self,
        chunk_stream: Generator[Tuple[str, ChunkMetadata], None, None],
        stage_name: str,
        batch_size: int = 10_000,
        protected_slices: Optional[List] = None,
        checkpoint_callback=None,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Select coreset from streaming chunk generator.
        
        Args:
            chunk_stream: Generator yielding (chunk_id, metadata) tuples
            stage_name: Name of training stage
            batch_size: Chunks per batch for processing
            protected_slices: Protected slice rules
            checkpoint_callback: Function(batch_num, selected, stats) for checkpointing
        
        Returns:
            (selected_chunk_ids, selection_stats)
        """
        logger.info(f"Starting batched selection for stage: {stage_name}")
        
        # Validate curriculum
        curriculum_valid, errors = self.curriculum.validate_deterministic_guarantees()
        if not curriculum_valid:
            raise ValueError(f"Curriculum validation failed: {errors}")
        
        selected = set()
        all_chunks = {}
        batch_num = 0
        total_chunks = 0
        total_tokens = 0
        
        try:
            # Process in streaming batches
            batch = []
            for chunk_id, metadata in chunk_stream:
                batch.append((chunk_id, metadata))
                
                if len(batch) >= batch_size:
                    # Process batch
                    selected_in_batch, batch_stats = self._process_batch(
                        batch, stage_name, all_chunks, protected_slices
                    )
                    selected.update(selected_in_batch)
                    
                    total_chunks += len(batch)
                    total_tokens += batch_stats['batch_tokens']
                    
                    logger.info(f"Batch {batch_num}: {len(batch)} chunks, "
                              f"{batch_stats['batch_tokens']:,} tokens, "
                              f"selected so far: {len(selected)}")
                    
                    # Checkpoint
                    if checkpoint_callback:
                        checkpoint_callback(batch_num, selected, {
                            'total_chunks': total_chunks,
                            'total_tokens': total_tokens,
                            'selected_chunks': len(selected),
                        })
                    
                    batch = []
                    batch_num += 1
            
            # Process final batch
            if batch:
                selected_in_batch, batch_stats = self._process_batch(
                    batch, stage_name, all_chunks, protected_slices
                )
                selected.update(selected_in_batch)
                
                total_chunks += len(batch)
                total_tokens += batch_stats['batch_tokens']
                
                logger.info(f"Final batch {batch_num}: {len(batch)} chunks, "
                          f"{batch_stats['batch_tokens']:,} tokens, "
                          f"total selected: {len(selected)}")
                
                if checkpoint_callback:
                    checkpoint_callback(batch_num, selected, {
                        'total_chunks': total_chunks,
                        'total_tokens': total_tokens,
                        'selected_chunks': len(selected),
                    })
        
        except Exception as e:
            logger.error(f"Error during batched selection: {e}", exc_info=True)
            raise
        
        # Compute final stats
        stats = self._compute_selection_stats(selected, all_chunks, stage_name)
        
        logger.info(f"Batched selection complete. {batch_num} batches, "
                   f"{len(selected)} chunks, {stats['selected_tokens']} tokens")
        
        return selected, stats
    
    def _process_batch(
        self,
        batch: List[Tuple[str, ChunkMetadata]],
        stage_name: str,
        all_chunks: Dict[str, ChunkMetadata],
        protected_slices=None,
    ) -> Tuple[Set[str], Dict]:
        """
        Process a single batch of chunks.
        
        Returns:
            (selected_in_batch, batch_stats)
        """
        # Register chunks
        batch_chunks = {}
        batch_tokens = 0
        
        for chunk_id, metadata in batch:
            all_chunks[chunk_id] = metadata
            batch_chunks[chunk_id] = metadata
            batch_tokens += metadata.token_count
        
        # Register for dedup and diversity analysis
        self.register_chunks([(cid, meta, None) for cid, meta in batch_chunks.items()])
        
        # Apply dedup at batch level (not pairwise across all)
        if self.config.dedup.enable_exact_dedup:
            self._apply_batch_deduplication(batch_chunks)
        
        # Create buckets for batch
        self._create_buckets(batch_chunks, stage_name)
        
        # Score chunks in this batch
        for bucket in self.buckets.values():
            self._score_chunks_in_bucket(bucket, batch_chunks)
        
        # Select from batch buckets
        selected_in_batch = set()
        for bucket in self.buckets.values():
            bucket_selection = self._stratified_sample_from_bucket(bucket)
            selected_in_batch.update(bucket_selection)
        
        return selected_in_batch, {
            'batch_tokens': batch_tokens,
            'batch_chunks': len(batch_chunks),
            'batch_selected': len(selected_in_batch),
        }
    
    def _apply_batch_deduplication(self, batch_chunks: Dict[str, ChunkMetadata]) -> None:
        """
        Apply deduplication within a batch only (not globally).
        
        This is more efficient than pairwise comparison across all chunks
        and reduces memory usage for large datasets.
        """
        if not self.config.dedup.enable_exact_dedup:
            return
        
        # Simple hash-based dedup within batch
        hashes_seen = {}
        
        for chunk_id, metadata in batch_chunks.items():
            if chunk_id in self.removed_chunks:
                continue
            
            # Compute hash if text available
            if hasattr(metadata, 'chunk_text'):
                chunk_hash = self.exact_dedup.compute_hash(chunk_id, metadata.chunk_text)
                
                if chunk_hash in hashes_seen:
                    # Keep first, mark second for removal
                    self.removed_chunks.add(chunk_id)
                else:
                    hashes_seen[chunk_hash] = chunk_id
            else:
                # No text, use chunk_id as hash
                if chunk_id in hashes_seen:
                    self.removed_chunks.add(chunk_id)
                else:
                    hashes_seen[chunk_id] = chunk_id
    
    def select_from_checkpoint(
        self,
        chunk_stream: Generator[Tuple[str, ChunkMetadata], None, None],
        stage_name: str,
        last_batch_checkpoint: Optional[int] = None,
        batch_size: int = 10_000,
        protected_slices=None,
        checkpoint_callback=None,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Resume selection from a checkpoint, skipping already-processed batches.
        
        Args:
            chunk_stream: Fresh generator of chunks
            stage_name: Stage name
            last_batch_checkpoint: Number of last successfully processed batch
            batch_size: Chunks per batch
            protected_slices: Protected slice rules
            checkpoint_callback: Checkpoint save function
        
        Returns:
            (selected, stats)
        """
        if last_batch_checkpoint is None:
            # No resumption needed
            return self.select_for_stage_batched(
                chunk_stream, stage_name, batch_size,
                protected_slices, checkpoint_callback
            )
        
        logger.info(f"Resuming selection for {stage_name} from batch {last_batch_checkpoint + 1}")
        
        # Skip to last checkpoint
        batch_num = 0
        for _ in chunk_stream:
            if batch_num * batch_size >= (last_batch_checkpoint + 1) * batch_size:
                break
            batch_num += 1
        
        # Continue from current position
        return self.select_for_stage_batched(
            chunk_stream, stage_name, batch_size,
            protected_slices, checkpoint_callback
        )
