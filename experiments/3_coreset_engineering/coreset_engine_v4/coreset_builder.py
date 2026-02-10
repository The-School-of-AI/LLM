#!/usr/bin/env python3
"""
Coreset Selection Engine - Main Entry Point
==============================================

Production-grade coreset selection pipeline for 70B LLM pre-training.
Compresses 2 trillion tokens to ~400 billion tokens across stages.

Usage:
    python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml

Author: Coreset Selection Team
Version: 1.0.0
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Dict, Optional, Any, Iterator, Tuple, List

from src.core.config import PipelineConfig
from src.core.types import StageName, ProtectedSliceRule, CoresetManifest
from src.curriculum.loader import CurriculumLoader
from src.selection.engine import SelectionEngine
from src.selection.engine_batched import BatchedSelectionEngine
from src.io.loaders import ChunkLoader, CoresetWriter, AblationReporter
from src.io.batch_processor import BatchProcessor, CheckpointMetadata
from src.io.used_chunks_store import UsedChunksStore
from src.error_handling import ErrorRecoveryManager, ErrorSeverity, retry_with_backoff


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coreset_selection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CoresetBuilder:
    """Main orchestrator for coreset selection"""
    
    def __init__(self, config_path: str, curriculum_path: str):
        """Initialize builder with configuration files"""
        self._config_path_str = config_path  # store for merge_shard_reports
        self.config = PipelineConfig.load_from_file(config_path)
        
        self.curriculum = CurriculumLoader(curriculum_path)
        success, errors = self.curriculum.load()
        if not success:
            raise ValueError(f"Failed to load curriculum: {errors}")
        
        # Validate curriculum is frozen
        if not self.curriculum.validate_curriculum_frozen():
            logger.warning("Curriculum is not frozen - reproducibility may be compromised")
        
        # Validate deterministic guarantees
        valid, errors = self.curriculum.validate_deterministic_guarantees()
        if not valid:
            raise ValueError(f"Curriculum doesn't guarantee determinism: {errors}")
        
        self.config_hash = self.config.compute_hash()
        self.curriculum_hash = self.curriculum.config_hash
        
        logger.info(f"Config hash: {self.config_hash[:16]}...")
        logger.info(f"Curriculum hash: {self.curriculum_hash[:16]}...")
        # Track chunk ids already selected in earlier stages to ensure disjoint coresets
        self.used_chunk_ids = set()
    
    def build_coresets(self) -> dict:
        """Build coresets for all configured stages"""
        
        results = {}
        
        for stage_name_str, stage_config in self.config.stages.items():
            #if stage_name_str not in ["1B", "3B", "8B", "70B", "SFT", "ALIGNMENT"]:
            if stage_name_str not in ["1B", "3B", "8B", "70B"]:
                continue
            
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing stage: {stage_name_str}")
                logger.info(f"{'='*60}")
                
                stage_result = self._build_stage_coreset(stage_name_str, stage_config)
                results[stage_name_str] = stage_result
                
            except Exception as e:
                logger.error(f"Failed to build coreset for {stage_name_str}: {e}", exc_info=True)
                raise
        
        return results
    
    def _build_stage_coreset(self, stage_name: str, stage_config) -> dict:
        """Build coreset for a single stage"""
        
        # Load chunks
        logger.info(f"Loading chunks for {stage_name}...")
        chunk_loader = ChunkLoader(
            base_path=self.config.io.input_dataset_path,
            use_object_store=self.config.io.use_object_store,
            object_store_type=self.config.io.object_store_type,
            object_store_bucket=self.config.io.object_store_bucket,
            num_parallel_loaders=self.config.io.num_parallel_loaders,
        )
        
        all_chunks = chunk_loader.load_all_chunks()
        # Remove any chunks already selected by previous stages to ensure disjoint coresets
        if self.used_chunk_ids:
            removed = 0
            for uid in list(self.used_chunk_ids):
                if uid in all_chunks:
                    all_chunks.pop(uid, None)
                    removed += 1
            logger.info(f"Filtered out {removed} previously-selected chunks from input pool")

        logger.info(f"Loaded {len(all_chunks)} total chunks (after filtering)")
        
        if not all_chunks:
            raise ValueError(f"No chunks loaded for stage {stage_name}")
        
        # Initialize selection engine
        engine = SelectionEngine(self.config, self.curriculum)
        
        # Register chunks
        logger.info("Registering chunks...")
        chunks_list = [(cid, meta, None) for cid, meta in all_chunks.items()]
        engine.register_chunks(chunks_list)
        
        # Define protected slices - only for bands/domains allocated in this stage
        # Get stage band_ratios to check allocations
        stage_bands = self.curriculum.get_stage_config(stage_name)
        if not stage_bands:
            protected_slices = []
        else:
            protected_slices = []
            # Only protect bands that have > 0 allocation
            if getattr(stage_bands.band_ratios, 'B4', 0.0) > 0:
                protected_slices.append(ProtectedSliceRule("B4", 0.95, "Graduate-level reasoning critical"))
            if getattr(stage_bands.band_ratios, 'B5', 0.0) > 0:
                protected_slices.append(ProtectedSliceRule("B5", 0.95, "PhD-level content for capability emergence"))
            
            # Protect domains if they appear in curriculum allowed_domains
            # Only protect code if it's in any allowed band for this stage
            has_code = False
            has_agentic = False
            has_indic = False
            for band_enum, band_def in self.curriculum.bands.items():
                band_name = band_enum.value
                if getattr(stage_bands.band_ratios, band_name, 0.0) > 0:
                    if 'code' in band_def.allowed_domains:
                        has_code = True
                    if 'agentic' in band_def.allowed_domains:
                        has_agentic = True
                    if 'indic' in band_def.allowed_domains:
                        has_indic = True
            
            if has_code:
                protected_slices.append(ProtectedSliceRule("code", 0.90, "Code capability foundation"))
            if has_agentic:
                protected_slices.append(ProtectedSliceRule("agentic", 0.90, "Emerging agentic behavior"))
            if has_indic:
                protected_slices.append(ProtectedSliceRule("indic", 0.85, "Multilingual grounding"))
        
        logger.info(f"Protected slices for {stage_name}: {len(protected_slices)} rules")
        
        # Run selection
        logger.info("Running selection algorithm...")
        selected_chunks, stats = engine.select_for_stage(
            all_chunks=all_chunks,
            stage_name=stage_name,
            protected_slices=protected_slices
        )
        
        # Get target tokens from curriculum (not pipeline)
        # First try to get from stage_profiles in growth_schedule
        target_tokens_value = stage_config.target_tokens  # Default to pipeline value
        
        if self.curriculum.growth_schedule and self.curriculum.growth_schedule.stage_profiles:
            # Get profile name for this stage
            curriculum_stage = self.curriculum.get_stage_config(stage_name)
            if curriculum_stage and curriculum_stage.profile:
                profile_name = curriculum_stage.profile
                profile = self.curriculum.growth_schedule.stage_profiles.get(profile_name, {})
                profile_total_tokens = profile.get('total_tokens')
                if profile_total_tokens:
                    target_tokens_value = profile_total_tokens
        
        # Create manifest
        manifest = CoresetManifest(
            stage_name=StageName(stage_name),
            coreset_id=hashlib.sha256(
                f"{stage_name}_{self.config_hash}_{self.curriculum_hash}".encode()
            ).hexdigest(),
            target_tokens=target_tokens_value,
            actual_tokens=stats['selected_tokens'],
            created_at=datetime.now().isoformat(),
            pipeline_version=self.config.pipeline_version,
            curriculum_version=self.curriculum.version,
            seed=self.config.curriculum.deterministic_seed,
            config_hash=self.config_hash,
            selected_chunks_count=stats['selected_chunks'],
            composition=self._build_composition(stats),
            protected_slices_preserved=self._estimate_protected_preservation(),
            deterministic=True,
        )
        
        # Save outputs
        logger.info("Saving outputs...")
        writer = CoresetWriter(self.config.io.output_coreset_path)
        
        # Save index
        metadata_dict = {
            cid: {
                'dataset_id': all_chunks[cid].dataset_id,
                'token_count': all_chunks[cid].token_count,
                'band': all_chunks[cid].band.value,
                'domain': all_chunks[cid].domain,
                'language': all_chunks[cid].language,
            }
            for cid in selected_chunks
        }
        
        index_path = writer.save_selected_indices(
            stage_name,
            selected_chunks,
            metadata_dict,
            format=self.config.io.output_index_format,
        )
        manifest.selected_chunks_file = str(index_path)
        
        # Save manifest
        manifest_path = writer.save_manifest(manifest, stage_name)
        
        logger.info(f"Stage {stage_name} coreset complete")
        logger.info(f"  - Chunks: {stats['selected_chunks']:,}")
        logger.info(f"  - Tokens: {stats['selected_tokens']:,}")
        logger.info(f"  - Compression: {stats['compression_ratio']:.2f}x")
        
        # Mark selected chunks as used to prevent reuse in subsequent stages
        self.used_chunk_ids.update(selected_chunks)

        return stats
    
    def _build_composition(self, stats: dict):
        """Build CoresetComposition from stats"""
        from src.core.types import CoresetComposition, BandDistribution, DomainDistribution, LanguageDistribution
        
        return CoresetComposition(
            band_distribution=stats.get('band_distribution'),
            domain_distribution=stats.get('domain_distribution'),
            language_distribution=stats.get('language_distribution'),
        )
    
    def _estimate_protected_preservation(self):
        """Estimate protected slices preservation"""
        from src.core.types import ProtectedSlicesPreserved
        
        return ProtectedSlicesPreserved(
            B4_preservation_ratio=0.95,
            B5_preservation_ratio=0.95,
            code_preservation_ratio=0.90,
            agentic_preservation_ratio=0.90,
            indic_preservation_ratio=0.85,
        )
    
    def generate_reports(self, results: dict):
        """Generate ablation and diagnostic reports.
        
        In sharded mode (num_shards > 1), each shard saves its per-shard
        results as JSON.  The final unified report is produced by a
        separate merge step (--merge-shard-reports).
        """
        logger.info("\nGenerating reports...")
        
        # Collect global dedup stats if available (set by StreamingCoresetBuilder)
        dedup_stats = None
        if hasattr(self, '_global_dedup_total') and self._global_dedup_total > 0:
            dedup_stats = {
                'total': self._global_dedup_total,
                'dropped': self._global_dedup_dropped,
            }
        
        num_shards = getattr(self, 'num_shards', 1)
        shard_id = getattr(self, 'shard_id', 0)
        
        if num_shards > 1:
            # --- Sharded mode: save per-shard JSON ---
            manifest_dir = Path(self.config.io.output_manifest_path)
            manifest_dir.mkdir(parents=True, exist_ok=True)
            
            # Make results JSON-serializable (convert distribution objects)
            serializable_results = {}
            for stage_name, stage_data in results.items():
                sr = {}
                for k, v in stage_data.items():
                    if k in ('band_distribution', 'domain_distribution', 'language_distribution'):
                        sr[k] = v.to_dict() if hasattr(v, 'to_dict') else (v if isinstance(v, dict) else str(v))
                    elif k == 'timings_s':
                        sr[k] = {tk: float(tv) for tk, tv in v.items()} if v else {}
                    else:
                        try:
                            json.dumps(v)  # test serializable
                            sr[k] = v
                        except (TypeError, ValueError):
                            sr[k] = str(v)
                serializable_results[stage_name] = sr
            
            shard_json = manifest_dir / f"shard_results_{shard_id:03d}.json"
            payload = {
                'shard_id': shard_id,
                'num_shards': num_shards,
                'stages_results': serializable_results,
                'dedup_stats': dedup_stats,
            }
            with open(shard_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Saved shard results to {shard_json}")
            
            # Auto-merge: check if all N shard JSONs are now present.
            # The last shard to finish will find all files and produce the
            # unified report automatically — no external merge step needed.
            existing = sorted(manifest_dir.glob("shard_results_*.json"))
            if len(existing) >= num_shards:
                logger.info(f"All {num_shards} shard results present — merging into unified report...")
                
                # Build a minimal args-like namespace for merge_shard_reports
                class _MergeArgs:
                    config = self.config_path if hasattr(self, 'config_path') else 'config/pipeline.yaml'
                
                # Use the config path stored during __init__
                _MergeArgs.config = self._config_path_str
                merge_shard_reports(_MergeArgs())
            else:
                logger.info(
                    f"Shard {shard_id} done ({len(existing)}/{num_shards} shards finished). "
                    f"Unified report will be generated when the last shard completes."
                )
        else:
            # --- Single-process mode: generate report directly ---
            report_path = AblationReporter.generate_report(
                results,
                self.config.io.output_manifest_path,
                dedup_stats=dedup_stats,
            )
            logger.info(f"Report saved to: {report_path}")


class StreamingCoresetBuilder(CoresetBuilder):
    """Streaming + fault-tolerant builder suitable for 2T-scale datasets."""

    def __init__(
        self,
        config_path: str,
        curriculum_path: str,
        *,
        input_path: str,
        input_format: str,
        batch_size: int = 10_000,
        checkpoint_dir: Optional[str] = None,
        total_input_tokens_estimate: Optional[int] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        max_rows: Optional[int] = None,
        stages: Optional[List[str]] = None,
        stage_target_scale: float = 1.0,
    ):
        super().__init__(config_path, curriculum_path)
        self.input_path = input_path
        self.input_format = input_format.lower()
        self.batch_size = int(batch_size)
        self.total_input_tokens_estimate = int(total_input_tokens_estimate) if total_input_tokens_estimate else None
        self.shard_id = int(shard_id)
        self.num_shards = int(num_shards)
        self.max_rows = int(max_rows) if max_rows else None
        self.stages = stages or ["1B", "3B", "8B", "70B"]
        self.stage_target_scale = float(stage_target_scale)

        self.batch_processor = BatchProcessor(batch_size=self.batch_size, checkpoint_dir=checkpoint_dir)
        self.error_recovery = ErrorRecoveryManager()

        # Global hash-based dedup state (same logic as notebook cell #6).
        # Persists across all batches and stages so duplicates across
        # the entire dataset are caught, not just within a single batch.
        self._global_seen_hashes: set = set()
        self._global_dedup_total: int = 0
        self._global_dedup_dropped: int = 0

        # Enforce cross-stage non-overlap for streaming runs via disk-backed membership.
        used_dir = Path(self.config.io.output_coreset_path) / ".used_chunks"
        used_db = used_dir / f"used_chunks_shard{self.shard_id:03d}.sqlite"
        self.used_store = UsedChunksStore(used_db)

    def build_coresets(self) -> dict:
        results = {}

        for stage_name in self.stages:
            if stage_name not in self.config.stages:
                continue
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Streaming stage: {stage_name} (shard {self.shard_id}/{self.num_shards})")
                logger.info(f"{'='*60}")
                results[stage_name] = self._build_stage_coreset(stage_name, self.config.stages[stage_name])
            except Exception as e:
                logger.error(f"Failed stage {stage_name}: {e}", exc_info=True)
                raise

        return results

    def _iter_batches(self) -> Iterator[Tuple[int, List[Tuple[str, Dict[str, Any]]]]]:
        """Yield (batch_idx, batch_rows) where batch_rows is [(chunk_id, row_dict), ...]."""

        # Reset per-stage dedup state.  Each stage re-scans the full dataset,
        # so we only want to remove duplicates *within* a single pass — not
        # across stages (that would cause stages 3B/8B/70B to see 0 rows).
        self._global_seen_hashes.clear()

        if self.input_format == "jsonl":
            files = self.batch_processor.list_input_files(self.input_path, "jsonl")
            if not files:
                raise ValueError(f"No JSONL files found under {self.input_path}")

            # File-level sharding works well when there are many files. If there's only one file
            # total (either input_path is a file or the directory contains a single file), then
            # file sharding would assign that file to exactly one shard. In that case we switch
            # to row-level sharding by chunk_id so all shards can work.
            row_level_shard = (self.num_shards > 1 and len(files) == 1)
            if not row_level_shard:
                files = self.batch_processor.shard_files(files, self.shard_id, self.num_shards)

            emitted = 0
            batch_idx = 0
            for f in files:
                for batch in self.batch_processor.batch_iterator(
                    str(f),
                    max_chunks=self.max_rows,
                    shard_id=(self.shard_id if row_level_shard else 0),
                    num_shards=(self.num_shards if row_level_shard else 1),
                    shard_key="chunk_id",
                ):
                    if self.max_rows is not None:
                        remaining = self.max_rows - emitted
                        if remaining <= 0:
                            return
                        if len(batch) > remaining:
                            batch = batch[:remaining]
                    # ---- GLOBAL HASH-BASED DEDUP for JSONL rows ----
                    deduped_batch = []
                    for cid, row in batch:
                        row_hash = row.get("hash")
                        if row_hash is not None:
                            self._global_dedup_total += 1
                            if row_hash in self._global_seen_hashes:
                                self._global_dedup_dropped += 1
                                continue
                            self._global_seen_hashes.add(row_hash)
                        deduped_batch.append((cid, row))

                    emitted += len(deduped_batch)
                    if deduped_batch:
                        yield batch_idx, deduped_batch
                        batch_idx += 1
            return

        if self.input_format == "parquet":
            files = self.batch_processor.list_input_files(self.input_path, "parquet")
            if files:
                files = self.batch_processor.shard_files(files, self.shard_id, self.num_shards)
                paths = [str(p) for p in files]
            else:
                paths = [self.input_path]

            # OLD columns (original schema - did not match NCERT parquet):
            # columns = [
            #     "chunk_id",
            #     "dataset_id",
            #     "token_count",
            #     "byte_length",
            #     "domain",
            #     "language",
            #     "band",
            #     "source_doc_id",
            #     "source_url",
            #     "token_ids",
            # ]
            # NEW columns (matching actual NCERT parquet schema):
            # OLD: columns without hash
            # columns = [
            #     "uuid", "source", "token_count_estimate",
            #     "byte_length", "domain", "language", "assigned_band",
            # ]
            # NEW: added "hash" for global dedup (same as notebook cell #6)
            columns = [
                "uuid",              # was: chunk_id
                "source",            # was: dataset_id
                "token_count_estimate",  # was: token_count
                "byte_length",
                "domain",
                "language",
                "assigned_band",     # was: band
                "hash",              # for global dedup
            ]

            batch_idx = 0
            emitted = 0
            for p in paths:
                for rows in self.batch_processor.parquet_batch_iterator(
                    p,
                    batch_size_rows=self.batch_size,
                    columns=columns,
                    max_rows=(None if self.max_rows is None else self.max_rows - emitted),
                ):
                    out: List[Tuple[str, Dict[str, Any]]] = []
                    for r in rows:
                        # OLD: cid = r.get("chunk_id")
                        # NEW: NCERT parquet uses "uuid" as the chunk identifier
                        cid = r.get("uuid") or r.get("chunk_id")
                        if cid is None:
                            continue

                        # ---- GLOBAL HASH-BASED DEDUP (notebook cell #6 logic) ----
                        # If the row has a "hash" column, use it for global dedup.
                        # Identical hashes across ANY batch/stage are dropped.
                        row_hash = r.get("hash")
                        if row_hash is not None:
                            self._global_dedup_total += 1
                            if row_hash in self._global_seen_hashes:
                                self._global_dedup_dropped += 1
                                continue
                            self._global_seen_hashes.add(row_hash)

                        out.append((str(cid), r))
                    emitted += len(out)
                    if out:
                        yield batch_idx, out
                        batch_idx += 1
                    if self.max_rows is not None and emitted >= self.max_rows:
                        return
            return

        raise ValueError(f"Unsupported input_format: {self.input_format}")

    @retry_with_backoff(max_retries=3)
    def _write_checkpoint(self, stage_name: str, batch_idx: int, state: Dict[str, Any]) -> None:
        metadata = CheckpointMetadata(
            stage_name=stage_name,
            batch_num=batch_idx,
            chunks_processed=int(state.get("total_chunks_seen", 0)),
            tokens_processed=int(state.get("total_tokens_seen", 0)),
            selected_chunks=int(state.get("selected_chunks", 0)),
            timestamp=datetime.now().isoformat(),
            config_hash=self.config_hash[:16],
        )
        self.batch_processor.save_checkpoint(stage_name, batch_idx, state, metadata)

    def _build_stage_coreset(self, stage_name: str, stage_config) -> dict:
        # Resolve stage target tokens from curriculum profile if present
        target_tokens_value = int(stage_config.target_tokens)
        if self.curriculum.growth_schedule and self.curriculum.growth_schedule.stage_profiles:
            curriculum_stage = self.curriculum.get_stage_config(stage_name)
            if curriculum_stage and curriculum_stage.profile:
                profile = self.curriculum.growth_schedule.stage_profiles.get(curriculum_stage.profile, {})
                profile_total_tokens = profile.get("total_tokens")
                if profile_total_tokens:
                    target_tokens_value = int(profile_total_tokens)

        # Shard scaling: each worker targets 1/num_shards of stage target
        stage_target_tokens = int(target_tokens_value)
        # Test scaling: allow running end-to-end on small datasets while exercising real selection
        if self.stage_target_scale and self.stage_target_scale != 1.0:
            stage_target_tokens = max(0, int(stage_target_tokens * self.stage_target_scale))
        if self.num_shards > 1:
            stage_target_tokens = int(stage_target_tokens / self.num_shards)

        shard_total_tokens_est = None
        if self.total_input_tokens_estimate is not None:
            shard_total_tokens_est = int(self.total_input_tokens_estimate / max(1, self.num_shards))

        # Resume from checkpoint
        last_batch = self.batch_processor.find_last_checkpoint(stage_name)
        start_batch = (last_batch + 1) if last_batch is not None else 0
        if start_batch > 0:
            logger.info(f"Resuming {stage_name} from batch {start_batch}")

        engine = BatchedSelectionEngine(self.config, self.curriculum)

        # Protected slices (lightweight; doesn't require scanning all chunks)
        protected_slices: List[ProtectedSliceRule] = []
        stage_bands = self.curriculum.get_stage_config(stage_name)
        if stage_bands:
            if getattr(stage_bands.band_ratios, 'B4', 0.0) > 0:
                protected_slices.append(ProtectedSliceRule("B4", 0.95, "Graduate-level reasoning critical"))
            if getattr(stage_bands.band_ratios, 'B5', 0.0) > 0:
                protected_slices.append(ProtectedSliceRule("B5", 0.95, "PhD-level content for capability emergence"))

            # Protect domains if they appear in curriculum allowed_domains for bands allocated in this stage.
            has_code = False
            has_agentic = False
            has_indic = False
            for band_enum, band_def in self.curriculum.bands.items():
                band_name = band_enum.value
                if getattr(stage_bands.band_ratios, band_name, 0.0) > 0:
                    if 'code' in (band_def.allowed_domains or []):
                        has_code = True
                    if 'agentic' in (band_def.allowed_domains or []):
                        has_agentic = True
                    if 'indic' in (band_def.allowed_domains or []):
                        has_indic = True

            if has_code:
                protected_slices.append(ProtectedSliceRule("code", 0.90, "Code capability foundation"))
            if has_agentic:
                protected_slices.append(ProtectedSliceRule("agentic", 0.90, "Emerging agentic behavior"))
            if has_indic:
                protected_slices.append(ProtectedSliceRule("indic", 0.85, "Multilingual grounding"))

        writer = CoresetWriter(self.config.io.output_coreset_path)
        stage_dir = Path(self.config.io.output_coreset_path) / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        total_chunks_seen = 0
        total_tokens_seen = 0
        selected_chunks = 0
        selected_tokens = 0
        parts_written = 0

        timing_totals: Dict[str, float] = {}

        from collections import Counter
        band_tokens: Counter[str] = Counter()
        domain_tokens: Counter[str] = Counter()
        language_tokens: Counter[str] = Counter()

        # Availability stats: remaining eligible pool after non-overlap filtering and
        # stage gating (language + allowed_domains + bands present in stage config).
        eligible_unused_tokens_total = 0
        eligible_unused_chunks_total = 0
        eligible_unused_tokens_by_band: Counter[str] = Counter()
        eligible_unused_chunks_by_band: Counter[str] = Counter()

        from src.core.types import ChunkMetadata, DifficultyBand
        import pandas as pd

        # Pre-compute allowed languages for this stage (match BatchedSelectionEngine early filtering).
        explicitly_excluded_langs = set()
        allowed_languages_for_stage = None
        if self.curriculum.language_policy:
            explicitly_excluded_langs = set(self.curriculum.language_policy.explicitly_excluded or set())
            allowed_languages_for_stage = set(self.curriculum.language_policy.primary_languages.keys())

            stage_order = ['1B', '3B', '8B', '70B', 'SFT', 'ALIGNMENT']
            current_stage_idx = stage_order.index(stage_name) if stage_name in stage_order else len(stage_order)
            raw = getattr(self.curriculum, 'raw_curriculum', {}) or {}
            secondary_specs = (
                raw.get('language_and_context', {})
                   .get('language_policy', {})
                   .get('secondary_languages', [])
            )
            if isinstance(secondary_specs, list) and secondary_specs:
                for spec in secondary_specs:
                    # OLD: lang was assumed to be a string
                    # lang = spec.get('lang')
                    # earliest = spec.get('earliest_stage')
                    # if not lang:
                    #     continue
                    # if not earliest:
                    #     allowed_languages_for_stage.add(lang)
                    #     continue
                    # try:
                    #     if stage_order.index(str(earliest)) <= current_stage_idx:
                    #         allowed_languages_for_stage.add(lang)
                    # except ValueError:
                    #     allowed_languages_for_stage.add(lang)

                    # NEW: Handle lang as either a string ("hi") or list (["as","bn",...])
                    lang_val = spec.get('lang')
                    earliest = spec.get('earliest_stage')
                    if not lang_val:
                        continue
                    # Normalize to a list of language codes
                    langs = lang_val if isinstance(lang_val, list) else [lang_val]
                    for lang in langs:
                        if not earliest:
                            allowed_languages_for_stage.add(lang)
                            continue
                        try:
                            if stage_order.index(str(earliest)) <= current_stage_idx:
                                allowed_languages_for_stage.add(lang)
                        except ValueError:
                            allowed_languages_for_stage.add(lang)
            else:
                allowed_languages_for_stage.update(self.curriculum.language_policy.secondary_languages.keys())

        if last_batch is not None and last_batch >= 0:
            loaded = self.batch_processor.load_checkpoint(stage_name, last_batch)
            if loaded is not None:
                state, _metadata = loaded
                total_chunks_seen = int(state.get("total_chunks_seen", 0))
                total_tokens_seen = int(state.get("total_tokens_seen", 0))
                selected_chunks = int(state.get("selected_chunks", 0))
                selected_tokens = int(state.get("selected_tokens", 0))
                parts_written = int(state.get("parts_written", 0))
                band_tokens.update(state.get("band_tokens", {}) or {})
                domain_tokens.update(state.get("domain_tokens", {}) or {})
                language_tokens.update(state.get("language_tokens", {}) or {})
                eligible_unused_tokens_total = int(state.get("eligible_unused_tokens_total", 0) or 0)
                eligible_unused_chunks_total = int(state.get("eligible_unused_chunks_total", 0) or 0)
                eligible_unused_tokens_by_band.update(state.get("eligible_unused_tokens_by_band", {}) or {})
                eligible_unused_chunks_by_band.update(state.get("eligible_unused_chunks_by_band", {}) or {})

        for batch_idx, batch in self._iter_batches():
            if batch_idx < start_batch:
                continue

            try:
                # Parse batch into ChunkMetadata (after non-overlap filtering)
                batch_ids = [str(chunk_id) for chunk_id, _row in batch]
                allowed_ids = self.used_store.filter_unused(batch_ids)

                stream: List[Tuple[str, ChunkMetadata]] = []
                batch_tokens = 0
                for chunk_id, row in batch:
                    if str(chunk_id) not in allowed_ids:
                        continue
                    try:
                        # OLD row parsing (original schema):
                        # token_count = int(row.get("token_count", 0) or 0)
                        # meta = ChunkMetadata(
                        #     chunk_id=str(chunk_id),
                        #     dataset_id=row.get("dataset_id", "ds"),
                        #     token_count=token_count,
                        #     byte_length=int(row.get("byte_length", 0) or 0),
                        #     domain=row.get("domain", "clean_web"),
                        #     language=row.get("language", "en"),
                        #     band=DifficultyBand(row.get("band", "B0")),
                        #     source_doc_id=row.get("source_doc_id", ""),
                        #     source_url=row.get("source_url", None),
                        # )
                        # token_ids = row.get("token_ids")
                        # if token_ids is not None:
                        #     try:
                        #         setattr(meta, "token_ids", list(token_ids))
                        #     except Exception:
                        #         pass

                        # NEW row parsing (NCERT parquet schema):
                        token_count = int(row.get("token_count_estimate", 0) or row.get("token_count", 0) or 0)
                        batch_tokens += token_count
                        meta = ChunkMetadata(
                            chunk_id=str(chunk_id),
                            dataset_id=row.get("source", "") or row.get("dataset_id", "ds"),
                            token_count=token_count,
                            byte_length=int(row.get("byte_length", 0) or 0),
                            domain=row.get("domain", "clean_web"),
                            language=row.get("language", "en"),
                            band=DifficultyBand(row.get("assigned_band", "B0") or row.get("band", "B0")),
                            source_doc_id=row.get("source_doc_id", ""),
                            source_url=row.get("source_url", None),
                        )

                        # Availability accounting: only count chunks that are eligible for selection
                        # given stage band ratios, allowed_domains, and language gating.
                        band_name = meta.band.value
                        band_ratio = getattr(stage_bands.band_ratios, band_name, 0.0) if stage_bands else 0.0
                        band_in_stage = (band_ratio > 0.0) if stage_bands else True
                        allowed_domains = self.curriculum.get_allowed_domains_for_band(meta.band)
                        domain_allowed = (meta.domain in allowed_domains) if allowed_domains else True
                        language_allowed = True
                        if allowed_languages_for_stage is not None:
                            language_allowed = meta.language in allowed_languages_for_stage
                        if meta.language in explicitly_excluded_langs:
                            language_allowed = False

                        if band_in_stage and domain_allowed and language_allowed:
                            eligible_unused_chunks_total += 1
                            eligible_unused_tokens_total += int(meta.token_count)
                            eligible_unused_chunks_by_band[band_name] += 1
                            eligible_unused_tokens_by_band[band_name] += int(meta.token_count)

                        stream.append((str(chunk_id), meta))
                    except Exception as e:
                        self.error_recovery.handle_error(e, "RowParseError", stage_name=stage_name, batch_num=batch_idx)
                        continue

                if not stream:
                    continue

                selected_ids, batch_stats = engine._process_batch(
                    stream,
                    stage_name,
                    protected_slices,
                    total_input_tokens_estimate=shard_total_tokens_est,
                    stage_target_tokens=stage_target_tokens,
                )

                for k, v in (batch_stats.get("timings_s") or {}).items():
                    timing_totals[k] = float(timing_totals.get(k, 0.0)) + float(v)

                # Write selected indices for this batch as a parquet part file
                if selected_ids:
                    meta_by_id = {cid: meta for cid, meta in stream}
                    rows = []
                    for cid in selected_ids:
                        meta = meta_by_id.get(cid)
                        if not meta:
                            continue
                        tc = int(meta.token_count)
                        band_tokens[meta.band.value] += tc
                        dom = meta.domain if meta.domain in {"code", "math", "reasoning", "agentic", "indic", "clean_web"} else "clean_web"
                        domain_tokens[dom] += tc
                        language_tokens[str(meta.language)] += tc
                        rows.append({
                            "chunk_id": meta.chunk_id,
                            "dataset_id": meta.dataset_id,
                            "token_count": tc,
                            "band": meta.band.value,
                            "domain": meta.domain,
                            "language": meta.language,
                        })
                    if rows:
                        part_path = stage_dir / f"selected_indices_part_shard{self.shard_id:03d}_batch{batch_idx:06d}.parquet"
                        pd.DataFrame(rows).to_parquet(part_path, index=False)
                        parts_written += 1

                    # Update used-chunk membership immediately so later stages cannot re-select.
                    self.used_store.add_many(selected_ids)

                total_chunks_seen += len(stream)
                total_tokens_seen += batch_tokens
                selected_chunks += int(batch_stats.get("batch_selected", 0))
                selected_tokens += int(batch_stats.get("batch_selected_tokens", 0))

                # Checkpoint after successful batch
                state = {
                    "shard_id": self.shard_id,
                    "num_shards": self.num_shards,
                    "stage_target_tokens": stage_target_tokens,
                    "total_chunks_seen": total_chunks_seen,
                    "total_tokens_seen": total_tokens_seen,
                    "selected_chunks": selected_chunks,
                    "selected_tokens": selected_tokens,
                    "parts_written": parts_written,
                    "band_tokens": dict(band_tokens),
                    "domain_tokens": dict(domain_tokens),
                    "language_tokens": dict(language_tokens),
                    "eligible_unused_tokens_total": int(eligible_unused_tokens_total),
                    "eligible_unused_chunks_total": int(eligible_unused_chunks_total),
                    "eligible_unused_tokens_by_band": dict(eligible_unused_tokens_by_band),
                    "eligible_unused_chunks_by_band": dict(eligible_unused_chunks_by_band),
                }
                self._write_checkpoint(stage_name, batch_idx, state)

                logger.info(
                    f"{stage_name} batch {batch_idx}: seen_tokens={total_tokens_seen:,} "
                    f"selected_tokens={selected_tokens:,} batch_target={batch_stats.get('batch_target_tokens', 0):,}"
                )

            except Exception as e:
                ctx = self.error_recovery.handle_error(e, "BatchProcessingError", stage_name=stage_name, batch_num=batch_idx)
                logger.warning(f"Recovery suggestion: {self.error_recovery.get_recovery_action(ctx)}")
                if ctx.severity == ErrorSeverity.FATAL:
                    raise
                continue

        # Save minimal manifest for this shard
        from src.core.types import CoresetComposition, BandDistribution, DomainDistribution, LanguageDistribution

        if selected_tokens > 0:
            band_dist = BandDistribution(
                B0=float(band_tokens.get("B0", 0)) / float(selected_tokens),
                B1=float(band_tokens.get("B1", 0)) / float(selected_tokens),
                B2=float(band_tokens.get("B2", 0)) / float(selected_tokens),
                B3=float(band_tokens.get("B3", 0)) / float(selected_tokens),
                B4=float(band_tokens.get("B4", 0)) / float(selected_tokens),
                B5=float(band_tokens.get("B5", 0)) / float(selected_tokens),
            )
            domain_dist = DomainDistribution(
                code=float(domain_tokens.get("code", 0)) / float(selected_tokens),
                math=float(domain_tokens.get("math", 0)) / float(selected_tokens),
                reasoning=float(domain_tokens.get("reasoning", 0)) / float(selected_tokens),
                agentic=float(domain_tokens.get("agentic", 0)) / float(selected_tokens),
                indic=float(domain_tokens.get("indic", 0)) / float(selected_tokens),
                clean_web=float(domain_tokens.get("clean_web", 0)) / float(selected_tokens),
            )
            language_dist = LanguageDistribution(
                languages={k: float(v) / float(selected_tokens) for k, v in language_tokens.items()}
            )
        else:
            band_dist = BandDistribution()
            domain_dist = DomainDistribution()
            language_dist = LanguageDistribution(languages={})

        composition = CoresetComposition(
            band_distribution=band_dist,
            domain_distribution=domain_dist,
            language_distribution=language_dist,
        )

        manifest = CoresetManifest(
            stage_name=StageName(stage_name),
            coreset_id=hashlib.sha256(
                f"{stage_name}_{self.config_hash}_{self.curriculum_hash}_shard{self.shard_id}_{self.num_shards}".encode()
            ).hexdigest(),
            target_tokens=stage_target_tokens,
            actual_tokens=selected_tokens,
            created_at=datetime.now().isoformat(),
            pipeline_version=self.config.pipeline_version,
            curriculum_version=self.curriculum.version,
            seed=self.config.curriculum.deterministic_seed,
            config_hash=self.config_hash,
            selected_chunks_count=selected_chunks,
            composition=composition,
            protected_slices_preserved=self._estimate_protected_preservation(),
            rolling_window_stats=engine.get_rolling_window_stats(),
            availability_stats={
                "eligible_unused_tokens_total": int(eligible_unused_tokens_total),
                "eligible_unused_chunks_total": int(eligible_unused_chunks_total),
                "eligible_unused_tokens_by_band": {k: int(v) for k, v in dict(eligible_unused_tokens_by_band).items()},
                "eligible_unused_chunks_by_band": {k: int(v) for k, v in dict(eligible_unused_chunks_by_band).items()},
                "definition": (
                    "Counts chunks/tokens that were unused (non-overlap filtered) and eligible for this stage "
                    "by band/domain/language policy before selection."
                ),
            },
            deterministic=True,
        )
        manifest.selected_chunks_file = str(stage_dir)
        manifest_path = stage_dir / f"manifest_shard{self.shard_id:03d}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))
        logger.info(f"Saved manifest to {manifest_path}")

        # Backward-compatible manifest filename for single-shard runs
        if self.num_shards == 1:
            legacy_manifest_path = stage_dir / "manifest.json"
            with open(legacy_manifest_path, "w", encoding="utf-8") as f:
                f.write(manifest.to_json(indent=2))
            logger.info(f"Saved manifest to {legacy_manifest_path}")

        if timing_totals:
            timing_str = " | ".join(
                f"{k}={timing_totals[k]:.3f}s" for k in sorted(timing_totals.keys())
            )
            logger.info(f"{stage_name} timing totals: {timing_str}")

        return {
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "total_chunks_seen": total_chunks_seen,
            "total_tokens_seen": total_tokens_seen,
            "selected_chunks": selected_chunks,
            "selected_tokens": selected_tokens,
            "parts_written": parts_written,
            "timings_s": timing_totals,
            # Fields expected by AblationReporter
            "total_input_chunks": total_chunks_seen,
            "total_input_tokens": total_tokens_seen,
            "band_distribution": band_dist,
            "domain_distribution": domain_dist,
            "language_distribution": language_dist,
        }


def merge_shard_reports(args) -> int:
    """Merge per-shard result JSONs into a single unified ablation report.
    
    Each shard writes a ``shard_results_NNN.json`` file.  This function
    reads them all, aggregates stage metrics (sums tokens/chunks across
    shards, merges distributions), combines dedup stats, and generates
    one unified ablation report via AblationReporter.
    """
    from src.core.config import PipelineConfig
    
    config = PipelineConfig.load_from_file(args.config)
    manifest_dir = Path(config.io.output_manifest_path)
    
    shard_files = sorted(manifest_dir.glob("shard_results_*.json"))
    if not shard_files:
        logger.error(f"No shard_results_*.json files found in {manifest_dir}")
        return 1
    
    logger.info(f"Found {len(shard_files)} shard result files in {manifest_dir}")
    
    # Load all shard payloads
    shards = []
    for sf in shard_files:
        with open(sf, 'r', encoding='utf-8') as f:
            shards.append(json.load(f))
    
    # Aggregate: sum per-stage metrics across shards
    all_stage_names = set()
    for s in shards:
        all_stage_names.update(s['stages_results'].keys())
    
    merged_results = {}
    for stage_name in sorted(all_stage_names):
        merged = {
            'total_input_tokens': 0,
            'total_input_chunks': 0,
            'selected_tokens': 0,
            'selected_chunks': 0,
        }
        # Collect band/domain/language dicts from each shard and weighted-merge
        band_weighted = {}   # band_name -> total_tokens
        domain_weighted = {} # domain -> total_tokens
        lang_weighted = {}   # lang -> total_tokens
        
        for s in shards:
            sr = s['stages_results'].get(stage_name)
            if not sr:
                continue
            merged['total_input_tokens'] += sr.get('total_input_tokens', 0)
            merged['total_input_chunks'] += sr.get('total_input_chunks', 0)
            merged['selected_tokens'] += sr.get('selected_tokens', 0)
            merged['selected_chunks'] += sr.get('selected_chunks', 0)
            
            shard_selected = sr.get('selected_tokens', 0)
            
            # Aggregate band distribution (weighted by selected tokens)
            bd = sr.get('band_distribution', {})
            if isinstance(bd, dict):
                for band, ratio in bd.items():
                    band_weighted[band] = band_weighted.get(band, 0) + ratio * shard_selected
            
            # Aggregate domain distribution
            dd = sr.get('domain_distribution', {})
            if isinstance(dd, dict):
                for domain, ratio in dd.items():
                    domain_weighted[domain] = domain_weighted.get(domain, 0) + ratio * shard_selected
            
            # Aggregate language distribution
            ld = sr.get('language_distribution', {})
            if isinstance(ld, dict):
                for lang, ratio in ld.items():
                    lang_weighted[lang] = lang_weighted.get(lang, 0) + ratio * shard_selected
        
        # Convert weighted sums back to ratios
        total_selected = merged['selected_tokens'] or 1
        if band_weighted:
            merged['band_distribution'] = {b: t / total_selected for b, t in band_weighted.items()}
        if domain_weighted:
            merged['domain_distribution'] = {d: t / total_selected for d, t in domain_weighted.items()}
        if lang_weighted:
            merged['language_distribution'] = {l: t / total_selected for l, t in lang_weighted.items()}
        
        merged_results[stage_name] = merged
    
    # Aggregate dedup stats across shards
    merged_dedup = {'total': 0, 'dropped': 0}
    for s in shards:
        ds = s.get('dedup_stats')
        if ds:
            merged_dedup['total'] += ds.get('total', 0)
            merged_dedup['dropped'] += ds.get('dropped', 0)
    
    if merged_dedup['total'] == 0:
        merged_dedup = None
    
    # ---- Cross-shard dedup on selected indices ----
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    coreset_base = Path(config.io.output_manifest_path).parent / 'coresets'
    cross_shard_dedup_stats = {'total_chunks': 0, 'duplicates_removed': 0, 'tokens_removed': 0}
    
    for stage_name in sorted(all_stage_names):
        stage_dir = coreset_base / stage_name
        if not stage_dir.exists():
            continue
        
        idx_files = sorted(stage_dir.glob('selected_indices_*.parquet'))
        if not idx_files:
            continue
        
        seen_chunk_ids: set = set()
        stage_total = 0
        stage_dupes = 0
        stage_tokens_removed = 0
        
        for idx_file in idx_files:
            table = pq.read_table(idx_file)
            n_before = len(table)
            stage_total += n_before
            
            if 'chunk_id' not in table.column_names:
                continue
            
            # Build keep mask
            chunk_ids = table.column('chunk_id').to_pylist()
            token_counts = table.column('token_count').to_pylist() if 'token_count' in table.column_names else [0] * len(chunk_ids)
            mask = []
            for cid, tc in zip(chunk_ids, token_counts):
                if cid in seen_chunk_ids:
                    mask.append(False)
                    stage_dupes += 1
                    stage_tokens_removed += tc
                else:
                    seen_chunk_ids.add(cid)
                    mask.append(True)
            
            # Rewrite file if any duplicates were found
            if not all(mask):
                filtered = table.filter(pa.array(mask, type=pa.bool_()))
                pq.write_table(filtered, idx_file)
        
        cross_shard_dedup_stats['total_chunks'] += stage_total
        cross_shard_dedup_stats['duplicates_removed'] += stage_dupes
        cross_shard_dedup_stats['tokens_removed'] += stage_tokens_removed
        
        # Adjust merged_results for this stage
        if stage_dupes > 0 and stage_name in merged_results:
            merged_results[stage_name]['selected_chunks'] -= stage_dupes
            merged_results[stage_name]['selected_tokens'] -= stage_tokens_removed
            logger.info(
                f"  Cross-shard dedup ({stage_name}): removed {stage_dupes} duplicate chunks "
                f"({stage_tokens_removed:,} tokens)"
            )
    
    if cross_shard_dedup_stats['duplicates_removed'] == 0:
        logger.info("  Cross-shard dedup: no duplicate chunks found across shards")
    
    # Log aggregated summary
    logger.info("\n--- Merged Shard Summary ---")
    for stage_name, m in sorted(merged_results.items()):
        logger.info(
            f"  {stage_name}: input_tokens={m['total_input_tokens']:,}  "
            f"selected_tokens={m['selected_tokens']:,}  "
            f"selected_chunks={m['selected_chunks']:,}"
        )
    if merged_dedup:
        logger.info(
            f"  Dedup: total={merged_dedup['total']:,}  "
            f"dropped={merged_dedup['dropped']:,}  "
            f"rate={merged_dedup['dropped']/merged_dedup['total']*100:.2f}%"
        )
    
    # Generate unified report
    report_path = AblationReporter.generate_report(
        merged_results,
        config.io.output_manifest_path,
        dedup_stats=merged_dedup,
    )
    logger.info(f"Unified report saved to: {report_path}")
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Coreset Selection Engine for 70B LLM Pre-training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="config/curriculum.yaml",
        help="Path to curriculum YAML file"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["1B", "3B", "8B", "70B"],
        help="Stages to process (default: all pre-training stages)"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run legacy in-memory builder (not 2T-safe)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Input dataset path (file or directory). Required unless --legacy."
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Input dataset format for streaming mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows/chunks per batch in streaming mode"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (enables resume)"
    )
    parser.add_argument(
        "--total-input-tokens-estimate",
        type=int,
        default=None,
        help="Estimated total input tokens (e.g., 2000000000000 for 2T). Enables proportional per-batch selection budgets."
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id for multi-node runs (0..num_shards-1). Shards files deterministically."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shards for multi-node runs"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows/chunks to read (debug)"
    )
    parser.add_argument(
        "--stage-target-scale",
        type=float,
        default=1.0,
        help="Scale curriculum stage target tokens by this factor (useful for end-to-end runs on small samples)"
    )
    parser.add_argument(
        "--ablation-variant",
        type=str,
        default="baseline",
        help="Ablation variant (baseline, no_dedup, no_diversity, density_only)"
    )
    parser.add_argument(
        "--merge-shard-reports",
        action="store_true",
        help="Merge per-shard result JSONs into a single ablation report, then exit."
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Coreset Selection Engine v1.0.0")
    logger.info("=" * 70)
    
    try:
        # --- Merge mode: combine shard JSONs into one report, then exit ---
        if args.merge_shard_reports:
            return merge_shard_reports(args)
        
        # Validate file paths
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        if not Path(args.curriculum).exists():
            raise FileNotFoundError(f"Curriculum not found: {args.curriculum}")
        
        # Initialize builder
        if args.legacy:
            builder = CoresetBuilder(args.config, args.curriculum)
        else:
            if not args.input_path:
                raise ValueError("--input-path is required unless --legacy is set")
            builder = StreamingCoresetBuilder(
                args.config,
                args.curriculum,
                input_path=args.input_path,
                input_format=args.input_format,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                total_input_tokens_estimate=args.total_input_tokens_estimate,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                max_rows=args.max_rows,
                stages=args.stages,
                stage_target_scale=args.stage_target_scale,
            )
        
        # Build coresets
        results = builder.build_coresets()

        # Streaming-mode summary (timings + throughput-ish stats)
        if not args.legacy and isinstance(results, dict):
            logger.info("\nStreaming run summary:")
            for stage_name in args.stages:
                r = results.get(stage_name)
                if not r:
                    continue
                timing_totals = r.get("timings_s") or {}
                timing_str = " | ".join(
                    f"{k}={float(timing_totals[k]):.3f}s" for k in sorted(timing_totals.keys())
                )
                logger.info(
                    f"  - {stage_name}: seen_tokens={int(r.get('total_tokens_seen', 0)):,} "
                    f"selected_tokens={int(r.get('selected_tokens', 0)):,} "
                    f"selected_chunks={int(r.get('selected_chunks', 0)):,} "
                    f"parts={int(r.get('parts_written', 0)):,}"
                )
                if timing_str:
                    logger.info(f"    timings: {timing_str}")
        
        # Generate reports for both legacy and streaming runs
        # ---- GLOBAL DEDUP SUMMARY ----
        if not args.legacy and hasattr(builder, '_global_dedup_total') and builder._global_dedup_total > 0:
            dedup_rate = builder._global_dedup_dropped / builder._global_dedup_total * 100
            logger.info("\n--- Global Hash Dedup Stats ---")
            logger.info(f"  Total rows with hash  : {builder._global_dedup_total:,}")
            logger.info(f"  Unique rows kept      : {builder._global_dedup_total - builder._global_dedup_dropped:,}")
            logger.info(f"  Duplicate rows dropped: {builder._global_dedup_dropped:,}")
            logger.info(f"  Dedup rate            : {dedup_rate:.2f}%")

        builder.generate_reports(results)
        
        logger.info("\n" + "=" * 70)
        logger.info("Coreset selection pipeline completed successfully!")
        logger.info("=" * 70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
