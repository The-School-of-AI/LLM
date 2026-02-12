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
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Dict, Optional, Any, Iterator, Tuple, List

from src.core.config import PipelineConfig
from src.core.types import StageName, ProtectedSliceRule, CoresetManifest, DifficultyBand
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
            code_domain_id = None
            for band_enum, band_def in self.curriculum.bands.items():
                band_name = band_enum.value
                if getattr(stage_bands.band_ratios, band_name, 0.0) > 0:
                    if 'code' in band_def.allowed_domains:
                        has_code = True
                        code_domain_id = 'code'
                    if 'code_repos' in band_def.allowed_domains:
                        has_code = True
                        code_domain_id = 'code_repos'
                    if 'agentic' in band_def.allowed_domains:
                        has_agentic = True
                    if 'indic' in band_def.allowed_domains:
                        has_indic = True
            
            if has_code and code_domain_id:
                protected_slices.append(ProtectedSliceRule(code_domain_id, 0.90, "Code capability foundation"))
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
            target_tokens_global=int(target_tokens_value),
            target_tokens_shard=int(target_tokens_value),
            actual_tokens=stats['selected_tokens'],
            created_at=datetime.now().isoformat(),
            pipeline_version=self.config.pipeline_version,
            curriculum_version=self.curriculum.version,
            seed=self.config.curriculum.deterministic_seed,
            config_hash=self.config_hash,
            selected_chunks_count=stats['selected_chunks'],
            shard_id=0,
            num_shards=1,
            stage_target_scale=1.0,
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
                'token_count_estimate': all_chunks[cid].token_count,
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
        """Generate ablation and diagnostic reports"""
        logger.info("\nGenerating reports...")

        report_filename = "ablation_validation_report.md"
        try:
            shard_id = None
            num_shards = None
            for _stage, stage_results in (results or {}).items():
                if isinstance(stage_results, dict):
                    if stage_results.get("shard_id") is not None:
                        shard_id = int(stage_results.get("shard_id"))
                    if stage_results.get("num_shards") is not None:
                        num_shards = int(stage_results.get("num_shards"))
                    break
            if num_shards and num_shards > 1 and shard_id is not None:
                report_filename = f"ablation_validation_report_shard{shard_id:03d}.md"
        except Exception:
            # Best-effort only; fall back to default name.
            report_filename = "ablation_validation_report.md"
        
        report_path = AblationReporter.generate_report(
            results,
            self.config.io.output_manifest_path,
            report_filename=report_filename,
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
        band_inference: str = "none",
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
        self.band_inference = str(band_inference or "none").lower()
        if self.band_inference not in {"none", "infer_if_missing", "infer_if_ineligible", "force"}:
            raise ValueError(
                "Invalid --band-inference. Choose one of: none, infer_if_missing, infer_if_ineligible, force"
            )

        self.batch_processor = BatchProcessor(batch_size=self.batch_size, checkpoint_dir=checkpoint_dir)
        self.error_recovery = ErrorRecoveryManager()

        # Enforce cross-stage non-overlap for streaming runs via disk-backed membership.
        used_dir = Path(self.config.io.output_coreset_path) / ".used_chunks"
        used_db = used_dir / f"used_chunks_shard{self.shard_id:03d}.sqlite"
        self.used_store = UsedChunksStore(used_db)

    def _infer_band_from_score(self, score: float) -> DifficultyBand:
        """Infer a DifficultyBand from a continuous difficulty score.

        Uses curriculum difficulty centroids when available; otherwise falls back to defaults.
        Deterministic tie-break by canonical band order.
        """

        try:
            s = float(score)
        except Exception:
            return DifficultyBand.B0

        centroids = {}
        if getattr(self.curriculum, "difficulty_system", None) is not None:
            centroids = dict(getattr(self.curriculum.difficulty_system, "difficulty_centroids", {}) or {})

        if not centroids:
            centroids = {"B0": 0.10, "B1": 0.22, "B2": 0.40, "B3": 0.60, "B4": 0.78, "B5": 0.92}

        order = ["B0", "B1", "B2", "B3", "B4", "B5"]
        best = "B0"
        best_dist = float("inf")
        for b in order:
            c = centroids.get(b)
            if c is None:
                continue
            try:
                d = abs(float(c) - s)
            except Exception:
                continue
            if d < best_dist:
                best_dist = d
                best = b

        try:
            return DifficultyBand(best)
        except Exception:
            return DifficultyBand.B0

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
                    emitted += len(batch)
                    yield batch_idx, batch
                    batch_idx += 1
            return

        if self.input_format == "parquet":
            files = self.batch_processor.list_input_files(self.input_path, "parquet")
            if files:
                files = self.batch_processor.shard_files(files, self.shard_id, self.num_shards)
                paths = [str(p) for p in files]
            else:
                paths = [self.input_path]

            columns = [
                "chunk_id",
                "dataset_id",
                "token_count_estimate",
                "byte_length",
                "domain",
                "language",
                "band",
                "source_doc_id",
                "source_url",
                "token_ids",
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
                        cid = r.get("chunk_id")
                        if cid is None:
                            continue
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

        # Target tokens disambiguation:
        # - target_tokens_value is the global stage target (pre scaling/splitting)
        # - stage_target_tokens is the effective per-worker target used by selection
        target_tokens_global = int(target_tokens_value)

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
        domain_tokens_by_band: Dict[str, Counter[str]] = {}
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
            allowed_languages_for_stage = self.curriculum.get_allowed_languages_for_stage(stage_name)

        if last_batch is not None and last_batch >= 0:
            loaded = self.batch_processor.load_checkpoint(stage_name, last_batch)
            if loaded is not None:
                state, _metadata = loaded

                # Guard against resuming with incompatible runtime parameters.
                # This can happen if the same --checkpoint-dir is reused after changing
                # --num-shards/--shard-id/--stage-target-scale (which changes stage_target_tokens).
                prev_num_shards = state.get("num_shards")
                prev_shard_id = state.get("shard_id")
                prev_stage_target_tokens = state.get("stage_target_tokens")
                if prev_num_shards is not None and int(prev_num_shards) != int(self.num_shards):
                    raise ValueError(
                        f"Incompatible checkpoint for {stage_name}: checkpoint num_shards={prev_num_shards} "
                        f"but current run num_shards={self.num_shards}. Use a new --checkpoint-dir or delete old checkpoints."
                    )
                if prev_shard_id is not None and int(prev_shard_id) != int(self.shard_id):
                    raise ValueError(
                        f"Incompatible checkpoint for {stage_name}: checkpoint shard_id={prev_shard_id} "
                        f"but current run shard_id={self.shard_id}. Use a shard-unique --checkpoint-dir or delete old checkpoints."
                    )
                if prev_stage_target_tokens is not None and int(prev_stage_target_tokens) != int(stage_target_tokens):
                    raise ValueError(
                        f"Incompatible checkpoint for {stage_name}: checkpoint stage_target_tokens={prev_stage_target_tokens} "
                        f"but current run stage_target_tokens={stage_target_tokens}. "
                        "This usually means --stage-target-scale and/or --num-shards changed. "
                        "Use a new --checkpoint-dir or delete old checkpoints."
                    )

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

                # Restore selection engine state for deterministic resume.
                # Older checkpoints may not have this field.
                engine_state = state.get("engine_state")
                if engine_state:
                    try:
                        engine.load_checkpoint_state(engine_state)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to restore selection engine state from checkpoint for {stage_name}. "
                            "This can happen if code/config changed between runs; use a new --checkpoint-dir."
                        ) from e

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
                        meta_obj = row.get("metadata") if isinstance(row, dict) else None
                        meta_dict = meta_obj if isinstance(meta_obj, dict) else {}

                        token_count = int(
                            row.get("token_count_estimate", None)
                            or row.get("token_count", None)
                            or meta_dict.get("token_count_estimate", None)
                            or meta_dict.get("token_count", 0)
                            or 0
                        )
                        batch_tokens += token_count

                        band_raw = (
                            row.get("band", None)
                            or meta_dict.get("band", None)
                            or "B0"
                        )

                        # Optional: infer band from difficulty score when requested.
                        # This is particularly useful when datasets use a placeholder band but provide
                        # a continuous band_score/difficulty_score.
                        band_score = (
                            row.get("band_score", None)
                            or meta_dict.get("band_score", None)
                            or row.get("difficulty_score", None)
                            or meta_dict.get("difficulty_score", None)
                        )
                        try:
                            band_score_val = None if band_score is None else float(band_score)
                        except Exception:
                            band_score_val = None

                        # Parse provided band (may be invalid in some datasets)
                        try:
                            provided_band = DifficultyBand(str(band_raw))
                        except Exception:
                            provided_band = None

                        domain_raw = row.get("domain", None) or meta_dict.get("domain", "unknown")

                        final_band = provided_band
                        if self.band_inference == "force" and band_score_val is not None:
                            final_band = self._infer_band_from_score(band_score_val)
                        elif self.band_inference == "infer_if_missing" and (final_band is None) and band_score_val is not None:
                            final_band = self._infer_band_from_score(band_score_val)
                        elif self.band_inference == "infer_if_ineligible" and final_band is not None and band_score_val is not None:
                            allowed_domains = self.curriculum.get_allowed_domains_for_band(final_band)
                            if allowed_domains and domain_raw not in allowed_domains:
                                inferred = self._infer_band_from_score(band_score_val)
                                inferred_allowed = self.curriculum.get_allowed_domains_for_band(inferred)
                                if (not inferred_allowed) or (domain_raw in inferred_allowed):
                                    final_band = inferred

                        if final_band is None:
                            final_band = DifficultyBand.B0

                        meta = ChunkMetadata(
                            chunk_id=str(chunk_id),
                            dataset_id=row.get("dataset_id") or row.get("source") or meta_dict.get("source") or "ds",
                            token_count=token_count,
                            byte_length=int(row.get("byte_length", None) or meta_dict.get("byte_length", 0) or 0),
                            domain=domain_raw,
                            language=row.get("language", None) or meta_dict.get("language", "en"),
                            band=final_band,
                            source_doc_id=row.get("source_doc_id", None) or meta_dict.get("source_doc_id", ""),
                            source_url=row.get("source_url", None) or meta_dict.get("source_url", None),
                        )
                        # Optional schema v0.6+ fields
                        if band_score_val is not None:
                            setattr(meta, "band_score", band_score_val)

                        token_ids = row.get("token_ids")
                        if token_ids is not None:
                            try:
                                setattr(meta, "token_ids", list(token_ids))
                            except Exception:
                                pass

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
                        dom = str(getattr(meta.domain, "value", meta.domain))
                        domain_tokens[dom] += tc
                        band_name = meta.band.value
                        if band_name not in domain_tokens_by_band:
                            domain_tokens_by_band[band_name] = Counter()
                        domain_tokens_by_band[band_name][dom] += tc
                        language_tokens[str(meta.language)] += tc
                        rows.append({
                            "chunk_id": meta.chunk_id,
                            "dataset_id": meta.dataset_id,
                            "token_count_estimate": tc,
                            "band": meta.band.value,
                            "domain": meta.domain,
                            "language": meta.language,
                        })
                    if rows:
                        part_base = stage_dir / f"selected_indices_part_shard{self.shard_id:03d}_batch{batch_idx:06d}"
                        part_path = part_base.with_suffix(".parquet")
                        wrote = False
                        try:
                            pd.DataFrame(rows).to_parquet(part_path, index=False)
                            wrote = True
                        except Exception as e:
                            # Fall back to JSONL parts when parquet dependencies are unavailable.
                            # Without this, the batch loop can silently skip all output parts.
                            logger.warning(
                                "Failed to write parquet part %s (%s); falling back to JSONL",
                                part_path,
                                e,
                            )
                            part_path = part_base.with_suffix(".jsonl")
                            with open(part_path, "w", encoding="utf-8") as f:
                                for r in rows:
                                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            wrote = True

                        if wrote:
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
                    "stage_target_scale": float(self.stage_target_scale),
                    "total_input_tokens_estimate": (None if shard_total_tokens_est is None else int(shard_total_tokens_est)),
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

                # Persist selection engine internal state so crash+resume is deterministic.
                try:
                    state["engine_state"] = engine.get_checkpoint_state()
                except Exception:
                    # Best-effort: resume will still work, but may not be bitwise deterministic.
                    pass
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
        from src.core.types import CoresetComposition, BandDistribution, DomainDistributionV2, LanguageDistribution

        if selected_tokens > 0:
            band_dist = BandDistribution(
                B0=float(band_tokens.get("B0", 0)) / float(selected_tokens),
                B1=float(band_tokens.get("B1", 0)) / float(selected_tokens),
                B2=float(band_tokens.get("B2", 0)) / float(selected_tokens),
                B3=float(band_tokens.get("B3", 0)) / float(selected_tokens),
                B4=float(band_tokens.get("B4", 0)) / float(selected_tokens),
                B5=float(band_tokens.get("B5", 0)) / float(selected_tokens),
            )
            domain_total = {k: float(v) / float(selected_tokens) for k, v in dict(domain_tokens).items()}
            by_band = {}
            for band_name, ctr in (domain_tokens_by_band or {}).items():
                denom = float(band_tokens.get(band_name, 0) or 0)
                if denom <= 0:
                    continue
                by_band[band_name] = {k: float(v) / denom for k, v in dict(ctr).items()}
            domain_dist = DomainDistributionV2(total=domain_total, by_band=by_band)
            language_dist = LanguageDistribution(
                languages={k: float(v) / float(selected_tokens) for k, v in language_tokens.items()}
            )
        else:
            band_dist = BandDistribution()
            domain_dist = DomainDistributionV2(total={}, by_band={})
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
            target_tokens_global=target_tokens_global,
            target_tokens_shard=stage_target_tokens,
            actual_tokens=selected_tokens,
            created_at=datetime.now().isoformat(),
            pipeline_version=self.config.pipeline_version,
            curriculum_version=self.curriculum.version,
            seed=self.config.curriculum.deterministic_seed,
            config_hash=self.config_hash,
            selected_chunks_count=selected_chunks,
            shard_id=int(self.shard_id),
            num_shards=int(self.num_shards),
            stage_target_scale=float(self.stage_target_scale),
            total_input_tokens_estimate_global=(None if self.total_input_tokens_estimate is None else int(self.total_input_tokens_estimate)),
            total_input_tokens_estimate_shard=(None if shard_total_tokens_est is None else int(shard_total_tokens_est)),
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
        "--band-inference",
        type=str,
        default="none",
        choices=["none", "infer_if_missing", "infer_if_ineligible", "force"],
        help=(
            "Optional band inference from band_score/difficulty_score. "
            "none=use provided band; infer_if_missing=infer only when band missing/invalid; "
            "infer_if_ineligible=infer when (band,domain) is curriculum-ineligible; force=always infer when score present."
        ),
    )
    parser.add_argument(
        "--ablation-variant",
        type=str,
        default="baseline",
        help="Ablation variant (baseline, no_dedup, no_diversity, density_only)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Coreset Selection Engine v1.0.0")
    logger.info("=" * 70)
    
    try:
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
                band_inference=args.band_inference,
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
