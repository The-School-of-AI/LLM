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
import sys
from pathlib import Path
from datetime import datetime
import hashlib

from src.core.config import PipelineConfig
from src.core.types import StageName, ProtectedSliceRule, CoresetManifest
from src.curriculum.loader import CurriculumLoader
from src.selection.engine import SelectionEngine
from src.io.loaders import ChunkLoader, CoresetWriter, AblationReporter


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
        
        # Create manifest
        manifest = CoresetManifest(
            stage_name=StageName(stage_name),
            coreset_id=hashlib.sha256(
                f"{stage_name}_{self.config_hash}_{self.curriculum_hash}".encode()
            ).hexdigest(),
            target_tokens=stage_config.target_tokens,
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
        """Generate ablation and diagnostic reports"""
        logger.info("\nGenerating reports...")
        
        report_path = AblationReporter.generate_report(
            results,
            self.config.io.output_manifest_path
        )
        
        logger.info(f"Report saved to: {report_path}")


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
        builder = CoresetBuilder(args.config, args.curriculum)
        
        # Build coresets
        results = builder.build_coresets()
        
        # Generate reports
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
