"""
I/O utilities for loading and saving data.
Supports filesystem and object store (S3/GCS) backends.
"""

from typing import Dict, List, Iterator, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from dataclasses import asdict
import concurrent.futures
import pandas as pd

from ..core.types import ChunkMetadata, CoresetManifest, DifficultyBand

logger = logging.getLogger(__name__)


class ChunkLoader:
    """Load chunks from various sources"""
    
    def __init__(self, base_path: str, use_object_store: bool = False,
                 object_store_type: Optional[str] = None,
                 object_store_bucket: Optional[str] = None,
                 num_parallel_loaders: int = 16):
        self.base_path = Path(base_path)
        self.use_object_store = use_object_store
        self.object_store_type = object_store_type
        self.object_store_bucket = object_store_bucket
        self.num_parallel_loaders = int(num_parallel_loaders or 1)
        
        if use_object_store and object_store_type == "s3":
            try:
                import boto3
                self.s3_client = boto3.client('s3')
            except ImportError:
                logger.warning("boto3 not available, falling back to filesystem")
                self.use_object_store = False
    
    def load_chunks_from_jsonl(self, filepath: str, max_chunks: Optional[int] = None) -> Iterator[Tuple[str, ChunkMetadata]]:
        """
        Load chunks from JSONL file.
        
        Yields:
            (chunk_id, ChunkMetadata)
        """
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                if max_chunks and count >= max_chunks:
                    break
                
                try:
                    data = json.loads(line)
                    chunk_id = data.get('chunk_id')
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        dataset_id=data.get('dataset_id'),
                        token_count=data.get('token_count', 0),
                        byte_length=data.get('byte_length', 0),
                        domain=data.get('domain', 'unknown'),
                        language=data.get('language', 'en'),
                        band=DifficultyBand(data.get('band', 'B0')),
                        source_doc_id=data.get('source_doc_id'),
                        source_url=data.get('source_url'),
                        quality_flags=data.get('quality_flags', []),
                        sensitive_markers=data.get('sensitive_markers', []),
                        start_offset=data.get('start_offset', 0),
                    )
                    # Attach optional token ids if present (token->chunk mapping)
                    token_ids = data.get('token_ids')
                    if token_ids is not None:
                        setattr(metadata, 'token_ids', list(token_ids))

                    yield chunk_id, metadata
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse chunk: {e}")
                    continue
    
    def load_chunks_from_parquet(self, filepath: str, max_chunks: Optional[int] = None) -> Iterator[Tuple[str, ChunkMetadata]]:
        """
                        domains_list = ', '.join(sorted(all_domains)) if all_domains else 'None'
                        report.append(f"- **Domains**: Provides diverse content ({domains_list})\n")
        
        Yields:
            (chunk_id, ChunkMetadata)
        """
        df = pd.read_parquet(filepath)
        
        if max_chunks:
            df = df.head(max_chunks)
        
        for _, row in df.iterrows():
            try:
                metadata = ChunkMetadata(
                    chunk_id=row['chunk_id'],
                    dataset_id=row['dataset_id'],
                    token_count=int(row['token_count']),
                    byte_length=int(row['byte_length']),
                    domain=row['domain'],
                    language=row['language'],
                    band=DifficultyBand(row['band']),
                    source_doc_id=row['source_doc_id'],
                    source_url=row.get('source_url'),
                    quality_flags=row.get('quality_flags', []),
                    sensitive_markers=row.get('sensitive_markers', []),
                    start_offset=int(row.get('start_offset', 0)),
                )
                # If parquet contains token ids column, attach it
                if 'token_ids' in row.index and row['token_ids'] is not None:
                    try:
                        setattr(metadata, 'token_ids', list(row['token_ids']))
                    except Exception:
                        pass
                
                yield row['chunk_id'], metadata
            except Exception as e:
                logger.warning(f"Failed to parse chunk: {e}")
                continue
    
    def load_all_chunks(self, dataset_id: Optional[str] = None) -> Dict[str, ChunkMetadata]:
        """Load all chunks into memory"""
        chunks = {}
        
        if self.use_object_store:
            # TODO: Implement S3/GCS loading
            logger.warning("Object store loading not yet implemented")
            return chunks
        
        # Collect file paths
        parquet_files = list(self.base_path.glob("**/*.parquet"))
        jsonl_files = list(self.base_path.glob("**/*.jsonl"))

        def _load_parquet_file(p: Path) -> Dict[str, ChunkMetadata]:
            out = {}
            try:
                for chunk_id, metadata in self.load_chunks_from_parquet(str(p)):
                    out[chunk_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load parquet {p}: {e}")
            return out

        def _load_jsonl_file(p: Path) -> Dict[str, ChunkMetadata]:
            out = {}
            try:
                for chunk_id, metadata in self.load_chunks_from_jsonl(str(p)):
                    out[chunk_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load jsonl {p}: {e}")
            return out

        # Use ThreadPoolExecutor to load files in parallel
        file_tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_loaders) as exc:
            for p in parquet_files:
                file_tasks.append(exc.submit(_load_parquet_file, p))
            for p in jsonl_files:
                file_tasks.append(exc.submit(_load_jsonl_file, p))

            for fut in concurrent.futures.as_completed(file_tasks):
                try:
                    result = fut.result()
                    chunks.update(result)
                except Exception as e:
                    logger.warning(f"Error loading file in worker: {e}")

        logger.info(f"Loaded {len(chunks)} chunks (from {len(parquet_files)+len(jsonl_files)} files using {self.num_parallel_loaders} workers)")
        return chunks


class CoresetWriter:
    """Write coreset outputs"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def save_selected_indices(self, stage_name: str, selected_chunks: set, 
                             metadata: Dict[str, Any],
                             format: str = "parquet") -> Path:
        """
        Save selected chunk indices.
        
        Args:
            stage_name: Name of training stage (1B, 3B, etc.)
            selected_chunks: Set of selected chunk IDs
            metadata: Additional metadata for each chunk
            format: Output format (parquet or jsonl)
        
        Returns:
            Path to output file
        """
        stage_dir = self.output_path / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Build dataframe
        rows = []
        for chunk_id in sorted(selected_chunks):
            if chunk_id in metadata:
                rows.append({
                    'chunk_id': chunk_id,
                    **metadata[chunk_id]
                })
        
        df = pd.DataFrame(rows)
        
        fmt = format.lower()
        if fmt == "parquet":
            output_file = stage_dir / "selected_indices.parquet"
            df.to_parquet(output_file, index=False)
        elif fmt == "jsonl" or fmt == "json":
            output_file = stage_dir / "selected_indices.jsonl"
            df.to_json(output_file, orient='records', lines=True)
        elif fmt == "csv":
            output_file = stage_dir / "selected_indices.csv"
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output index format: {format}")
        
        logger.info(f"Saved {len(rows)} indices to {output_file}")
        return output_file
    
    def save_manifest(self, manifest: CoresetManifest, stage_name: str) -> Path:
        """Save manifest as JSON"""
        stage_dir = self.output_path / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = stage_dir / "manifest.json"
        
        with open(manifest_path, 'w') as f:
            f.write(manifest.to_json(indent=2))
        
        logger.info(f"Saved manifest to {manifest_path}")
        return manifest_path


class AblationReporter:
    """Generate ablation and validation reports"""
    
    @staticmethod
    def generate_report(stages_results: Dict[str, Dict[str, Any]],
                       output_path: str,
                       dedup_stats: Optional[Dict[str, int]] = None) -> str:
        """
        Generate comprehensive ablation report with:
        - Methods evaluated
        - Achieved reduction ratios (per-stage, accurately)
        - Coverage diagnostics
        - Global dedup stats
        
        Returns:
            Path to generated report
        """
        report = []
        report.append("# Coreset Selection Validation Report\n\n")
        
        # ===== EXECUTIVE SUMMARY =====
        report.append("## Executive Summary\n\n")
        report.append("This report documents coreset selection results including:\n")
        report.append("- Reduction ratios achieved per curriculum stage\n")
        report.append("- Band, domain, and language coverage diagnostics\n")
        report.append("- Global deduplication impact\n\n")
        
        # ===== DATASET OVERVIEW =====
        # The same dataset is re-scanned per stage, so the true dataset size
        # is the MAXIMUM across stages (not the sum).
        stage_inputs = [r.get('total_input_tokens', 0) for r in stages_results.values()]
        stage_chunk_inputs = [r.get('total_input_chunks', 0) for r in stages_results.values()]
        
        true_input_tokens = max(stage_inputs) if stage_inputs else 0
        true_input_chunks = max(stage_chunk_inputs) if stage_chunk_inputs else 0
        num_stages = len(stages_results)
        
        report.append("## Dataset Overview\n\n")
        report.append(f"| Metric | Value |\n")
        report.append(f"|--------|-------|\n")
        report.append(f"| Dataset Size (tokens) | {true_input_tokens:,} |\n")
        report.append(f"| Dataset Size (chunks) | {true_input_chunks:,} |\n")
        report.append(f"| Stages Processed | {num_stages} |\n\n")
        
        # ===== GLOBAL DEDUP IMPACT =====
        if dedup_stats and dedup_stats.get('total', 0) > 0:
            dt = dedup_stats['total']
            dd = dedup_stats['dropped']
            dk = dt - dd
            dr = dd / dt * 100
            report.append("## Global Hash Dedup Impact\n\n")
            report.append(f"| Metric | Value |\n")
            report.append(f"|--------|-------|\n")
            report.append(f"| Total rows with hash | {dt:,} |\n")
            report.append(f"| Unique rows kept | {dk:,} |\n")
            report.append(f"| Duplicate rows dropped | {dd:,} |\n")
            report.append(f"| Dedup rate | {dr:.2f}% |\n\n")
        
        # ===== STAGE-WISE BREAKDOWN =====
        report.append("## Stage-wise Breakdown\n\n")
        for stage_name, results in sorted(stages_results.items()):
            stage_input = results.get('total_input_tokens', 0)
            stage_selected = results.get('selected_tokens', 0)
            stage_chunks_in = results.get('total_input_chunks', 0)
            stage_chunks_out = results.get('selected_chunks', 0)
            stage_ratio = (stage_input / stage_selected) if stage_selected > 0 else None
            
            report.append(f"### {stage_name}\n\n")
            report.append(f"| Metric | Value |\n")
            report.append(f"|--------|-------|\n")
            report.append(f"| Input Tokens | {stage_input:,} |\n")
            report.append(f"| Selected Tokens | {stage_selected:,} |\n")
            if stage_ratio and stage_ratio > 0:
                reduction = 100*(1 - 1/stage_ratio)
                report.append(f"| Compression Ratio | {stage_ratio:.2f}x ({reduction:.1f}% reduction) |\n")
            else:
                report.append(f"| Compression Ratio | N/A (no tokens selected) |\n")
            report.append(f"| Input Chunks | {stage_chunks_in:,} |\n")
            report.append(f"| Selected Chunks | {stage_chunks_out:,} |\n\n")
            
            # Band distribution
            if 'band_distribution' in results:
                band_dist = results['band_distribution']
                report.append("**Band Distribution:**\n\n")
                report.append("| Band | Ratio | Tokens | Coverage |\n")
                report.append("|------|-------|--------|----------|\n")
                for band_name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']:
                    ratio = band_dist.get(band_name, 0.0) if isinstance(band_dist, dict) else getattr(band_dist, band_name, 0.0)
                    band_tokens = int(stage_selected * ratio)
                    report.append(f"| {band_name} | {ratio:.2%} | {band_tokens:,} | {'✓' if ratio > 0 else '-'} |\n")
                report.append("\n")
            
            # Domain distribution
            if 'domain_distribution' in results:
                domain_dist = results['domain_distribution']
                report.append("**Domain Distribution:**\n\n")
                report.append("| Domain | Ratio | Tokens |\n")
                report.append("|--------|-------|--------|\n")
                if isinstance(domain_dist, dict):
                    items = sorted(domain_dist.items())
                else:
                    items = sorted(domain_dist.to_dict().items())
                for domain, ratio in items:
                    if ratio > 0:
                        domain_tokens = int(stage_selected * ratio)
                        report.append(f"| {domain} | {ratio:.2%} | {domain_tokens:,} |\n")
                report.append("\n")
            
            # Language distribution
            if 'language_distribution' in results:
                lang_dist = results['language_distribution']
                report.append("**Language Distribution:**\n\n")
                report.append("| Language | Ratio | Tokens |\n")
                report.append("|----------|-------|--------|\n")
                if isinstance(lang_dist, dict):
                    lang_items = sorted(lang_dist.items())
                else:
                    lang_items = sorted(lang_dist.to_dict().items())
                for lang, ratio in lang_items:
                    if ratio > 0:
                        lang_tokens = int(stage_selected * ratio)
                        report.append(f"| {lang} | {ratio:.2%} | {lang_tokens:,} |\n")
                report.append("\n")
            
            report.append("---\n\n")
        
        # ===== STAGE SUMMARY TABLE =====
        def _fmt_tokens(n):
            """Format large token counts as human-readable (e.g., 4.52B, 630M)."""
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.2f}B"
            elif n >= 1_000_000:
                return f"{n / 1_000_000:.0f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.0f}K"
            return str(n)
        
        if len(stages_results) > 1:
            report.append("## Stage Summary\n\n")
            report.append("| Stage | Input Tokens | Selected Tokens | Compression | Chunks |\n")
            report.append("|-------|-------------|----------------|-------------|--------|\n")
            for stage_name in sorted(stages_results.keys(), key=lambda s: stages_results[s].get('total_input_tokens', 0), reverse=True):
                r = stages_results[stage_name]
                inp = r.get('total_input_tokens', 0)
                sel = r.get('selected_tokens', 0)
                chunks = r.get('selected_chunks', 0)
                ratio = inp / sel if sel > 0 else 0
                pct = (1 - sel / inp) * 100 if inp > 0 else 0
                report.append(f"| {stage_name} | {_fmt_tokens(inp)} | {_fmt_tokens(sel)} | {ratio:.2f}x ({pct:.1f}%) | {chunks:,} |\n")
            report.append("\n")
        
        # ===== COVERAGE DIAGNOSTICS =====
        all_bands = set()
        all_domains = set()
        all_languages = set()
        for results in stages_results.values():
            if 'band_distribution' in results:
                band_dist = results['band_distribution']
                all_bands.update([b for b in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5'] 
                                 if (band_dist.get(b, 0.0) if isinstance(band_dist, dict) else getattr(band_dist, b, 0.0)) > 0])
            if 'domain_distribution' in results:
                dom = results['domain_distribution']
                if isinstance(dom, dict):
                    all_domains.update(k for k, v in dom.items() if v > 0)
                else:
                    all_domains.update(k for k, v in dom.to_dict().items() if v > 0)
            if 'language_distribution' in results:
                lang = results['language_distribution']
                if isinstance(lang, dict):
                    all_languages.update(k for k, v in lang.items() if v > 0)
                else:
                    all_languages.update(k for k, v in lang.to_dict().items() if v > 0)

        report.append("## Coverage Summary\n\n")
        report.append(f"| Dimension | Covered | Details |\n")
        report.append(f"|-----------|---------|--------|\n")
        report.append(f"| Difficulty Bands | {len(all_bands)}/6 | {', '.join(sorted(all_bands))} |\n")
        report.append(f"| Domains | {len(all_domains)} | {', '.join(sorted(all_domains))} |\n")
        report.append(f"| Languages | {len(all_languages)} | {', '.join(sorted(all_languages))} |\n\n")
        
        # ===== SELECTION METHODOLOGY =====
        report.append("## Selection Methodology\n\n")
        report.append("The pipeline applies the following steps in order:\n\n")
        report.append("1. **Global Hash Dedup** — drops rows with previously-seen `hash` values\n")
        report.append("2. **Curriculum Filtering** — enforces band/domain/language policies\n")
        report.append("3. **Diversity Scoring** — prioritizes rare/tail tokens and domain diversity\n")
        report.append("4. **Stratified Sampling** — maintains target band/domain distributions\n")
        report.append("5. **Cross-stage Non-overlap** — ensures no chunk appears in multiple stages\n\n")
        
        # ===== VERSION INFO =====
        report.append("---\n\n")
        report.append("## Reproducibility\n\n")
        report.append(f"- **Output Path**: `{Path(output_path).resolve()}`\n")
        report.append("- **Deterministic**: Yes (fixed seed)\n")
        report.append("- **Configuration**: Tracked via config hash\n")
        
        report_text = "".join(report)
        
        # Write to file
        output_file = Path(output_path) / "ablation_validation_report.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Saved comprehensive ablation report to {output_file}")
        return str(output_file)


# OLD:         """
# OLD:         Generate comprehensive ablation report with:
# OLD:         - Methods evaluated (ablation variants)
# OLD:         - Achieved reduction ratios
# OLD:         - Coverage diagnostics
# OLD:         - Proxy training comparisons
# OLD:         
# OLD:         Returns:
# OLD:             Path to generated report
# OLD:         """
# OLD:         report = []
# OLD:         report.append("# Coreset Selection Ablation & Validation Report\n\n")
# OLD:         
# OLD:         # ===== EXECUTIVE SUMMARY =====
# OLD:         report.append("## Executive Summary\n\n")
# OLD:         report.append("This report documents comprehensive coreset selection results including:\n")
# OLD:         report.append("- Reduction ratios achieved across all curriculum stages\n")
# OLD:         report.append("- Coverage diagnostics and quality metrics\n")
# OLD:         report.append("- Ablation study comparing different selection strategies\n")
# OLD:         report.append("- Proxy training comparisons (coreset vs full dataset baseline)\n\n")
# OLD:         
# OLD:         # ===== OVERALL METRICS =====
# OLD:         total_input = sum(r.get('total_input_tokens', 0) for r in stages_results.values())
# OLD:         total_selected = sum(r.get('selected_tokens', 0) for r in stages_results.values())
# OLD:         total_chunks_input = sum(r.get('total_input_chunks', 0) for r in stages_results.values())
# OLD:         total_chunks_selected = sum(r.get('selected_chunks', 0) for r in stages_results.values())
# OLD:         
# OLD:         compression_ratio = (total_input / total_selected) if total_selected > 0 else None
# OLD:         chunk_reduction = (total_chunks_input / total_chunks_selected) if total_chunks_selected > 0 else None
# OLD:         
# OLD:         report.append("## Overall Reduction Metrics\n\n")
# OLD:         report.append(f"| Metric | Value | Reduction |\n")
# OLD:         report.append(f"|--------|-------|----------|\n")
# OLD:         report.append(f"| Total Input Tokens | {total_input:,} | - |\n")
# OLD:         report.append(f"| Selected Tokens | {total_selected:,} | {100*(1 - total_selected/total_input):.1f}% |\n")
# OLD:         if compression_ratio and compression_ratio > 0:
# OLD:             overall_reduction = 100*(1 - 1/compression_ratio)
# OLD:             report.append(f"| **Compression Ratio** | **{compression_ratio:.2f}x** | **{overall_reduction:.1f}%** |\n")
# OLD:         else:
# OLD:             report.append(f"| **Compression Ratio** | **N/A** | **N/A** |\n")
# OLD:         report.append(f"| Total Input Chunks | {total_chunks_input:,} | - |\n")
# OLD:         report.append(f"| Selected Chunks | {total_chunks_selected:,} | {100*(1 - total_chunks_selected/total_chunks_input):.1f}% |\n")
# OLD:         if chunk_reduction and chunk_reduction > 0:
# OLD:             chunk_reduction_pct = 100*(1 - 1/chunk_reduction)
# OLD:             report.append(f"| **Chunk Reduction** | **{chunk_reduction:.2f}x** | **{chunk_reduction_pct:.1f}%** |\n\n")
# OLD:         else:
# OLD:             report.append(f"| **Chunk Reduction** | **N/A** | **N/A** |\n\n")
# OLD:         
# OLD:         # ===== STAGE-WISE BREAKDOWN =====
# OLD:         report.append("## Stage-wise Breakdown\n\n")
# OLD:         for stage_name, results in sorted(stages_results.items()):
# OLD:             stage_input = results.get('total_input_tokens', 0)
# OLD:             stage_selected = results.get('selected_tokens', 0)
# OLD:             stage_ratio = (stage_input / stage_selected) if stage_selected > 0 else None
# OLD:             
# OLD:             report.append(f"### {stage_name}\n\n")
# OLD:             report.append(f"**Selection Metrics:**\n")
# OLD:             report.append(f"- Input Tokens: {stage_input:,}\n")
# OLD:             report.append(f"- Selected Tokens: {stage_selected:,}\n")
# OLD:             if stage_ratio and stage_ratio > 0:
# OLD:                 reduction = 100*(1 - 1/stage_ratio)
# OLD:                 report.append(f"- Compression Ratio: **{stage_ratio:.2f}x** (reduction: {reduction:.1f}%)\n")
# OLD:             else:
# OLD:                 report.append(f"- Compression Ratio: **N/A** (no selected tokens)\n")
# OLD:             report.append(f"- Selected Chunks: {results.get('selected_chunks', 0):,}\n\n")
# OLD:             
# OLD:             # Band distribution
# OLD:             if 'band_distribution' in results:
# OLD:                 band_dist = results['band_distribution']
# OLD:                 report.append("**Band Distribution** (Difficulty Mix):\n\n")
# OLD:                 report.append("| Band | Ratio | Tokens | Coverage |\n")
# OLD:                 report.append("|------|-------|--------|----------|\n")
# OLD:                 for band_name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']:
# OLD:                     ratio = getattr(band_dist, band_name, 0.0)
# OLD:                     band_tokens = int(stage_selected * ratio)
# OLD:                     report.append(f"| {band_name} | {ratio:.2%} | {band_tokens:,} | {'✓' if ratio > 0 else '-'} |\n")
# OLD:                 report.append("\n")
# OLD:             
# OLD:             # Domain distribution
# OLD:             if 'domain_distribution' in results:
# OLD:                 domain_dist = results['domain_distribution']
# OLD:                 report.append("**Domain Distribution** (Content Diversity):\n\n")
# OLD:                 report.append("| Domain | Ratio | Tokens |\n")
# OLD:                 report.append("|--------|-------|--------|\n")
# OLD:                 # support both dicts and DomainDistribution objects
# OLD:                 if isinstance(domain_dist, dict):
# OLD:                     items = sorted(domain_dist.items())
# OLD:                 else:
# OLD:                     items = sorted(domain_dist.to_dict().items())
# OLD: 
# OLD:                 for domain, ratio in items:
# OLD:                     domain_tokens = int(stage_selected * ratio)
# OLD:                     report.append(f"| {domain} | {ratio:.2%} | {domain_tokens:,} |\n")
# OLD:                 report.append("\n")
# OLD:             
# OLD:             # Language distribution
# OLD:             if 'language_distribution' in results:
# OLD:                 lang_dist = results['language_distribution']
# OLD:                 report.append("**Language Distribution** (Linguistic Coverage):\n\n")
# OLD:                 report.append("| Language | Ratio | Tokens |\n")
# OLD:                 report.append("|----------|-------|--------|\n")
# OLD:                 # support both dicts and LanguageDistribution objects
# OLD:                 if isinstance(lang_dist, dict):
# OLD:                     lang_items = sorted(lang_dist.items())
# OLD:                 else:
# OLD:                     lang_items = sorted(lang_dist.to_dict().items())
# OLD: 
# OLD:                 for lang, ratio in lang_items:
# OLD:                     lang_tokens = int(stage_selected * ratio)
# OLD:                     report.append(f"| {lang} | {ratio:.2%} | {lang_tokens:,} |\n")
# OLD:                 report.append("\n")
# OLD:             
# OLD:             report.append("---\n\n")
# OLD:         
# OLD:         # ===== COVERAGE DIAGNOSTICS =====
# OLD:         # Pre-compute coverage sets from stage results so diagnostics reflect actual data
# OLD:         all_bands = set()
# OLD:         all_domains = set()
# OLD:         all_languages = set()
# OLD:         for results in stages_results.values():
# OLD:             if 'band_distribution' in results:
# OLD:                 band_dist = results['band_distribution']
# OLD:                 all_bands.update([b for b in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5'] 
# OLD:                                  if getattr(band_dist, b, 0.0) > 0])
# OLD:             if 'domain_distribution' in results:
# OLD:                 dom = results['domain_distribution']
# OLD:                 # support both dicts and DomainDistribution objects
# OLD:                 if isinstance(dom, dict):
# OLD:                     all_domains.update(dom.keys())
# OLD:                 else:
# OLD:                     all_domains.update([k for k, v in dom.to_dict().items() if v > 0])
# OLD:             if 'language_distribution' in results:
# OLD:                 lang = results['language_distribution']
# OLD:                 if isinstance(lang, dict):
# OLD:                     all_languages.update(lang.keys())
# OLD:                 else:
# OLD:                     all_languages.update(lang.to_dict().keys())
# OLD: 
# OLD:         report.append("## Coverage Diagnostics\n\n")
# OLD:         report.append("### Curriculum Adherence\n\n")
# OLD:         report.append("The selection maintains target distributions for:\n")
# OLD:         report.append("- **Difficulty Bands (B0-B5)**: Ensures learning progression from easy to hard examples\n")
# OLD:         domains_list = ', '.join(sorted(all_domains)) if all_domains else 'None'
# OLD:         report.append(f"- **Domains**: Provides diverse content ({domains_list})\n")
# OLD:         langs_list = ', '.join(sorted(all_languages)) if all_languages else 'None'
# OLD:         report.append(f"- **Languages**: Covers target languages ({langs_list})\n\n")
# OLD: 
# OLD:         report.append("### Coverage Achievement\n\n")
# OLD:         report.append(f"- **Difficulty Bands Covered**: {len(all_bands)}/6 bands ({', '.join(sorted(all_bands))})\n")
# OLD:         report.append(f"- **Domains Covered**: {len(all_domains)} domains ({', '.join(sorted(all_domains))})\n")
# OLD:         report.append(f"- **Languages Covered**: {len(all_languages)} languages ({', '.join(sorted(all_languages))})\n\n")
# OLD:         
# OLD:         # ===== METHODS EVALUATED =====
# OLD:         report.append("## Methods Evaluated\n\n")
# OLD:         report.append("### Core Selection Strategy\n\n")
# OLD:         report.append("**Stratified Density-Aware Selection** with the following components:\n\n")
# OLD:         report.append("1. **Deduplication**\n")
# OLD:         report.append("   - Exact deduplication: Removes byte-identical chunks\n")
# OLD:         report.append("   - Near-deduplication: Filters similar chunks (SimHash threshold: 0.85)\n")
# OLD:         report.append("   - Impact: Reduces redundancy while preserving diversity\n\n")
# OLD:         
# OLD:         report.append("2. **Diversity Scoring**\n")
# OLD:         report.append("   - Token frequency analysis: Prioritizes rare/tail tokens\n")
# OLD:         report.append("   - Rare token boost: 1.5x weight on 80-95th percentile tokens\n")
# OLD:         report.append("   - Tail token boost: 2.0x weight on 95-100th percentile tokens\n")
# OLD:         report.append("   - Domain diversity weight: 0.3 (bonus for new domains)\n")
# OLD:         report.append("   - Language diversity weight: 0.2 (bonus for new languages)\n\n")
# OLD:         
# OLD:         report.append("3. **Stratified Curriculum Sampling**\n")
# OLD:         report.append("   - Enforces band distribution: Ensures proper difficulty mix\n")
# OLD:         report.append("   - Domain preservation: Maintains content diversity\n")
# OLD:         report.append("   - Language coverage: Targets specified language ratios\n")
# OLD:         report.append("   - Protected slice enforcement: Preserves high-quality subsets (B4, B5, code, agentic, indic)\n\n")
# OLD:         
# OLD:         report.append("4. **Non-overlap Enforcement**\n")
# OLD:         report.append("   - Ensures disjoint stage coreset: No chunk selected for multiple stages\n")
# OLD:         report.append("   - Prevents data leakage between curriculum stages\n\n")
# OLD:         
# OLD:         report.append("### Ablation Variants Evaluated\n\n")
# OLD:         report.append("| Variant | Key Changes | Expected Impact |\n")
# OLD:         report.append("|---------|------------|----------|\n")
# OLD:         report.append("| Baseline | Full pipeline with all components | Balanced selection |\n")
# OLD:         report.append("| No Near-Dedup | Dedup disabled (only exact matches removed) | Higher redundancy, larger size |\n")
# OLD:         report.append("| No Diversity | Uniform sampling (diversity scoring disabled) | Less rare/tail token coverage |\n")
# OLD:         report.append("| High Compression | Aggressive sampling ratio | Smaller coreset, potential quality loss |\n\n")
# OLD:         
# OLD:         # ===== PROXY TRAINING COMPARISON =====
# OLD:         report.append("## Proxy Training Comparisons\n\n")
# OLD:         report.append("### Coreset vs Full Dataset\n\n")
# OLD:         report.append("**Estimated Training Efficiency Gains:**\n\n")
# OLD:         
# OLD:         if total_input > 0 and total_selected > 0:
# OLD:             speedup = total_input / total_selected
# OLD:             report.append(f"| Metric | Full Dataset | Coreset | Improvement |\n")
# OLD:             report.append(f"|--------|-------------|---------|----------|\n")
# OLD:             report.append(f"| Tokens Processed | {total_input:,} | {total_selected:,} | {speedup:.2f}x faster |\n")
# OLD:             report.append(f"| Training Time (est.) | ~{total_input/1e9:.1f}B tokens | ~{total_selected/1e9:.1f}B tokens | **{100*(1 - 1/speedup):.1f}% reduction** |\n")
# OLD:             report.append(f"| Compute Cost (est.) | 100% | {100/speedup:.1f}% | {100*(1 - 1/speedup):.1f}% savings |\n")
# OLD:             report.append(f"| Convergence Speed | Baseline | ~{speedup:.1f}x faster | Expected {speedup:.1f}x speedup |\n\n")
# OLD:             
# OLD:             report.append("**Expected Quality Trade-offs:**\n\n")
# OLD:             report.append(f"- Training time reduction: **{100*(1 - 1/speedup):.1f}%**\n")
# OLD:             report.append(f"- Compute cost reduction: **~{100*(1 - 1/speedup):.1f}%**\n")
# OLD:             report.append(f"- Estimated quality retention: **85-95%** (based on diversity coverage)\n")
# OLD:             report.append(f"- Quality loss (estimated): **5-15%** due to dataset reduction\n\n")
# OLD:             
# OLD:             report.append("### Effectiveness Metrics\n\n")
# OLD:             report.append(f"- **Coverage Score**: {100 * min(1.0, len(all_domains)/6):.1f}% domain coverage\n")
# OLD:             report.append(f"- **Difficulty Balance**: All {len(all_bands)} bands represented\n")
# OLD:             report.append(f"- **Linguistic Diversity**: {len(all_languages)} languages covered\n\n")
# OLD:         
# OLD:         # ===== DEDUPLICATION IMPACT =====
# OLD:         report.append("## Deduplication Impact\n\n")
# OLD:         total_dedup_removed = sum(r.get('exact_dedup_removed', 0) + r.get('near_dedup_removed', 0) 
# OLD:                                  for r in stages_results.values())
# OLD:         total_before_dedup = total_chunks_input
# OLD:         if total_before_dedup > 0:
# OLD:             dedup_ratio = (total_before_dedup - total_dedup_removed) / total_before_dedup
# OLD:             report.append(f"- Chunks removed by deduplication: {total_dedup_removed:,} ({100*(1 - dedup_ratio):.2f}%)\n")
# OLD:             report.append(f"- Chunks retained: {total_dedup_removed:,} ({dedup_ratio:.2%})\n")
# OLD:             report.append(f"- Redundancy elimination: Improved data quality without additional storage\n\n")
# OLD:         
# OLD:         # ===== RECOMMENDATIONS =====
# OLD:         report.append("## Recommendations\n\n")
# OLD:         report.append("1. **For Production Deployment**:\n")
# OLD:         if compression_ratio and compression_ratio > 0:
# OLD:             report.append(f"   - Use baseline coreset with {compression_ratio:.2f}x compression\n")
# OLD:             report.append(f"   - Expect {100*(1 - 1/compression_ratio):.1f}% training time reduction\n")
# OLD:         else:
# OLD:             report.append("   - Use baseline coreset (compression ratio unavailable for empty selection)\n")
# OLD:         report.append(f"   - All coverage targets met: {len(all_bands)} bands, {len(all_domains)} domains, {len(all_languages)} languages\n\n")
# OLD:         
# OLD:         report.append("2. **For Maximum Compression**:\n")
# OLD:         report.append("   - Use 'High Compression' variant from ablation\n")
# OLD:         report.append("   - Trade-off: Faster training at potential quality cost\n\n")
# OLD:         
# OLD:         report.append("3. **For Quality Assurance**:\n")
# OLD:         report.append("   - Validate on held-out test set\n")
# OLD:         report.append("   - Compare model performance: coreset-trained vs full-dataset-trained\n")
# OLD:         report.append("   - Adjust compression ratios based on quality metrics\n\n")
# OLD:         
# OLD:         # ===== VERSION INFO =====
# OLD:         report.append("---\n\n")
# OLD:         report.append("## Version & Reproducibility\n\n")
# OLD:         report.append(f"- **Report Generated**: {Path(output_path).resolve()}\n")
# OLD:         report.append("- **Reproducibility**: Deterministic seed ensures same results across runs\n")
# OLD:         report.append("- **Configuration**: All settings tracked in config hash\n")
# OLD:         
# OLD:         report_text = "".join(report)
# OLD:         
# OLD:         # Write to file
# OLD:         output_file = Path(output_path) / "ablation_validation_report.md"
# OLD:         output_file.parent.mkdir(parents=True, exist_ok=True)
# OLD:         
# OLD:         with open(output_file, 'w', encoding='utf-8') as f:
# OLD:             f.write(report_text)
# OLD:         
# OLD:         logger.info(f"Saved comprehensive ablation report to {output_file}")
# OLD:         return str(output_file)
