"""
Core selection engine - main orchestrator for coreset selection.
Implements stratified, density-aware selection with curriculum compliance.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.config import PipelineConfig
from ..core.types import (
    BandDistribution,
    ChunkMetadata,
    DifficultyBand,
    LanguageDistribution,
    ProtectedSliceRule,
    difficulty_band_order,
)
from ..curriculum.loader import CurriculumLoader
from ..dedup.deduplicator import ExactDeduplicator, NearDeduplicator
from ..diversity.scorer import (
    DiversityScorer,
    DomainDiversityMatrix,
    LanguageDiversityMatrix,
    ProtectedSliceManager,
    TokenFrequencyAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkBucket:
    """Bucket for stratified sampling within a band+domain combination"""

    band: DifficultyBand
    domain: str
    chunks: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    target_tokens: int = 0
    current_tokens: int = 0


class SelectionEngine:
    """
    Main coreset selection engine.
    Orchestrates deduplication, diversity scoring, and deterministic selection.
    """

    def __init__(self, config: PipelineConfig, curriculum: CurriculumLoader):
        self.config = config
        self.curriculum = curriculum
        self.logger = logging.getLogger(__name__)

        # Reproducibility
        self.seed = config.curriculum.deterministic_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize components
        self.exact_dedup = ExactDeduplicator(hash_algorithm=config.dedup.hash_algorithm)
        self.near_dedup = None
        if config.dedup.enable_near_dedup:
            self.near_dedup = NearDeduplicator(
                strategy="simhash", threshold=config.dedup.near_dedup_threshold
            )

        # TBD:Disabled since token ids at document level are not being computed
        self.token_analyzer = TokenFrequencyAnalyzer(vocab_size=128_000)
        self.diversity_scorer = DiversityScorer(
            self.token_analyzer,
            rare_token_boost=config.diversity.rare_token_boost,
            tail_token_boost=config.diversity.tail_token_boost,
            domain_diversity_weight=config.diversity.domain_diversity_weight,
            language_diversity_weight=config.diversity.language_diversity_weight,
        )

        self.domain_diversity = DomainDiversityMatrix()
        self.language_diversity = LanguageDiversityMatrix()
        self.protected_slice_manager = ProtectedSliceManager()

        # Selection tracking
        self.selected_chunks: Set[str] = set()
        self.removed_chunks: Set[str] = set()
        self.buckets: Dict[Tuple[DifficultyBand, str], ChunkBucket] = {}

        # Avoid spamming the same curriculum warnings repeatedly across batches
        self._logged_domain_warnings: Set[Tuple[str, str]] = set()

    def register_chunks(
        self, chunks: List[Tuple[str, ChunkMetadata, Optional[List[int]]]]
    ) -> None:
        """
        Register chunks for processing.

        Args:
            chunks: List of (chunk_id, metadata, token_ids)
        """
        for chunk_id, metadata, token_ids in chunks:
            # Compute exact hash (if text available)
            if hasattr(metadata, "chunk_text"):
                self.exact_dedup.compute_hash(chunk_id, metadata.chunk_text)

            # Compute near-dedup signature if enabled
            if self.config.dedup.enable_near_dedup and hasattr(metadata, "chunk_text"):
                self.near_dedup.compute_signature(chunk_id, metadata.chunk_text)

            # Prefer token_ids param, otherwise look for metadata.token_ids
            tokens = (
                token_ids
                if token_ids is not None
                else getattr(metadata, "token_ids", None)
            )
            if tokens:
                self.token_analyzer.add_tokens(tokens)

    def _create_buckets(
        self,
        all_chunks: Dict[str, ChunkMetadata],
        stage_name: str,
        target_tokens_override: Optional[int] = None,
    ) -> None:
        """Create stratified buckets for a stage based on curriculum definitions"""
        self.buckets = {}

        # Group chunks by (band, domain)
        for chunk_id, metadata in all_chunks.items():
            if chunk_id in self.removed_chunks:
                continue

            key = (metadata.band, metadata.domain)
            if key not in self.buckets:
                self.buckets[key] = ChunkBucket(
                    band=metadata.band, domain=metadata.domain
                )

            self.buckets[key].chunks.append(chunk_id)

        # Compute target tokens for each bucket based on curriculum
        stage_config = self.curriculum.get_stage_config(stage_name)
        if not stage_config:
            self.logger.warning(f"Stage {stage_name} not found in curriculum")
            return

        band_ratios = stage_config.band_ratios
        target_tokens = (
            int(target_tokens_override)
            if target_tokens_override is not None
            else self.config.stages[stage_name].target_tokens
        )

        # Pre-compute allowed domains with chunks for each band (avoid O(n²) iteration)
        band_to_allowed_domains_with_chunks = defaultdict(set)
        for (band, domain), bucket in self.buckets.items():
            if bucket.chunks:  # Only if band-domain has chunks
                allowed_domains = self.curriculum.get_allowed_domains_for_band(band)
                if domain in allowed_domains:
                    band_to_allowed_domains_with_chunks[band].add(domain)

        # For each band: allocate target_tokens equally across its allowed domains
        # This ensures curriculum distribution, not data distribution
        for (band, domain), bucket in self.buckets.items():
            # Get band ratio from curriculum
            band_ratio = getattr(band_ratios, band.value, 0.0)
            band_target = band_ratio * target_tokens

            # Get allowed domains for this band from curriculum
            allowed_domains = self.curriculum.get_allowed_domains_for_band(band)

            # Filter domains in this bucket to only allowed ones
            if domain not in allowed_domains:
                # Log warning only once per (band, domain) pair
                warning_key = (band.value, domain)
                if warning_key not in self._logged_domain_warnings:
                    self.logger.warning(
                        f"Domain {domain} not allowed for band {band.value}"
                    )
                    self._logged_domain_warnings.add(warning_key)
                bucket.target_tokens = 0
                continue

            # Get pre-computed allowed domains with chunks for this band
            allowed_domains_with_chunks = band_to_allowed_domains_with_chunks.get(
                band, set()
            )

            if not allowed_domains_with_chunks:
                bucket.target_tokens = 0
                continue

            # Distribute band target EQUALLY across allowed domains that have chunks
            # This enforces curriculum targets, not data skew
            num_domains = len(allowed_domains_with_chunks)
            bucket.target_tokens = int(band_target / num_domains)

            self.logger.debug(
                f"Bucket ({band.value}, {domain}): "
                f"band_ratio={band_ratio:.2%}, "
                f"band_target={band_target:,}, "
                f"allowed_domains_with_chunks={num_domains}, "
                f"bucket_target={bucket.target_tokens:,}"
            )

    def _score_chunks_in_bucket(
        self, bucket: ChunkBucket, all_chunks: Dict[str, ChunkMetadata]
    ) -> None:
        """Score all chunks in a bucket"""
        for chunk_id in bucket.chunks:
            metadata = all_chunks[chunk_id]
            # Get token IDs if available, else fall back to small placeholder
            token_ids = getattr(metadata, "token_ids", None)
            if token_ids is None:
                token_ids = list(range(min(100, metadata.token_count or 0)))

            # Composite score
            score = self.diversity_scorer.score_chunk_composite(
                token_ids=token_ids,
                domain=metadata.domain,
                language=metadata.language,
                rarity_weight=0.4,
                coverage_weight=0.6,
            )

            bucket.scores[chunk_id] = score

    def _stratified_sample_from_bucket(
        self, bucket: ChunkBucket, all_chunks: Dict[str, ChunkMetadata]
    ) -> List[str]:
        """Stratified sample from a bucket to meet target tokens"""
        if not bucket.chunks or bucket.target_tokens <= 0:
            return []

        # Sort by band_score when present (descending). Fall back to composite score.
        # Tie-break deterministically by chunk_id.
        def _sort_key(cid: str):
            meta = all_chunks.get(cid)
            band_score = None
            if meta is not None:
                band_score = getattr(meta, "band_score", None)
            has_band_score = 1 if band_score is not None else 0
            try:
                bs = float(band_score) if band_score is not None else 0.0
            except Exception:
                bs = 0.0
            score = float(bucket.scores.get(cid, 0.0) or 0.0)
            return (has_band_score, bs, score, str(cid))

        sorted_chunks = sorted(bucket.chunks, key=_sort_key, reverse=True)

        # Greedily select until target or all chunks exhausted
        selected = []
        current_tokens = 0

        for chunk_id in sorted_chunks:
            if current_tokens >= bucket.target_tokens:
                break
            # Only add if chunk exists in all_chunks
            if chunk_id in all_chunks:
                selected.append(chunk_id)
                current_tokens += all_chunks[chunk_id].token_count

        return selected

    def select_for_stage(
        self,
        all_chunks: Dict[str, ChunkMetadata],
        stage_name: str,
        protected_slices: Optional[List[ProtectedSliceRule]] = None,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Select coreset for a specific training stage.

        Returns:
            (selected_chunk_ids, selection_stats)
        """
        self.logger.info(f"Starting selection for stage: {stage_name}")

        # Validate curriculum
        curriculum_valid, errors = self.curriculum.validate_deterministic_guarantees()
        if not curriculum_valid:
            self.logger.error(f"Curriculum validation failed: {errors}")
            raise ValueError("Curriculum validation failed")

        # Handle deduplication
        # TBD:Disable dedup as it would be handled before the coreset selection process
        # if self.config.dedup.enable_exact_dedup or self.config.dedup.enable_near_dedup:
        # self._apply_deduplication(all_chunks)

        # Create stratified buckets
        self._create_buckets(all_chunks, stage_name)

        # Score chunks
        for bucket in self.buckets.values():
            self._score_chunks_in_bucket(bucket, all_chunks)

        # Select from buckets
        selected = set()
        for bucket in self.buckets.values():
            bucket_selection = self._stratified_sample_from_bucket(bucket, all_chunks)
            selected.update(bucket_selection)

        # Enforce language policy from curriculum
        selected = self._enforce_language_policy(selected, all_chunks, stage_name)

        # Enforce rolling-window smoothness constraints (anti-spike)
        selected = self._enforce_rolling_window(selected, all_chunks, stage_name)

        # Enforce protected slices
        if protected_slices and self.config.selection.include_protected_slices:
            selected = self._enforce_protected_slices(
                selected, all_chunks, protected_slices, stage_name
            )

            # Re-enforce language policy after protected slices may have added chunks
            # that violate language constraints
            selected = self._enforce_language_policy(selected, all_chunks, stage_name)

        self.selected_chunks.update(selected)

        # Compute stats
        stats = self._compute_selection_stats(selected, all_chunks, stage_name)

        self.logger.info(
            f"Selection complete. Selected {len(selected)} chunks, "
            f"{stats['selected_tokens']} tokens (compression {stats['compression_ratio']:.2f}x)"
        )

        return selected, stats

    def _apply_deduplication(self, all_chunks: Dict[str, ChunkMetadata]) -> None:
        """Apply deduplication and mark duplicates for removal"""
        self.logger.info("Applying deduplication...")

        # Find exact duplicates
        if self.config.dedup.enable_exact_dedup:
            exact_dups = self.exact_dedup.find_exact_duplicates()
            self.logger.info(f"Found {len(exact_dups)} exact duplicate pairs")

            # Mark lower-scoring duplicates for removal
            for chunk_id_1, chunk_id_2 in exact_dups:
                # Keep the first one (arbitrarily)
                self.removed_chunks.add(chunk_id_2)

        # Find near-duplicates (more expensive, optional)
        if self.config.dedup.enable_near_dedup and self.near_dedup:
            near_dups = self.near_dedup.find_near_duplicates()
            self.logger.info(
                f"Found {len(near_dups)} near-duplicate pairs (threshold={self.config.dedup.near_dedup_threshold})"
            )

            for chunk_id_1, chunk_id_2, similarity in near_dups:
                if (
                    chunk_id_1 not in self.removed_chunks
                    and chunk_id_2 not in self.removed_chunks
                ):
                    # Remove second one arbitrarily
                    self.removed_chunks.add(chunk_id_2)

    def _enforce_language_policy(
        self, selected: Set[str], all_chunks: Dict[str, ChunkMetadata], stage_name: str
    ) -> Set[str]:
        """Enforce language policy from curriculum.

        Ensures that:
        1. No excluded languages are present
        2. Only allowed languages are present (primary + stage-gated secondary)
        3. Primary languages don't exceed max_share
        4. Secondary languages are included to reach their target share (which equals max_share)
        """
        if not self.curriculum.language_policy:
            return selected

        lang_policy = self.curriculum.language_policy
        selected_filtered = set(selected)

        # Build set of allowed languages for this stage (stage-gated secondary languages).
        allowed_languages = self.curriculum.get_allowed_languages_for_stage(stage_name)

        secondary_langs_allowed = {}
        for lang_code, max_share in lang_policy.secondary_languages.items():
            if lang_code in allowed_languages:
                secondary_langs_allowed[lang_code] = max_share

        # Step 1: Remove ALL languages not in allowed_languages
        to_remove_disallowed = set()
        for cid in selected_filtered:
            if cid in all_chunks and all_chunks[cid].language not in allowed_languages:
                to_remove_disallowed.add(cid)

        if to_remove_disallowed:
            removed_count = len(to_remove_disallowed)
            disallowed_langs = set()
            for cid in to_remove_disallowed:
                if cid in all_chunks:
                    disallowed_langs.add(all_chunks[cid].language)
            selected_filtered -= to_remove_disallowed
            self.logger.info(
                f"Removed {removed_count} chunks in disallowed languages "
                f"({', '.join(sorted(disallowed_langs))}) in stage {stage_name}"
            )

        # Count tokens per language
        def count_tokens():
            lang_counts = defaultdict(int)
            total = 0
            for chunk_id in selected_filtered:
                if chunk_id in all_chunks:
                    lang = all_chunks[chunk_id].language
                    tokens = all_chunks[chunk_id].token_count
                    lang_counts[lang] += tokens
                    total += tokens
            return lang_counts, total

        lang_tokens, total_tokens = count_tokens()

        # Step 2: Enforce max token shares for primary languages
        # We'll remove excess primary language chunks to make room for secondary languages
        if total_tokens > 0:
            for lang_code, max_share in lang_policy.primary_languages.items():
                lang_count = lang_tokens.get(lang_code, 0)
                current_share = lang_count / total_tokens if total_tokens > 0 else 0

                if current_share > max_share:
                    # Remove excess chunks of this language (keep highest scored ones)
                    lang_chunks = []
                    for cid in selected_filtered:
                        if cid in all_chunks and all_chunks[cid].language == lang_code:
                            # Get score from bucket if available
                            band = all_chunks[cid].band
                            domain = all_chunks[cid].domain
                            bucket_key = (band, domain)
                            score = 0
                            if bucket_key in self.buckets:
                                score = self.buckets[bucket_key].scores.get(cid, 0.0)
                            lang_chunks.append(
                                (cid, all_chunks[cid].token_count, score)
                            )

                    # Sort deterministically by score desc, then chunk_id asc.
                    # This prevents cross-process nondeterminism from set iteration order
                    # when many chunks share identical scores.
                    lang_chunks = sorted(
                        lang_chunks, key=lambda x: (-float(x[2] or 0.0), str(x[0]))
                    )

                    # Keep chunks within max_share limit
                    allowed_tokens = int(max_share * total_tokens)
                    kept_tokens = 0
                    chunks_to_keep = set()

                    for cid, tokens, _ in lang_chunks:
                        if kept_tokens + tokens <= allowed_tokens:
                            chunks_to_keep.add(cid)
                            kept_tokens += tokens

                    # Remove others
                    lang_to_remove = {cid for cid, _, _ in lang_chunks} - chunks_to_keep
                    selected_filtered -= lang_to_remove

                    self.logger.info(
                        f"Language {lang_code} exceeds max_share {max_share:.2%}. "
                        f"Current: {current_share:.2%}, Removed {len(lang_to_remove)} chunks"
                    )

        # Recount tokens
        lang_tokens, total_tokens = count_tokens()

        # Step 3: Add secondary language chunks to reach their target share
        # BUT ONLY if the available data pool doesn't already exceed the target
        if total_tokens > 0:
            for lang_code, target_share in secondary_langs_allowed.items():
                current_count = lang_tokens.get(lang_code, 0)
                current_share = current_count / total_tokens if total_tokens > 0 else 0
                target_tokens = int(target_share * total_tokens)

                # Only add if below target
                if current_count < target_tokens and current_share < target_share:
                    # Find available chunks of this language
                    available_chunks = []
                    for chunk_id, metadata in all_chunks.items():
                        if (
                            metadata.language == lang_code
                            and chunk_id not in selected_filtered
                        ):
                            # Keep curriculum band/domain constraints intact
                            allowed_domains = (
                                self.curriculum.get_allowed_domains_for_band(
                                    metadata.band
                                )
                            )
                            if metadata.domain not in allowed_domains:
                                continue
                            # Get score
                            band = metadata.band
                            domain = metadata.domain
                            bucket_key = (band, domain)
                            score = 0
                            if bucket_key in self.buckets:
                                score = self.buckets[bucket_key].scores.get(
                                    chunk_id, 0.0
                                )
                            available_chunks.append(
                                (chunk_id, metadata.token_count, score)
                            )

                    if not available_chunks:
                        # No available chunks for this language
                        continue

                    # Sort deterministically by score desc, then chunk_id asc.
                    available_chunks = sorted(
                        available_chunks, key=lambda x: (-float(x[2] or 0.0), str(x[0]))
                    )

                    old_count = current_count
                    for chunk_id, tokens, _ in available_chunks:
                        if current_count + tokens <= target_tokens:
                            selected_filtered.add(chunk_id)
                            current_count += tokens
                        else:
                            break

                    if current_count > old_count:
                        lang_tokens[lang_code] = current_count
                        self.logger.info(
                            f"Language {lang_code}: Added chunks to reach target. "
                            f"Tokens: {old_count} -> {current_count} (target: {target_tokens})"
                        )

        # Recount tokens after Step 3
        lang_tokens, total_tokens = count_tokens()

        # Step 4: Enforce maximum token shares for secondary languages
        # Ensure secondary languages don't exceed their max_share
        if total_tokens > 0:
            for lang_code, max_share in secondary_langs_allowed.items():
                lang_count = lang_tokens.get(lang_code, 0)
                current_share = lang_count / total_tokens if total_tokens > 0 else 0

                if current_share > max_share:
                    # Remove excess chunks of this language (keep highest scored ones)
                    lang_chunks = []
                    for cid in selected_filtered:
                        if cid in all_chunks and all_chunks[cid].language == lang_code:
                            # Get score from bucket if available
                            band = all_chunks[cid].band
                            domain = all_chunks[cid].domain
                            bucket_key = (band, domain)
                            score = 0
                            if bucket_key in self.buckets:
                                score = self.buckets[bucket_key].scores.get(cid, 0.0)
                            lang_chunks.append(
                                (cid, all_chunks[cid].token_count, score)
                            )

                    # Sort deterministically by score desc, then chunk_id asc.
                    lang_chunks = sorted(
                        lang_chunks, key=lambda x: (-float(x[2] or 0.0), str(x[0]))
                    )

                    # Keep chunks within max_share limit
                    allowed_tokens = int(max_share * total_tokens)
                    kept_tokens = 0
                    chunks_to_keep = set()

                    for cid, tokens, _ in lang_chunks:
                        if kept_tokens + tokens <= allowed_tokens:
                            chunks_to_keep.add(cid)
                            kept_tokens += tokens

                    # Remove others
                    lang_to_remove = {cid for cid, _, _ in lang_chunks} - chunks_to_keep
                    selected_filtered -= lang_to_remove

                    self.logger.info(
                        f"Language {lang_code} exceeds max_share {max_share:.2%}. "
                        f"Current: {current_share:.2%}, Removed {len(lang_to_remove)} chunks"
                    )

        return selected_filtered

        return selected_filtered

    def _build_language_distribution(
        self, selected: Set[str], all_chunks: Dict[str, ChunkMetadata]
    ) -> LanguageDistribution:
        """Build language distribution from selected chunks"""
        lang_counts = defaultdict(int)
        total_tokens = 0

        for chunk_id in selected:
            if chunk_id in all_chunks:
                lang = all_chunks[chunk_id].language
                tokens = all_chunks[chunk_id].token_count
                lang_counts[lang] += tokens
                total_tokens += tokens

        # Convert to shares
        lang_dist = {}
        if total_tokens > 0:
            for lang, count in lang_counts.items():
                lang_dist[lang] = count / total_tokens

        return LanguageDistribution(languages=lang_dist)

    def _enforce_protected_slices(
        self,
        selected: Set[str],
        all_chunks: Dict[str, ChunkMetadata],
        protected_slices: List[ProtectedSliceRule],
        stage_name: str,
        *,
        target_tokens_override: Optional[int] = None,
    ) -> Set[str]:
        """
        Restore critical slices that may have been pruned by enforcement steps,
        but ONLY UP TO curriculum targets. This ensures quality while respecting distribution.

        For band-based rules (e.g., B4):
        - Add chunks if band is below its curriculum target
        - Don't exceed the target even if preservation_ratio would require it

        For domain-based rules (e.g., "code"):
        - Only operate within allowed bands
        - Add chunks if domain is below its curriculum-implied target

        IMPORTANT: Respect language policy - don't add chunks in disallowed languages.
        """
        self.logger.info(
            "Enforcing protected slice constraints (curriculum-bounded)..."
        )

        # Get stage config
        stage_config = self.curriculum.get_stage_config(stage_name)
        if not stage_config:
            return selected

        # Collect band targets and allowed bands
        band_targets = {}
        allowed_bands_in_stage = set()
        for band_enum in self.curriculum.bands.keys():
            band_name = band_enum.value
            band_ratio = getattr(stage_config.band_ratios, band_name, 0.0)
            if band_ratio > 0:
                allowed_bands_in_stage.add(band_enum)
                band_targets[band_name] = band_ratio

        # Build set of allowed languages for this stage (stage-gated secondary languages)
        allowed_languages = self.curriculum.get_allowed_languages_for_stage(stage_name)

        # Disallowed = everything NOT allowed
        disallowed_languages = set()
        # Add all chunks in all_chunks to find what languages exist
        all_languages_in_pool = set()
        for chunk in all_chunks.values():
            all_languages_in_pool.add(chunk.language)

        # Any language not in allowed_languages is disallowed
        for lang in all_languages_in_pool:
            if lang not in allowed_languages:
                disallowed_languages.add(lang)

        # Compute current distribution
        band_tokens = defaultdict(int)
        domain_tokens = defaultdict(int)
        total_tokens = 0
        for cid in selected:
            if cid in all_chunks:
                meta = all_chunks[cid]
                band_tokens[meta.band.value] += meta.token_count
                domain_tokens[meta.domain] += meta.token_count
                total_tokens += meta.token_count

        if total_tokens == 0:
            return selected

        target_tokens = (
            int(target_tokens_override)
            if target_tokens_override is not None
            else int(self.config.stages[stage_name].target_tokens)
        )

        total_added = 0

        band_names = set(difficulty_band_order())

        # For band-based rules: add chunks up to curriculum target
        for rule in protected_slices:
            if rule.band_or_domain in band_names:
                band_name = rule.band_or_domain
                target_share = band_targets.get(band_name, 0.0)
                target_tokens_for_band = int(target_share * target_tokens)
                current_tokens_in_band = band_tokens[band_name]

                if current_tokens_in_band < target_tokens_for_band:
                    # Band is below target, add chunks
                    needed_tokens = target_tokens_for_band - current_tokens_in_band

                    # Find band enum for this band name
                    band_enum = None
                    for b_enum in self.curriculum.bands.keys():
                        if b_enum.value == band_name:
                            band_enum = b_enum
                            break

                    if band_enum is None:
                        continue

                    # Get candidates: band must match, domain must be allowed, language must be allowed
                    # (no explicitly_excluded, no disallowed secondary languages)
                    band_def = self.curriculum.bands.get(band_enum)
                    raw = getattr(self.curriculum, "raw_curriculum", {}) or {}
                    explicitly_excluded = set(
                        (
                            raw.get("language_and_context", {})
                            .get("language_policy", {})
                            .get("excluded_languages", [])
                        )
                        or raw.get("languages", {}).get("explicitly_excluded", [])
                        or []
                    )

                    candidates = [
                        cid
                        for cid in all_chunks.keys()
                        if cid not in selected
                        and all_chunks[cid].band == band_enum
                        and all_chunks[cid].domain
                        in (band_def.allowed_domains if band_def else [])
                        and all_chunks[cid].language not in disallowed_languages
                        and all_chunks[cid].language not in explicitly_excluded
                    ]

                    # Sort by score descending
                    candidates_sorted = sorted(
                        candidates,
                        key=lambda x: self.buckets.get(
                            (all_chunks[x].band, all_chunks[x].domain),
                            ChunkBucket(all_chunks[x].band, all_chunks[x].domain),
                        ).scores.get(x, 0.0),
                        reverse=True,
                    )

                    # Add top candidates up to needed_tokens
                    added_tokens = 0
                    for cid in candidates_sorted:
                        if added_tokens >= needed_tokens:
                            break
                        selected.add(cid)
                        added_tokens += all_chunks[cid].token_count
                        total_added += 1

                    if added_tokens > 0:
                        self.logger.debug(
                            f"Band {band_name}: added {total_added} chunks ({added_tokens} tokens) "
                            f"to reach target {target_tokens_for_band}"
                        )

        # For domain-based rules: add chunks within allowed bands, up to curriculum target
        for rule in protected_slices:
            if rule.band_or_domain not in band_names:
                domain_name = rule.band_or_domain
                current_tokens_in_domain = domain_tokens[domain_name]

                # Compute implied domain target (distribute band targets across allowed domains)
                target_share = 0.0
                for band_enum, band_def in self.curriculum.bands.items():
                    band_name = band_enum.value
                    if domain_name in band_def.allowed_domains:
                        band_ratio = band_targets.get(band_name, 0.0)
                        allowed_count = len(band_def.allowed_domains or [])
                        target_share += band_ratio / max(1, allowed_count)

                target_tokens_for_domain = int(target_share * target_tokens)

                if current_tokens_in_domain < target_tokens_for_domain:
                    needed_tokens = target_tokens_for_domain - current_tokens_in_domain

                    # Get available chunks from this domain, within allowed bands, excluding disallowed languages
                    # Also exclude explicitly_excluded languages
                    raw = getattr(self.curriculum, "raw_curriculum", {}) or {}
                    explicitly_excluded = set(
                        (
                            raw.get("language_and_context", {})
                            .get("language_policy", {})
                            .get("excluded_languages", [])
                        )
                        or raw.get("languages", {}).get("explicitly_excluded", [])
                        or []
                    )
                    candidates = [
                        cid
                        for cid in all_chunks.keys()
                        if cid not in selected
                        and all_chunks[cid].domain == domain_name
                        and all_chunks[cid].band in allowed_bands_in_stage
                        and all_chunks[cid].language not in disallowed_languages
                        and all_chunks[cid].language not in explicitly_excluded
                    ]

                    # Sort by score descending
                    candidates_sorted = sorted(
                        candidates,
                        key=lambda x: self.buckets.get(
                            (all_chunks[x].band, all_chunks[x].domain),
                            ChunkBucket(all_chunks[x].band, all_chunks[x].domain),
                        ).scores.get(x, 0.0),
                        reverse=True,
                    )

                    # Add top candidates up to needed_tokens
                    added_tokens = 0
                    domain_added = 0
                    for cid in candidates_sorted:
                        if added_tokens >= needed_tokens:
                            break
                        selected.add(cid)
                        added_tokens += all_chunks[cid].token_count
                        domain_added += 1
                        total_added += 1

                    if added_tokens > 0:
                        self.logger.debug(
                            f"Domain {domain_name}: added {domain_added} chunks ({added_tokens} tokens) "
                            f"to reach target {target_tokens_for_domain}"
                        )

        if total_added > 0:
            self.logger.info(
                f"Protected slice enforcement: added {total_added} chunks to restore curriculum targets"
            )
        else:
            self.logger.info(
                "Protected slice coverage already meets curriculum targets"
            )

        return selected

        self.logger.info(f"Protected slice enforcement: added {total_added} chunks")

        return selected

    def _enforce_rolling_window(
        self, selected: Set[str], all_chunks: Dict[str, ChunkMetadata], stage_name: str
    ) -> Set[str]:
        """Enforce rolling-window smoothness constraints from curriculum.

        This prevents spikes by ensuring band and domain shares do not
        exceed curriculum.rolling_window deltas relative to stage targets.

        Key fix: Removal must maintain band distribution proportions. When a
        band exceeds delta limits, we remove chunks to reach the limit while
        preserving the band's intended target share relative to other bands.
        """
        # Configs / guards
        if not getattr(self.config.curriculum, "enforce_rolling_window", False):
            return selected
        if not self.curriculum.rolling_window:
            return selected

        rw = self.curriculum.rolling_window
        # Get stage band targets
        stage_spec = self.curriculum.stages.get(stage_name)
        if not stage_spec:
            return selected

        # Compute total tokens in selected
        total_tokens = 0
        band_tokens = defaultdict(int)
        domain_tokens = defaultdict(int)
        for cid in selected:
            meta = all_chunks.get(cid)
            if not meta:
                continue
            t = meta.token_count
            total_tokens += t
            band_tokens[meta.band.value] += t
            domain_tokens[meta.domain] += t

        if total_tokens == 0:
            return selected

        # Band targets from curriculum
        band_targets = {}
        for b in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            band_targets[b] = getattr(stage_spec.band_ratios, b, 0.0)

        # Domain inferred targets: distribute each band's ratio equally among its allowed domains
        inferred_domain_target = defaultdict(float)
        for band_enum, band_def in self.curriculum.bands.items():
            band_name = band_enum.value
            band_ratio = band_targets.get(band_name, 0.0)
            allowed = band_def.allowed_domains or []
            if not allowed:
                continue
            per_domain = band_ratio / max(1, len(allowed))
            for d in allowed:
                inferred_domain_target[d] += per_domain

        # Normalize inferred_domain_target to sum to 1.0 if it has any mass
        total_inferred = sum(inferred_domain_target.values())
        if total_inferred > 0:
            for d in list(inferred_domain_target.keys()):
                inferred_domain_target[d] = inferred_domain_target[d] / total_inferred

        # Enforce band deltas - remove excess from bands that exceed (target + delta)
        # Key: when removing, target the lowest-scored chunks first to preserve quality
        to_remove = set()
        for band, tok in band_tokens.items():
            current_share = tok / total_tokens
            target_share = band_targets.get(band, 0.0)
            allowed = target_share + rw.max_band_delta
            if current_share > allowed:
                # This band's share exceeds the allowed limit
                # Remove lowest-scored chunks until current_share <= allowed
                excess_tokens = int((current_share - allowed) * total_tokens)

                band_chunk_ids = [
                    cid
                    for cid in selected
                    if all_chunks.get(cid)
                    and all_chunks[cid].band.value == band
                    and cid not in to_remove
                ]
                # Sort by score ascending (lowest-scoring chunks first)
                band_chunk_ids_sorted = sorted(
                    band_chunk_ids,
                    key=lambda x: self.buckets.get(
                        (all_chunks[x].band, all_chunks[x].domain),
                        ChunkBucket(all_chunks[x].band, all_chunks[x].domain),
                    ).scores.get(x, 0.0),
                )

                removed = 0
                for cid in band_chunk_ids_sorted:
                    to_remove.add(cid)
                    removed += all_chunks[cid].token_count
                    if removed >= excess_tokens:
                        break

        # Enforce domain deltas - same approach but per domain
        for domain, tok in domain_tokens.items():
            current_share = tok / total_tokens
            target_share = inferred_domain_target.get(domain, 0.0)
            allowed = target_share + rw.max_domain_delta
            if current_share > allowed:
                excess_tokens = int((current_share - allowed) * total_tokens)

                domain_chunk_ids = [
                    cid
                    for cid in selected
                    if all_chunks.get(cid)
                    and all_chunks[cid].domain == domain
                    and cid not in to_remove
                ]
                domain_chunk_ids_sorted = sorted(
                    domain_chunk_ids,
                    key=lambda x: self.buckets.get(
                        (all_chunks[x].band, all_chunks[x].domain),
                        ChunkBucket(all_chunks[x].band, all_chunks[x].domain),
                    ).scores.get(x, 0.0),
                )

                removed = 0
                for cid in domain_chunk_ids_sorted:
                    to_remove.add(cid)
                    removed += all_chunks[cid].token_count
                    if removed >= excess_tokens:
                        break

        if to_remove:
            self.logger.info(
                f"Rolling window enforcement removing {len(to_remove)} chunks to respect band/domain deltas"
            )
            selected = set(selected) - to_remove

        return selected

    def _compute_selection_stats(
        self, selected: Set[str], all_chunks: Dict[str, ChunkMetadata], stage_name: str
    ) -> Dict[str, Any]:
        """Compute selection statistics including deduplication metrics"""

        # Count tokens and compute distributions
        selected_chunks = {
            cid: all_chunks[cid] for cid in selected if cid in all_chunks
        }

        band_counts = defaultdict(int)
        domain_counts = defaultdict(int)
        language_counts = defaultdict(int)
        total_tokens = 0

        for chunk_id, metadata in selected_chunks.items():
            band_counts[metadata.band.value] += metadata.token_count
            domain_counts[metadata.domain] += metadata.token_count
            language_counts[metadata.language] += metadata.token_count
            total_tokens += metadata.token_count

        total_input_tokens = sum(m.token_count for m in all_chunks.values())

        # Build distributions
        band_dist = BandDistribution(
            B0=band_counts.get("B0", 0) / total_tokens if total_tokens > 0 else 0.0,
            B1=band_counts.get("B1", 0) / total_tokens if total_tokens > 0 else 0.0,
            B2=band_counts.get("B2", 0) / total_tokens if total_tokens > 0 else 0.0,
            B3=band_counts.get("B3", 0) / total_tokens if total_tokens > 0 else 0.0,
            B4=band_counts.get("B4", 0) / total_tokens if total_tokens > 0 else 0.0,
            B5=band_counts.get("B5", 0) / total_tokens if total_tokens > 0 else 0.0,
            B6=band_counts.get("B6", 0) / total_tokens if total_tokens > 0 else 0.0,
        )

        from ..core.types import DomainDistributionV2

        domain_total = (
            {d: (c / total_tokens) for d, c in domain_counts.items()}
            if total_tokens > 0
            else {}
        )

        # by_band shares (within each band)
        band_domain_tokens = defaultdict(lambda: defaultdict(int))
        band_total_tokens = defaultdict(int)
        for _cid, meta in selected_chunks.items():
            b = meta.band.value
            d = str(getattr(meta.domain, "value", meta.domain))
            band_domain_tokens[b][d] += int(meta.token_count)
            band_total_tokens[b] += int(meta.token_count)

        by_band = {}
        for b, dom_counts in band_domain_tokens.items():
            denom = float(band_total_tokens.get(b, 0) or 0)
            if denom <= 0:
                continue
            by_band[b] = {d: (float(c) / denom) for d, c in dom_counts.items()}

        domain_dist = DomainDistributionV2(total=domain_total, by_band=by_band)

        language_dist = LanguageDistribution(
            languages={
                lang: count / total_tokens for lang, count in language_counts.items()
            }
        )

        # Count deduplication metrics from removed_chunks
        dedup_removed = len(self.removed_chunks)

        return {
            "total_input_chunks": len(all_chunks),
            "total_input_tokens": total_input_tokens,
            "selected_chunks": len(selected),
            "selected_tokens": total_tokens,
            "compression_ratio": (
                total_input_tokens / total_tokens if total_tokens > 0 else float("inf")
            ),
            "band_distribution": band_dist,
            "domain_distribution": domain_dist,
            "language_distribution": language_dist,
            "exact_dedup_removed": dedup_removed,
            "near_dedup_removed": 0,  # Tracked together in removed_chunks
        }
