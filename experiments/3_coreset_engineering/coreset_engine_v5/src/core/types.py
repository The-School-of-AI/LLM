"""
Core data structures and type definitions for the coreset selection engine.
Provides type-safe interfaces for all pipeline components.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ProcessingStatus(str, Enum):
    """Processing status for datasets"""

    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class DifficultyBand(str, Enum):
    """Difficulty bands (B0-B5)"""

    B0 = "B0"  # Nursery
    B1 = "B1"  # Primary
    B2 = "B2"  # HighSchool
    B3 = "B3"  # Undergraduate
    B4 = "B4"  # Graduate
    B5 = "B5"  # PhD
    B6 = "B6"  # Post-PhD / Frontier


def difficulty_band_order() -> List[str]:
    """Canonical ordered list of band names (e.g., ["B0", "B1", ...])."""
    return [b.value for b in DifficultyBand]


class StageName(str, Enum):
    """Training stages"""

    PRETRAIN_1B = "1B"
    PRETRAIN_3B = "3B"
    PRETRAIN_8B = "8B"
    PRETRAIN_70B = "70B"
    SFT = "SFT"
    ALIGNMENT = "ALIGNMENT"


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk"""

    chunk_id: str
    dataset_id: str
    token_count: int
    byte_length: int
    domain: str  # code, math, reasoning, agentic, indic, clean_web
    language: str  # ISO639-1 code
    band: DifficultyBand
    source_doc_id: str
    source_url: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)
    sensitive_markers: List[str] = field(default_factory=list)
    start_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "dataset_id": self.dataset_id,
            "token_count_estimate": self.token_count,
            "byte_length": self.byte_length,
            "domain": self.domain,
            "language": self.language,
            "band": self.band.value,
            "source_doc_id": self.source_doc_id,
            "source_url": self.source_url,
            "quality_flags": self.quality_flags,
            "sensitive_markers": self.sensitive_markers,
            "start_offset": self.start_offset,
        }


@dataclass
class ProtectedSliceRule:
    """Rule for protecting certain slices from aggressive downsampling"""

    band_or_domain: str
    minimum_preservation_ratio: float  # e.g., 0.95
    reason: str


@dataclass
class BandDistribution:
    """Distribution across difficulty bands"""

    B0: float = 0.0
    B1: float = 0.0
    B2: float = 0.0
    B3: float = 0.0
    B4: float = 0.0
    B5: float = 0.0
    B6: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "B0": self.B0,
            "B1": self.B1,
            "B2": self.B2,
            "B3": self.B3,
            "B4": self.B4,
            "B5": self.B5,
            "B6": self.B6,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BandDistribution":
        return cls(**d)

    def validate(self) -> bool:
        """Verify distribution sums to ~1.0"""
        total = sum([self.B0, self.B1, self.B2, self.B3, self.B4, self.B5, self.B6])
        return 0.99 <= total <= 1.01


@dataclass
class DomainDistribution:
    """Distribution across domains"""

    code: float = 0.0
    math: float = 0.0
    reasoning: float = 0.0
    agentic: float = 0.0
    indic: float = 0.0
    clean_web: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "code": self.code,
            "math": self.math,
            "reasoning": self.reasoning,
            "agentic": self.agentic,
            "indic": self.indic,
            "clean_web": self.clean_web,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "DomainDistribution":
        return cls(**d)

    def validate(self) -> bool:
        """Verify distribution sums to ~1.0"""
        total = sum(
            [
                self.code,
                self.math,
                self.reasoning,
                self.agentic,
                self.indic,
                self.clean_web,
            ]
        )
        return 0.99 <= total <= 1.01


@dataclass
class DomainDistributionV2:
    """Flexible domain distribution supporting arbitrary domain IDs.

    - total: token-share by domain across the whole coreset
    - by_band: token-share by domain within each band (shares sum to ~1 per band)
    """

    total: Dict[str, float] = field(default_factory=dict)
    by_band: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": dict(self.total),
            "by_band": {b: dict(d) for b, d in (self.by_band or {}).items()},
        }

    def validate(self) -> bool:
        # total sums to ~1.0 when present
        if self.total:
            s = float(sum(self.total.values()))
            if not (0.99 <= s <= 1.01):
                return False
        # by_band sums to ~1.0 per band when present
        for _band, dist in (self.by_band or {}).items():
            if not dist:
                continue
            s = float(sum(dist.values()))
            if not (0.99 <= s <= 1.01):
                return False
        return True


@dataclass
class LanguageDistribution:
    """Distribution across languages"""

    languages: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return self.languages

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "LanguageDistribution":
        return cls(languages=d)

    def validate(self) -> bool:
        """Verify distribution sums to ~1.0"""
        total = sum(self.languages.values())
        return 0.99 <= total <= 1.01


@dataclass
class CoresetComposition:
    """Composition of a coreset"""

    band_distribution: BandDistribution
    domain_distribution: Any
    language_distribution: LanguageDistribution

    def to_dict(self) -> Dict[str, Any]:
        def _maybe_to_dict(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()  # type: ignore[no-any-return]
            if isinstance(obj, dict):
                return obj
            return obj

        return {
            "band_distribution": _maybe_to_dict(self.band_distribution),
            "domain_distribution": _maybe_to_dict(self.domain_distribution),
            "language_distribution": _maybe_to_dict(self.language_distribution),
        }


@dataclass
class ProtectedSlicesPreserved:
    """Preservation ratios for protected slices"""

    B4_preservation_ratio: float
    B5_preservation_ratio: float
    code_preservation_ratio: float
    agentic_preservation_ratio: float
    indic_preservation_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "B4_preservation_ratio": self.B4_preservation_ratio,
            "B5_preservation_ratio": self.B5_preservation_ratio,
            "code_preservation_ratio": self.code_preservation_ratio,
            "agentic_preservation_ratio": self.agentic_preservation_ratio,
            "indic_preservation_ratio": self.indic_preservation_ratio,
        }

    def validate_preservation(
        self, rules: List[ProtectedSliceRule]
    ) -> Tuple[bool, List[str]]:
        """Validate that all protected slices meet preservation thresholds"""
        violations = []
        ratios = self.to_dict()

        for rule in rules:
            if rule.band_or_domain in ratios:
                if ratios[rule.band_or_domain] < rule.minimum_preservation_ratio:
                    violations.append(
                        f"{rule.band_or_domain}: {ratios[rule.band_or_domain]:.2%} "
                        f"< {rule.minimum_preservation_ratio:.2%} ({rule.reason})"
                    )

        return len(violations) == 0, violations


@dataclass
class DeduplicationStats:
    """Statistics on deduplication"""

    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    total_chunks_before: int = 0
    total_chunks_after: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0

    @property
    def dedup_ratio(self) -> float:
        """Tokens removed / tokens before"""
        if self.total_tokens_before == 0:
            return 0.0
        return (
            self.total_tokens_before - self.total_tokens_after
        ) / self.total_tokens_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "total_chunks_before": self.total_chunks_before,
            "total_chunks_after": self.total_chunks_after,
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "dedup_ratio": self.dedup_ratio,
        }


@dataclass
class CoverageAudit:
    """Coverage audit results"""

    passed: bool
    expected_coverage: Dict[str, float]  # domain -> expected ratio
    actual_coverage: Dict[str, float]  # domain -> actual ratio
    tolerance: float
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_coverage": self.expected_coverage,
            "actual_coverage": self.actual_coverage,
            "tolerance": self.tolerance,
            "violations": self.violations,
        }


@dataclass
class CoresetManifest:
    """Manifest for a stage-specific coreset"""

    stage_name: StageName
    coreset_id: str
    target_tokens: int
    actual_tokens: int
    created_at: str  # ISO8601 timestamp
    pipeline_version: str
    curriculum_version: str
    seed: int
    config_hash: str

    selected_chunks_count: int
    selected_chunks_file: Optional[str] = None  # reference to external file

    # Optional run/sharding metadata (primarily for streaming mode observability)
    shard_id: Optional[int] = None
    num_shards: Optional[int] = None
    stage_target_scale: Optional[float] = None
    total_input_tokens_estimate_global: Optional[int] = None
    total_input_tokens_estimate_shard: Optional[int] = None

    # Optional disambiguation fields for streaming/sharded runs.
    # - target_tokens_global: pre-scaling, pre-shard-split target for the whole stage
    # - target_tokens_shard: effective per-worker target after scaling and shard split
    target_tokens_global: Optional[int] = None
    target_tokens_shard: Optional[int] = None

    composition: Optional[CoresetComposition] = None
    protected_slices_preserved: Optional[ProtectedSlicesPreserved] = None
    dedup_stats: Optional[DeduplicationStats] = None
    coverage_audit: Optional[CoverageAudit] = None

    # Rolling-window anti-spike enforcement summary (if enabled)
    rolling_window_stats: Optional[Dict[str, float]] = None

    # Availability/feasibility summary for non-overlap + streaming runs.
    # When present, validators can downgrade target/ratio failures that are
    # provably unachievable due to insufficient remaining eligible pool.
    availability_stats: Optional[Dict[str, Any]] = None

    deterministic: bool = True
    algorithm_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        out: Dict[str, Any] = {
            "stage_name": self.stage_name.value,
            "coreset_id": self.coreset_id,
            "target_tokens": self.target_tokens,
            "target_tokens_global": self.target_tokens_global,
            "target_tokens_shard": self.target_tokens_shard,
            "actual_tokens": self.actual_tokens,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "curriculum_version": self.curriculum_version,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "selected_chunks_count": self.selected_chunks_count,
            "selected_chunks_file": self.selected_chunks_file,
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "stage_target_scale": self.stage_target_scale,
            "total_input_tokens_estimate_global": self.total_input_tokens_estimate_global,
            "total_input_tokens_estimate_shard": self.total_input_tokens_estimate_shard,
            "composition": self.composition.to_dict() if self.composition else None,
            "protected_slices_preserved": (
                self.protected_slices_preserved.to_dict()
                if self.protected_slices_preserved
                else None
            ),
            "dedup_stats": self.dedup_stats.to_dict() if self.dedup_stats else None,
            "coverage_audit": (
                self.coverage_audit.to_dict() if self.coverage_audit else None
            ),
            "rolling_window_stats": self.rolling_window_stats,
            "availability_stats": self.availability_stats,
            "deterministic": self.deterministic,
            "algorithm_version": self.algorithm_version,
        }

        # Backward compatibility / cleanliness: omit new keys if unset.
        if out.get("target_tokens_global") is None:
            out.pop("target_tokens_global", None)
        if out.get("target_tokens_shard") is None:
            out.pop("target_tokens_shard", None)

        if out.get("shard_id") is None:
            out.pop("shard_id", None)
        if out.get("num_shards") is None:
            out.pop("num_shards", None)
        if out.get("stage_target_scale") is None:
            out.pop("stage_target_scale", None)
        if out.get("total_input_tokens_estimate_global") is None:
            out.pop("total_input_tokens_estimate_global", None)
        if out.get("total_input_tokens_estimate_shard") is None:
            out.pop("total_input_tokens_estimate_shard", None)

        return out

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class SelectionStatistics:
    """Statistics for a selection run"""

    total_input_chunks: int
    total_input_tokens: int
    selected_chunks: int
    selected_tokens: int
    compression_ratio: float
    stage_name: StageName
    band_distribution: BandDistribution
    domain_distribution: DomainDistribution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_chunks": self.total_input_chunks,
            "total_input_tokens": self.total_input_tokens,
            "selected_chunks": self.selected_chunks,
            "selected_tokens": self.selected_tokens,
            "compression_ratio": self.compression_ratio,
            "stage_name": self.stage_name.value,
            "band_distribution": self.band_distribution.to_dict(),
            "domain_distribution": self.domain_distribution.to_dict(),
        }
