"""
Configuration management for the coreset selection engine.
Provides hierarchical, validated configuration with environment override support.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication strategies"""

    enable_exact_dedup: bool = True
    enable_near_dedup: bool = True
    near_dedup_threshold: float = 0.85  # SimHash / MinHash threshold
    hash_algorithm: str = "xxhash64"  # xxhash64, sha256
    chunk_size_for_hashing: int = 4096
    parallel_workers: int = 32


@dataclass
class DiversityConfig:
    """Configuration for diversity metrics"""

    enable_diversity_weighting: bool = True
    diversity_metric: str = "entropy"  # entropy, coverage_based
    rare_token_boost: float = 1.5  # Weight boost for rare tokens
    tail_token_boost: float = 2.0  # Weight boost for tail tokens
    domain_diversity_weight: float = 0.3
    language_diversity_weight: float = 0.2


@dataclass
class SelectionConfig:
    """Configuration for selection strategies"""

    strategy: str = (
        "stratified_density_aware"  # stratified_density_aware, uniform_random
    )
    include_protected_slices: bool = True
    protected_preservation_override: Dict[str, float] = field(
        default_factory=lambda: {
            "B4": 0.95,
            "B5": 0.95,
            "code": 0.90,
            "agentic": 0.90,
            "indic": 0.85,
        }
    )
    bucket_internal_sampling: str = "stratified"  # stratified, density_weighted
    non_overlap_enforcement: bool = True


@dataclass
class CurriculumConfig:
    """Configuration for curriculum adherence"""

    curriculum_yaml_path: str
    freeze_curriculum: bool = True
    enforce_rolling_window: bool = True
    rolling_window_tolerance: float = 0.03  # 3% max delta
    deterministic_seed: int = 42
    seed_scope: List[str] = field(
        default_factory=lambda: ["sampling", "shuffling", "stage_transition"]
    )


@dataclass
class StageConfig:
    """Configuration for a specific training stage"""

    stage_name: str
    target_tokens: int
    target_chunks: Optional[int] = None
    enable_sft_adaptation: bool = False
    enable_alignment_adaptation: bool = False


@dataclass
class AblationConfig:
    """Configuration for ablation studies"""

    enable_ablation_mode: bool = False
    ablation_variant: str = "baseline"  # baseline, no_dedup, no_diversity, density_only
    track_metrics: List[str] = field(
        default_factory=lambda: [
            "compression_ratio",
            "band_coverage",
            "domain_coverage",
            "protected_preservation",
            "convergence_speed",
        ]
    )
    save_intermediate_results: bool = True


@dataclass
class IOConfig:
    """Configuration for I/O and storage"""

    input_dataset_path: str = "/data/datasets"
    input_metadata_path: str = "/data/metadata"
    output_coreset_path: str = "/output/coresets"
    output_manifest_path: str = "/output/manifests"
    output_index_format: str = "jsonl"  # parquet, jsonl, csv
    use_object_store: bool = False
    object_store_type: Optional[str] = None  # s3, gcs
    object_store_bucket: Optional[str] = None
    object_store_prefix: Optional[str] = None
    num_parallel_loaders: int = 16
    cache_metadata: bool = True
    cache_dir: str = "/tmp/coreset_cache"


@dataclass
class ValidationConfig:
    """Configuration for validation checks"""

    enable_curriculum_validation: bool = True
    enable_coverage_validation: bool = True
    enable_determinism_check: bool = True
    enable_overlap_check: bool = True
    strict_mode: bool = True  # Fail on any violation
    validation_tolerance: float = 0.01  # 1% tolerance for coverage


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility"""

    enable_reproducibility_checks: bool = True
    save_config_hash: bool = True
    save_seed: bool = True
    save_algorithm_version: bool = True
    emit_reproducibility_manifest: bool = True
    version: str = "1.0.0"


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""

    # Core configs
    dedup: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    curriculum: CurriculumConfig = field(
        default_factory=lambda: CurriculumConfig(
            curriculum_yaml_path="config/curriculum.yaml"
        )
    )
    io: IOConfig = field(default_factory=IOConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Stage-specific configs
    stages: Dict[str, StageConfig] = field(
        default_factory=lambda: {
            "1B": StageConfig(stage_name="1B", target_tokens=20_000_000_000),
            "3B": StageConfig(stage_name="3B", target_tokens=40_000_000_000),
            "8B": StageConfig(stage_name="8B", target_tokens=100_000_000_000),
            "70B": StageConfig(stage_name="70B", target_tokens=240_000_000_000),
            "SFT": StageConfig(
                stage_name="SFT",
                target_tokens=10_000_000_000,
                enable_sft_adaptation=True,
            ),
            "ALIGNMENT": StageConfig(
                stage_name="ALIGNMENT",
                target_tokens=5_000_000_000,
                enable_alignment_adaptation=True,
            ),
        }
    )

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "1.0.0"
    config_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dedup": asdict(self.dedup),
            "diversity": asdict(self.diversity),
            "selection": asdict(self.selection),
            "curriculum": asdict(self.curriculum),
            "io": asdict(self.io),
            "validation": asdict(self.validation),
            "reproducibility": asdict(self.reproducibility),
            "ablation": asdict(self.ablation),
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "config_name": self.config_name,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialize to YAML"""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of config for reproducibility"""
        config_json = self.to_json()
        return hashlib.sha256(config_json.encode()).hexdigest()

    def save_to_file(self, path: str, format: str = "json") -> None:
        """Save configuration to file"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                f.write(self.to_json())
        elif format == "yaml":
            with open(path, "w") as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def load_from_file(path: str) -> "PipelineConfig":
        """Load configuration from file"""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path_obj.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif path_obj.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        return PipelineConfig.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary"""
        config = PipelineConfig()

        if "dedup" in data:
            config.dedup = DeduplicationConfig(**data["dedup"])
        if "diversity" in data:
            config.diversity = DiversityConfig(**data["diversity"])
        if "selection" in data:
            config.selection = SelectionConfig(**data["selection"])
        if "curriculum" in data:
            config.curriculum = CurriculumConfig(**data["curriculum"])
        if "io" in data:
            config.io = IOConfig(**data["io"])
        if "validation" in data:
            config.validation = ValidationConfig(**data["validation"])
        if "reproducibility" in data:
            config.reproducibility = ReproducibilityConfig(**data["reproducibility"])
        if "ablation" in data:
            config.ablation = AblationConfig(**data["ablation"])
        if "stages" in data:
            config.stages = {k: StageConfig(**v) for k, v in data["stages"].items()}
        if "created_at" in data:
            config.created_at = data["created_at"]
        if "pipeline_version" in data:
            config.pipeline_version = data["pipeline_version"]
        if "config_name" in data:
            config.config_name = data["config_name"]

        return config

    def validate(self) -> tuple[bool, List[str]]:
        """Validate configuration for consistency"""
        errors = []

        # Check dedup config
        if not (0.0 <= self.dedup.near_dedup_threshold <= 1.0):
            errors.append("dedup.near_dedup_threshold must be in [0.0, 1.0]")

        # Check diversity config
        if self.diversity.rare_token_boost <= 0:
            errors.append("diversity.rare_token_boost must be > 0")
        if self.diversity.tail_token_boost <= 0:
            errors.append("diversity.tail_token_boost must be > 0")

        # Check validation config
        if not (0.0 <= self.validation.validation_tolerance <= 0.1):
            errors.append("validation.validation_tolerance should be in [0.0, 0.1]")

        # Check stage configs
        for stage_name, stage_config in self.stages.items():
            if stage_config.target_tokens <= 0:
                errors.append(f"stages[{stage_name}].target_tokens must be > 0")

        return len(errors) == 0, errors
